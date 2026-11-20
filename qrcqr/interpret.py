# -*- coding: utf-8 -*-
"""Feature-level interpretability: per-input ablation importance. For each input feature, the feature is set to its test-set mean and the resulting increase in point-forecast RMSE is recorded per target. Outputs: tmp/feature_importance.csv and figures/fig_feature_importance.png."""
import os

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

from .config import Config, COORDS, BASE
from .model import QRNN
from .train import train_model
from .utils import build_adjacency, fit_stats, normalize, inv_transform

FEAT_NAMES = ["水温", "pH", "溶解氧", "电导率", "高锰酸盐", "氨氮", "总磷", "总氮"]
TGT_NAMES = ["CODMn", "NH3-N", "TP", "TN"]


def main():
    cfg = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    d = np.load(os.path.join(BASE, cfg.npz), allow_pickle=True)
    stations = list(d["stations"])
    n_st, n_t = len(stations), len(cfg.TGT_IDS)
    H, L, NQ = cfg.H, cfg.L, cfg.NQ

    def load(split):
        x = d[f"x_{split}"][:, :, :, cfg.FEAT_IDX].astype(np.float32)
        y = d[f"y_{split}"][:, :, :, cfg.TGT_IDS].astype(np.float32)
        mx = d[f"mx_{split}"][:, :, :, cfg.FEAT_IDX].astype(np.float32)
        my = d[f"my_{split}"][:, :, :, cfg.TGT_IDS].astype(np.float32)
        x[:, :, :, 5] = np.log1p(x[:, :, :, 5])
        x[:, :, :, 6] = np.log1p(x[:, :, :, 6])
        y[:, :, :, 1] = np.log1p(y[:, :, :, 1])
        y[:, :, :, 2] = np.log1p(y[:, :, :, 2])
        x[np.isnan(x)] = 0
        y[np.isnan(y)] = 0
        return x, y, mx, my

    x_tr, y_tr, mx_tr, my_tr = load("train")
    x_va, y_va, mx_va, my_va = load("val")
    x_te, y_te, mx_te, my_te = load("test")
    xm, xs = fit_stats(x_tr, mx_tr)
    ym, ys = fit_stats(y_tr, my_tr)
    x_tr_n = normalize(x_tr, mx_tr, xm, xs).astype(np.float32)
    x_va_n = normalize(x_va, mx_va, xm, xs).astype(np.float32)
    x_te_n = normalize(x_te, mx_te, xm, xs).astype(np.float32)
    y_tr_n = normalize(y_tr, my_tr, ym, ys).astype(np.float32)
    y_va_n = normalize(y_va, my_va, ym, ys).astype(np.float32)
    y_te_n = normalize(y_te, my_te, ym, ys).astype(np.float32)

    def anchor(x):
        a = x[:, :, -1, cfg.TGT_FEATPOS]
        return ((a - ym[None, None, :]) / ys[None, None, :]).astype(np.float32)

    yl_tr_n, yl_va_n, yl_te_n = anchor(x_tr), anchor(x_va), anchor(x_te)
    at = build_adjacency(stations, COORDS).to(device)
    data = {"train": (x_tr_n, yl_tr_n, y_tr_n, mx_tr, my_tr),
            "val": (x_va_n, yl_va_n, y_va_n, mx_va, my_va)}

    yte_raw = d["y_test"][:, :, :, cfg.TGT_IDS].astype(np.float32)
    myte = d["my_test"][:, :, :, cfg.TGT_IDS].astype(np.float32)

    def forward_pt(model, x_in, yl_override=None):
        with torch.no_grad():
            xb = torch.from_numpy(x_in).to(device)
            yl_use = yl_te_n if yl_override is None else yl_override
            ylb = torch.from_numpy(yl_use).to(device)
            pt = model(xb, yl0=ylb, need_point=True)[1].cpu().numpy()
        return np.stack([inv_transform(pt[:, :, ti], ti, ym, ys, cfg.LOG_TGT)
                         for ti in range(n_t)], axis=2)

    def rmse(pred, ti, h):
        y = yte_raw[:, :, h, ti]
        m = myte[:, :, h, ti] > 0
        return float(np.sqrt(np.mean((pred[:, :, ti, h][m] - y[m]) ** 2)))

    imp_sum = np.zeros((len(FEAT_NAMES), n_t, len(cfg.HORIZONS)))
    anchor_imp_sum = np.zeros((n_t, len(cfg.HORIZONS)))
    yl_zero = np.zeros_like(yl_te_n)
    for seed in cfg.SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = QRNN(cfg.f_main, n_t, n_st, at, cfg.TGT_FEATPOS, H=H, NQ=NQ,
                     hid=cfg.hid, dual=cfg.dual, enc=cfg.enc, gated=cfg.gated,
                     pgate=cfg.pgate, group_gate=cfg.group_gate, gmode=cfg.gmode,
                     dec=cfg.dec, dec_cell=cfg.dec_cell).to(device)
        model = train_model(model, data, cfg, device)
        model.eval()
        base_pt = forward_pt(model, x_te_n)
        pt_anchor_abl = forward_pt(model, x_te_n, yl_override=yl_zero)
        for f in range(len(FEAT_NAMES)):
            xp = x_te_n.copy()
            xp[:, :, :, f] = 0.0
            pt_abl = forward_pt(model, xp)
            for ti in range(n_t):
                for hi, Hh in enumerate(cfg.HORIZONS):
                    r0 = rmse(base_pt, ti, Hh - 1)
                    r1 = rmse(pt_abl, ti, Hh - 1)
                    imp_sum[f, ti, hi] += r1 - r0
        for ti in range(n_t):
            for hi, Hh in enumerate(cfg.HORIZONS):
                r0 = rmse(base_pt, ti, Hh - 1)
                r1 = rmse(pt_anchor_abl, ti, Hh - 1)
                anchor_imp_sum[ti, hi] += r1 - r0
        print("seed", seed, "done")
    imp = imp_sum / len(cfg.SEEDS)
    anchor_imp = anchor_imp_sum / len(cfg.SEEDS)

    rows = []
    for f, fn in enumerate(FEAT_NAMES):
        for ti, tn in enumerate(TGT_NAMES):
            rows.append([fn, tn, imp[f, ti, 0], imp[f, ti, 1], imp[f, ti, 2]])
    df = pd.DataFrame(rows, columns=["输入指标", "目标", "H7", "H14", "H30"])
    df.to_csv(os.path.join(BASE, "tmp", "feature_importance.csv"),
              index=False, encoding="utf-8-sig", float_format="%.4f")
    print("已保存: tmp/feature_importance.csv")

    # 对照图: 4 面板, 每个目标一张 "锚点 vs 8特征" 的 RMSE 上升条形图 (H=7)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ti in range(n_t):
        ax = axes[ti // 2][ti % 2]
        vals = [anchor_imp[ti, 0]] + list(imp[:, ti, 0])
        labels = ["锚点(昨日值)"] + FEAT_NAMES
        colors = ["#C0392B"] + ["#5D6D7E"] * len(FEAT_NAMES)
        ypos = np.arange(len(labels))[::-1]
        ax.barh(ypos, vals, color=colors, edgecolor="white")
        ax.set_yticks(ypos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("RMSE 上升 (mg/L)")
        ax.set_title(TGT_NAMES[ti], fontsize=12, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
        for k, v in enumerate(vals):
            ax.text(v + 0.005, ypos[k], "%.3f" % v, va="center", fontsize=8)
    fig.suptitle("锚点 vs 30天历史特征 的消融重要性 (H=7, 3种子平均)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_fig = os.path.join(BASE, "figures", "fig_feature_importance.png")
    fig.savefig(out_fig, dpi=200)
    plt.close(fig)
    print("已保存:", out_fig)
    print("\n锚点消融 RMSE 上升 (H=7):")
    for ti, tn in enumerate(TGT_NAMES):
        print("  %-6s %+.3f" % (tn, anchor_imp[ti, 0]))


if __name__ == "__main__":
    main()
