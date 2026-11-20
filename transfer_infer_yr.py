# -*- coding: utf-8 -*-
"""
Basin-scale zero-shot transfer: the frozen final Inner Mongolia weights are applied
to 17 mainstem stations. Standardization uses the first 18 months of target-station history,
conformal calibration the following 12 months, and evaluation the 2024 period; per station and
target RMSE and persistence-anchored skill at H1-H3 and H7/H14/H30, plus pooled AUC.
"""
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from qrcqr.config import Config  # noqa: E402
from qrcqr.model import QRNN  # noqa: E402
from qrcqr.utils import build_adjacency  # noqa: E402


def cdf_eval(xp, u):
    xp = np.maximum.accumulate(xp, axis=-1)
    k = np.sum(xp <= u[..., None], axis=-1) - 1
    k = np.clip(k, 0, NQ - 2)
    i1 = np.minimum(k + 1, NQ - 1)
    x0 = np.take_along_axis(xp, k[..., None], axis=-1)[..., 0]
    x1 = np.take_along_axis(xp, i1[..., None], axis=-1)[..., 0]
    qg = np.broadcast_to(QS, xp.shape)
    f0 = np.take_along_axis(qg, k[..., None], axis=-1)[..., 0]
    f1 = np.take_along_axis(qg, i1[..., None], axis=-1)[..., 0]
    return np.clip(f0 + (u - x0) / np.maximum(x1 - x0, 1e-12) * (f1 - f0), 0, 1)


cfg = Config()
cfg.FEAT_IDX = (0, 1, 2, 3, 5, 6, 7, 8)
QS = np.arange(0.05, 1.0, 0.05)
NQ = 19
STD = [6.0, 0.15, 0.10, 2.0]
TG = ["CODMn", "NH3-N", "TP", "TN"]
WEIGHTS = "nm9_phys"
COORDS = {
    "唐乃亥": (100.841, 35.979), "大河家": (102.815, 36.325), "玛曲": (102.073, 33.998),
    "青城桥": (103.826, 36.059), "什川桥": (103.993, 36.151), "扶和桥": (105.607, 33.327),
    "靖远桥": (104.90, 36.55), "五佛寺": (104.294, 37.169), "中卫下河沿(新墩)": (105.163, 37.479),
    "金沙湾": (105.298, 37.481), "银古公路桥": (106.298, 38.447), "叶盛公路桥": (106.235, 38.523),
    "平罗黄河大桥": (106.67, 38.814), "柏树坪": (110.712, 37.448), "碛塄": (110.983, 38.967),
    "小浪底水库(小浪底)": (112.401, 34.921), "济南泺口": (117.005, 36.728)}

d = np.load(os.path.join(BASE, "dataset_yr_transfer.npz"), allow_pickle=True)
stations = [str(s) for s in d["stations"]]
n_st = len(stations)
L, H = int(d["L"]), int(d["H"])
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def load_x(key):
    x = d[key][:, :, :, cfg.FEAT_IDX].astype(np.float32)
    x[:, :, :, 5] = np.log1p(x[:, :, :, 5])
    x[:, :, :, 6] = np.log1p(x[:, :, :, 6])
    x[np.isnan(x)] = 0
    return x


xh = load_x("x_train")
mxh = d["mx_train"][:, :, :, cfg.FEAT_IDX].astype(np.float32)
xte = load_x("x_test")
mxte = d["mx_test"][:, :, :, cfg.FEAT_IDX].astype(np.float32)
yte = d["y_test"].astype(np.float32)
myte = d["my_test"].astype(bool)

n_stats = int(len(xh) * 0.6)
xm = np.nanmean(np.where(mxh[:n_stats] > 0, xh[:n_stats], np.nan).reshape(-1, 8), axis=0)
xs = np.nanstd(np.where(mxh[:n_stats] > 0, xh[:n_stats], np.nan).reshape(-1, 8), axis=0) + 1e-6
yh = d["y_train"].astype(np.float32)
myh = d["my_train"].astype(bool)
yh[:, :, :, 1] = np.log1p(yh[:, :, :, 1])
yh[:, :, :, 2] = np.log1p(yh[:, :, :, 2])
ym = np.nanmean(np.where(myh[:n_stats] > 0, yh[:n_stats], np.nan).reshape(-1, 4), axis=0)
ys = np.nanstd(np.where(myh[:n_stats] > 0, yh[:n_stats], np.nan).reshape(-1, 4), axis=0) + 1e-6

xte_n = ((xte - xm) * mxte / xs).astype(np.float32)
anchor_raw = xte[:, :, -1, [4, 5, 6, 7]].copy()
anchor_raw[:, :, 1] = np.expm1(anchor_raw[:, :, 1])
anchor_raw[:, :, 2] = np.expm1(anchor_raw[:, :, 2])
anchor_ok = anchor_raw > 0
yl0 = ((anchor_raw - ym[None, None, :]) / ys[None, None, :]).astype(np.float32)

at = build_adjacency(stations, {**cfg.COORDS, **COORDS}, scale=100.0, cutoff=450.0).to(device)

pt_seeds, q_seeds = [], []
xb = torch.from_numpy(xte_n).to(device)
ylb = torch.from_numpy(yl0).to(device)
for seed in range(3):
    model = QRNN(cfg.f_main, 4, n_st, at, cfg.TGT_FEATPOS, H=H, NQ=NQ,
                 hid=cfg.hid, dual=cfg.dual, enc=cfg.enc, gated=cfg.gated,
                 pgate=cfg.pgate, group_gate=cfg.group_gate, gmode=cfg.gmode,
                 dec=cfg.dec, dec_cell=cfg.dec_cell).to(device)
    sd = torch.load(os.path.join(BASE, "tmp", "weights_%s_seed%d.pt" % (WEIGHTS, seed)),
                    map_location=device)
    sd.pop("at", None)
    model.load_state_dict(sd, strict=False)
    model.at = at
    model.eval()
    with torch.no_grad():
        q, pt = model(xb, yl0=ylb, need_point=True)
    pt_seeds.append(pt.cpu().numpy())
    q_seeds.append(q.cpu().numpy())
    print("seed", seed, "done", flush=True)

pt_n = np.mean(pt_seeds, axis=0)
q_n = np.mean(q_seeds, axis=0)


def inv(a, ti):
    v = a * ys[ti] + ym[ti]
    return np.expm1(v) if ti in (1, 2) else v


pt_raw = np.stack([inv(pt_n[:, :, ti], ti) for ti in range(4)], axis=2)
q_raw = np.stack([inv(q_n[:, :, ti, :, :], ti) for ti in range(4)], axis=2)
np.savez(os.path.join(BASE, "tmp", "transfer_yr_final_preds.npz"),
         pt_raw=pt_raw, q_raw=q_raw, pt_seed=np.stack(
             [np.stack([inv(s[:, :, ti], ti) for ti in range(4)], axis=2) for s in pt_seeds]),
         t_test=d["t_test"], stations=np.array(stations))

lines = ["# 全流域零样本迁移 v2 (冻结【最终】权重, 17干流站, 2024测试期)", ""]


def pooled(h):
    k = h - 1
    row = []
    for ti in range(4):
        m = myte[:, :, k, ti] & anchor_ok[:, :, ti] & np.isfinite(pt_raw[:, :, ti, k])
        e = (yte[:, :, k, ti][m] - pt_raw[:, :, ti, k][m]) ** 2
        ep = (yte[:, :, k, ti][m] - anchor_raw[:, :, ti][m]) ** 2
        sk = (1 - e.mean() / ep.mean()) * 100
        rm, rp = float(np.sqrt(e.mean())), float(np.sqrt(ep.mean()))
        s = "%s: %.3f/%.3f/%+.1f%%" % (TG[ti], rm, rp, sk)
        if h in (7, 14, 30):
            pe = 1 - cdf_eval(np.maximum.accumulate(q_raw[:, :, ti, :, k], axis=-1),
                              np.float32(STD[ti]))
            lab = (yte[:, :, k, ti] > STD[ti]) & m
            fa = roc_auc_score(lab[m].astype(int), pe[m]) if lab[m].sum() >= 10 else np.nan
            aa = roc_auc_score(lab[m].astype(int), anchor_raw[:, :, ti][m]) \
                if lab[m].sum() >= 10 else np.nan
            s += " 冻结AUC=%.3f 锚值AUC=%.3f" % (fa, aa)
        row.append(s)
    return "H%-2d %s" % (h, " | ".join(row))


lines.append("## 池化 RMSE模型/RMSE锚值/skill% 与 AUC")
for h in (7, 14, 30):
    lines.append(pooled(h))
lines.append("")
open(os.path.join(BASE, "tmp", "transfer_yr_final.md"), "w", encoding="utf-8").write(
    "\n".join(lines))
print("\n".join(lines))
print("\n已保存 transfer_yr_final.md / transfer_yr_final_preds.npz")
