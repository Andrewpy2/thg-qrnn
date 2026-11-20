# -*- coding: utf-8 -*-
"""
Leave-one-station-out generalization experiment under the main-model protocol:
the model is retrained with one monitoring station withheld and evaluated on that station.
"""
import os

import numpy as np
import pandas as pd
import torch

from qrcqr.config import Config, COORDS, BASE
from qrcqr.model import QRNN
from qrcqr.train import train_model, train_model_phys
from qrcqr.utils import build_adjacency, fit_stats, normalize, inv_transform

FEAT_NAMES = ["水温", "pH", "溶解氧", "电导率", "高锰酸盐", "氨氮", "总磷", "总氮"]
TGT_NAMES = ["CODMn", "NH3-N", "TP", "TN"]


def main():
    cfg = Config()
    cfg.lam_sp = 0.3
    cfg.lam_nc = 0.05
    cfg.lam_nn = 0.05
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    COORDS.update({
        "磴口": (106.78, 40.33), "三盛公": (106.87, 40.32), "总排干入黄口": (108.60, 40.75),
        "乌梁素海湖心": (108.68, 40.98), "黑柳子": (109.15, 40.82), "大黑河入黄口": (111.25, 40.35),
        "卧佛下": (112.35, 40.93), "喇嘛湾": (111.43, 40.40), "头道拐": (111.50, 40.34)})
    cfg.npz = "dataset_neimenggu_9st.npz"
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

    def anchor(x):
        a = x[:, :, -1, cfg.TGT_FEATPOS]
        return ((a - ym[None, None, :]) / ys[None, None, :]).astype(np.float32)

    yl_tr_n, yl_va_n, yl_te_n = anchor(x_tr), anchor(x_va), anchor(x_te)
    at = build_adjacency(stations, COORDS, scale=100.0, cutoff=450.0).to(device)
    yte_raw = d["y_test"][:, :, :, cfg.TGT_IDS].astype(np.float32)
    myte = d["my_test"][:, :, :, cfg.TGT_IDS].astype(np.float32)

    rows = []
    for k in range(n_st):
        my_tr_k = my_tr.copy()
        my_tr_k[:, k, :, :] = 0
        my_va_k = my_va.copy()
        my_va_k[:, k, :, :] = 0
        pt_seeds = []
        for seed in cfg.SEEDS:
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = QRNN(cfg.f_main, n_t, n_st, at, cfg.TGT_FEATPOS, H=H, NQ=NQ,
                         hid=cfg.hid, dual=cfg.dual, enc=cfg.enc, gated=cfg.gated,
                         pgate=cfg.pgate, group_gate=cfg.group_gate,
                         gmode=cfg.gmode, dec=cfg.dec,
                         dec_cell=cfg.dec_cell).to(device)
            data = {"train": (x_tr_n, yl_tr_n, y_tr_n, mx_tr, my_tr_k,
                              np.zeros_like(y_tr_n), None),
                    "val": (x_va_n, yl_va_n, y_va_n, mx_va, my_va_k,
                            np.zeros_like(y_va_n), None)}
        ph_npz = np.load(os.path.join(BASE, cfg.npz[:-4] + "_phys.npz"), allow_pickle=True)
        phys = {"stdD": float(ph_npz["stdD"]),
                "ym": torch.tensor(ph_npz["ym"], device=device, dtype=torch.float32),
                "ys": torch.tensor(ph_npz["ys"], device=device, dtype=torch.float32),
                "thr": torch.tensor(ph_npz["thr"], device=device, dtype=torch.float32)}
        wz = np.load(os.path.join(BASE, "wx.npz"), allow_pickle=True)
        rstd = float(np.nanstd(wz["wx_train"][..., 3, -1])) + 1e-6
        for sname in ["train", "val"]:
            phys[sname] = {k[:-(len(sname) + 1)]: ph_npz[k] for k in ph_npz.files if k.endswith("_" + sname)}
            phys[sname]["Rn"] = (wz[f"wx_{sname}"][..., 3, -1].astype(np.float32) / rstd)
        model = train_model_phys(model, data, cfg, device, phys)
        model.eval()
        with torch.no_grad():
            xb = torch.from_numpy(x_te_n).to(device)
            ylb = torch.from_numpy(yl_te_n).to(device)
            pt = model(xb, yl0=ylb, need_point=True)[1].cpu().numpy()
        pt_seeds.append(np.stack([
                inv_transform(pt[:, :, ti], ti, ym, ys, cfg.LOG_TGT)
                for ti in range(n_t)], axis=2))
        pt_raw = np.mean(pt_seeds, axis=0)
        h = 6
        rec = [stations[k]]
        for ti in range(n_t):
            y = yte_raw[:, k, h, ti]
            m = myte[:, k, h, ti] > 0
            if m.sum() < 5:
                rec += [np.nan, np.nan, np.nan]
                continue
            pv, yv = pt_raw[:, k, ti, h][m], y[m]
            mae = float(np.mean(np.abs(pv - yv)))
            rmse = float(np.sqrt(np.mean((pv - yv) ** 2)))
            smape = float(np.mean(2 * np.abs(pv - yv) /
                                  (np.abs(pv) + np.abs(yv) + 1e-9))) * 100
            rec += [mae, rmse, smape]
        rows.append(rec)
        print("%2d/9 %s" % (k + 1, stations[k]))

    cols = ["站"]
    for tn in TGT_NAMES:
        cols += ["%s_MAE" % tn, "%s_RMSE" % tn, "%s_SMAPE" % tn]
    df = pd.DataFrame(rows, columns=cols)
    out = os.path.join(BASE, "tmp", "loso_results_nm_phys.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig", float_format="%.4f")
    print("\n已保存:", out)
    print("\n每站 LOSO 结果 (H=7):")
    print(df.round(3).to_string(index=False))
    print("\n平均 (13 站, 排除 NaN):")
    for tn in TGT_NAMES:
        for met in ["MAE", "RMSE", "SMAPE"]:
            col = "%s_%s" % (tn, met)
            print("  %-5s %-6s %.3f" % (tn, met, df[col].mean()))


if __name__ == "__main__":
    main()
