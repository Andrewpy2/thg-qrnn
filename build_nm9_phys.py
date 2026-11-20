# -*- coding: utf-8 -*-
"""
Builds the physics-augmented dataset (*_phys.npz) from the base dataset npz: the
dissolved-oxygen equilibrium-deficit features consumed by the physics-consistency regularizer.
"""
import os

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
d = np.load(os.path.join(BASE, "dataset_neimenggu_9st.npz"), allow_pickle=True)
sts = [str(s) for s in d["stations"]]
S, L, H = len(sts), int(d["L"]), int(d["H"])
PANEL = os.path.join(BASE, "panel.csv")  # cleaned monitoring panel
panel = pd.read_csv(PANEL, encoding="utf-8-sig", parse_dates=["日期"])
DO = panel.pivot_table(index="日期", columns="站名", values="溶解氧(mg/L)").asfreq("D")
WT = panel.pivot_table(index="日期", columns="站名", values="水温(℃)").asfreq("D")


def dosat(T):
    return 14.652 - 0.41022 * T + 0.007991 * T ** 2 - 7.7774e-5 * T ** 3


Dser = {st: (dosat(WT[st]) - DO[st]) for st in sts}
Tser = {st: WT[st] for st in sts}

out = {}
for split in ["train", "val", "test"]:
    t = pd.to_datetime(d[f"t_{split}"])
    N = len(t)
    D0 = np.full((N, S), np.nan, np.float32)
    Dtgt = np.full((N, S, H), np.nan, np.float32)
    Ttgt = np.full((N, S, H), np.nan, np.float32)
    for r, ta in enumerate(t):
        a = Dser[sts[0]].index.get_loc(ta)
        for si, st in enumerate(sts):
            D0[r, si] = Dser[st].iloc[a]
            for h in range(H):
                Dtgt[r, si, h] = Dser[st].iloc[a + h + 1] if a + h + 1 < len(Dser[st]) else np.nan
                Ttgt[r, si, h] = Tser[st].iloc[a + h + 1] if a + h + 1 < len(Tser[st]) else np.nan
    Dbar = (D0[:, :, None] + Dtgt) / 2.0
    Dok = np.isfinite(D0)[:, :, None] & np.isfinite(Dtgt)
    out[f"D0_{split}"] = D0
    out[f"Dtgt_{split}"] = Dtgt.astype(np.float32)
    out[f"Dbar_{split}"] = np.nan_to_num(Dbar).astype(np.float32)
    out[f"Dok_{split}"] = Dok
    out[f"Ttgt_{split}"] = np.nan_to_num(Ttgt, nan=20.0).astype(np.float32)
    out[f"climn_{split}"] = np.zeros((N, S, H, 4), np.float32)
    out[f"calm_{split}"] = np.zeros((N, S, H, 4), bool)
    out[f"ancraw_{split}"] = np.zeros((N, S, 4), np.float32)

# ym/ys: 与 qrcqr 主流程同口径 (log空间, 训练集掩码统计)
FEAT = [0, 1, 2, 3, 5, 6, 7, 8]
x_tr = d["x_train"][:, :, :, FEAT].astype(np.float32)
y_tr = d["y_train"].astype(np.float32)
mx_tr = d["mx_train"][:, :, :, FEAT].astype(np.float32)
my_tr = d["my_train"].astype(np.float32)
y_tr[:, :, :, 1] = np.log1p(y_tr[:, :, :, 1])
y_tr[:, :, :, 2] = np.log1p(y_tr[:, :, :, 2])
ym = np.nanmean(np.where(my_tr > 0, y_tr, np.nan).reshape(-1, 4), axis=0)
ys = np.nanstd(np.where(my_tr > 0, y_tr, np.nan).reshape(-1, 4), axis=0) + 1e-6
thr = (0.0 - ym) / ys
stdD = float(np.nanstd(np.concatenate([out["Dtgt_train"][out["Dok_train"]],
                                       np.repeat(out["D0_train"][:, :, None], H, 2)[out["Dok_train"]]])))
out.update(ym=ym, ys=ys, thr=thr.astype(np.float32),
           stdD=np.float32(stdD), stations=np.array(sts))
np.savez(os.path.join(BASE, "dataset_neimenggu_9st_phys.npz"), **out)
print("已保存 dataset_neimenggu_9st_phys.npz | stdD=%.2f | Dok率 %.0f%%" % (
    stdD, out["Dok_train"].mean() * 100))
