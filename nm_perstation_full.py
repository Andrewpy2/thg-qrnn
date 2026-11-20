# -*- coding: utf-8 -*-
"""
Per-station, per-target, per-horizon evaluation of the deployed THG-QRNN model
(point forecasts with the deployed blend + per-station rolling conformal intervals) against the
baseline prediction files. NOTE: baseline predictions are not part of this release;
this script documents the full evaluation protocol of the paper.
"""
import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score

BASE = os.path.dirname(os.path.abspath(__file__))
QS = np.arange(0.05, 1.0, 0.05)
NQ = 19
STD = [6.0, 0.15, 0.10, 2.0]
TG = ["CODMn", "NH3-N", "TP", "TN"]

d = np.load(os.path.join(BASE, "dataset_neimenggu_9st.npz"), allow_pickle=True)
o = np.load(os.path.join(BASE, "tmp", "preds_nm9_phys.npz"), allow_pickle=True)
sts = [str(s) for s in d["stations"]]
S = len(sts)
y = d["y_test"].astype(np.float32)
my = d["my_test"].astype(bool)
anchor = d["x_test"][:, :, -1, [5, 6, 7, 8]].astype(np.float32)
yva = d["y_val"].astype(np.float32)
mva = d["my_val"].astype(bool)
blend = o["pt_raw_blend"].astype(np.float64)
q_raw = o["q_raw"].astype(np.float64)
q_va = o["qv_raw"].astype(np.float64)

b = np.load(os.path.join(BASE, "tmp", "preds_baselines_nm9.npz"), allow_pickle=True)
a2 = np.load(os.path.join(BASE, "tmp", "preds_baselines_add_nm9.npz"), allow_pickle=True)
b3 = np.load(os.path.join(BASE, "tmp", "preds_baselines_nm9_2.npz"), allow_pickle=True)
q1 = np.load(os.path.join(BASE, "tmp", "preds_q_baselines_nm9.npz"), allow_pickle=True)
q2 = np.load(os.path.join(BASE, "tmp", "preds_q_baselines_more_nm9.npz"), allow_pickle=True)
q3 = np.load(os.path.join(BASE, "tmp", "preds_q_baselines_nm9_2.npz"), allow_pickle=True)
pe_svr_all = np.load(os.path.join(BASE, "tmp", "prob_export_q_nm9.npz"), allow_pickle=True)["pe_SVR"]

MODELS = ["persistence", "LSTM", "BiLSTM", "GRU", "RNN", "TCN", "PatchTST", "Transformer",
          "CNN-LSTM", "GCN", "STGCN", "XGBoost", "SVR", "ours"]
PT = {"persistence": np.repeat(anchor[:, :, :, None], 30, axis=-1)}
for k in ["LSTM", "BiLSTM", "GRU", "RNN", "TCN", "PatchTST", "Transformer"]:
    PT[k] = b[k].mean(0)
PT["CNN-LSTM"] = b3["CNN-LSTM"].mean(0)
PT["GCN"] = b3["GCN"].mean(0)
PT["STGCN"] = b3["STGCN"].mean(0)
PT["XGBoost"] = b3["XGBoost"]
PT["SVR"] = a2["svr"]
PT["ours"] = blend


def cdf(xp, u):
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


point_out = {t: {} for t in TG}
for ti, t in enumerate(TG):
    for si, st in enumerate(sts):
        for hi, h in enumerate((1, 2, 3)):
            k = h - 1
            row = {}
            for m in MODELS:
                p = PT[m]
                m_ = my[:, si, k, ti] & np.isfinite(p[:, si, ti, k]) & (anchor[:, si, ti] != 0)
                row[m] = round(float(np.sqrt(((y[:, si, k, ti][m_] - p[m_ if False else slice(None), si, ti, k][m_]) ** 2).mean())), 3) if m_.sum() >= 20 else "n/a"
            point_out[t][f"{st}|H{h}"] = row

# 概率: 各基线逐种子共形 -> 逐站AUC; ours=滚动
src = dict(q1)
src.update(q2)
src.update(q3)
PROB_MODELS = ["persistence", "LSTM", "BiLSTM", "GRU", "RNN", "TCN", "PatchTST", "Transformer",
               "CNN-LSTM", "GCN", "STGCN", "XGBoost", "SVR", "ours(滚动)"]
prob_out = {t: {} for t in TG}


def auc_cell(lab, pe, m):
    lm = lab[m]
    if lm.sum() < 10 or (m.sum() - lm.sum()) < 10:
        return "—"
    return round(float(roc_auc_score(lm.astype(int), pe[m])), 2)


for ti, t in enumerate(TG):
    # 深度基线共形(种子均值AUC)
    deep_adj = {}
    for nm in ["LSTM", "BiLSTM", "GRU", "RNN", "TCN", "PatchTST", "Transformer", "CNN-LSTM", "GCN", "STGCN"]:
        qv = src["q_va_%s" % nm][:, :, :, :, :, [6, 13, 29]].astype(np.float64)
        qt = src["q_%s" % nm][:, :, :, :, :, [6, 13, 29]].astype(np.float64)
        per_seed = []
        for s_ in range(3):
            adj = np.array(qt[s_], copy=True)
            for si in range(S):
                for j, hk in enumerate([6, 13, 29]):
                    mv = mva[:, si, hk, ti] > 0
                    nv = int(mv.sum())
                    if nv < 30:
                        continue
                    sc = yva[mv, si, hk, ti][:, None] - qv[s_][mv, si, ti, :, j]
                    Ssc = np.sort(sc, axis=0)
                    kq = np.minimum(np.ceil((nv + 1) * QS).astype(int) - 1, nv - 1)
                    adj[:, si, ti, :, j] += Ssc[kq, np.arange(NQ)][None, :]
            per_seed.append(np.maximum.accumulate(adj, axis=3))
        deep_adj[nm] = per_seed
    # XGB 共形
    qvx = src["q_va_XGBoost"].astype(np.float64)
    qtx = src["q_XGBoost"].astype(np.float64)
    adjx = np.array(qtx, copy=True)
    for si in range(S):
        for j, hk in enumerate([6, 13, 29]):
            mv = mva[:, si, hk, ti] > 0
            nv = int(mv.sum())
            if nv < 30:
                continue
            sc = yva[mv, si, hk, ti][:, None] - qvx[mv, si, ti, :, j]
            Ssc = np.sort(sc, axis=0)
            kq = np.minimum(np.ceil((nv + 1) * QS).astype(int) - 1, nv - 1)
            adjx[:, si, ti, :, j] += Ssc[kq, np.arange(NQ)][None, :]
    adjx = np.maximum.accumulate(adjx, axis=3)
    # ours 滚动 (逐站)
    W = 90
    roll_adj = {}
    for hk in [6, 13, 29]:
        adj = np.array(q_raw[:, :, :, :, hk][..., ], copy=True)[:, :, ti, :]  # (N,S,19)
        for si in range(S):
            for qi in range(NQ):
                s_va = yva[:, si, hk, ti] - q_va[:, si, ti, qi, hk]
                ok = mva[:, si, hk, ti] & np.isfinite(s_va)
                sc = s_va[ok]
                for i in range(adj.shape[0]):
                    hi_ = i - hk - 1
                    if hi_ >= 0:
                        s2 = y[:hi_ + 1, si, hk, ti] - q_raw[:hi_ + 1, si, ti, qi, hk]
                        s2 = s2[my[:hi_ + 1, si, hk, ti] & np.isfinite(s2)]
                    else:
                        s2 = np.array([])
                    pool = np.concatenate([sc, s2])[-W:]
                    if len(pool) >= 30:
                        kk = min(int(np.ceil((len(pool) + 1) * QS[qi])) - 1, len(pool) - 1)
                        adj[i, si, qi] = q_raw[i, si, ti, qi, hk] + np.sort(pool)[kk]
        roll_adj[hk] = np.maximum.accumulate(adj, axis=-1)
    for si, st in enumerate(sts):
        for hi, h in enumerate((7, 14, 30)):
            hk = h - 1
            m = my[:, si, hk, ti]
            lab = (y[:, si, hk, ti] > STD[ti]) & m
            row = {}
            row["persistence"] = auc_cell(lab, anchor[:, si, ti], m) if m.sum() else "—"
            for nm in deep_adj:
                aucs = [roc_auc_score(lab[m].astype(int), (1 - cdf(ps[:, si, ti, :, hi], np.float32(STD[ti])))[m])
                        if lab[m].sum() >= 10 and (m.sum() - lab[m].sum()) >= 10 else np.nan
                        for ps in deep_adj[nm]]
                v = np.nanmean(aucs) if not np.all(np.isnan(aucs)) else np.nan
                row[nm] = round(float(v), 2) if np.isfinite(v) else "—"
            pe_x = 1 - cdf(adjx[:, si, ti, :, hi], np.float32(STD[ti]))
            row["XGBoost"] = auc_cell(lab, pe_x, m) if m.sum() else "—"
            pe_s = pe_svr_all[:, si, ti, hi]
            row["SVR"] = auc_cell(lab, pe_s, m) if m.sum() else "—"
            pe_r = 1 - cdf(roll_adj[hk][:, si, :], np.float32(STD[ti]))
            row["ours(滚动)"] = auc_cell(lab, pe_r, m) if m.sum() else "—"
            prob_out[t][f"{st}|H{h}"] = row

json.dump({"point": point_out, "prob": prob_out, "stations": sts, "models": MODELS,
           "prob_models": PROB_MODELS},
          open(os.path.join(BASE, "tmp", "perstation_full.json"), "w", encoding="utf-8"),
          ensure_ascii=False)
print("saved tmp/perstation_full.json")
