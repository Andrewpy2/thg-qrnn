# -*- coding: utf-8 -*-
"""Utility functions: adjacency, normalization, conformal, CDF, metrics."""
import numpy as np
import torch


def build_adjacency(stations, coords, scale=20.0, cutoff=50.0):
    """距离指数核行归一化邻接矩阵 (torch, float32)."""
    lon = np.array([coords[s][0] for s in stations])
    lat = np.array([coords[s][1] for s in stations])
    dlon = (lon[None, :] - lon[:, None]) * 111.0 * np.cos(np.radians(lat.mean()))
    dlat = (lat[None, :] - lat[:, None]) * 111.0
    dist = np.sqrt(dlon ** 2 + dlat ** 2)
    A = np.exp(-dist / scale)
    A[dist > cutoff] = 0
    A = A / A.sum(axis=1, keepdims=True)
    return torch.from_numpy(A.astype(np.float32))


def fit_stats(x, mx):
    x = x.reshape(-1, x.shape[-1])
    mx = mx.reshape(-1, mx.shape[-1])
    mean = np.nanmean(np.where(mx > 0, x, np.nan), axis=0)
    std = np.nanstd(np.where(mx > 0, x, np.nan), axis=0) + 1e-6
    return mean, std


def normalize(x, m, mean, std):
    return (x - mean) * m / std


def loader(x, yl, y, mx, my, cens, device, bs=16, shuffle=False, wf=None):
    n = len(x)
    idx = torch.randperm(n) if shuffle else torch.arange(n)
    for i in range(0, n, bs):
        j = idx[i:i + bs]
        wfb = (torch.from_numpy(wf[j]).to(device) if wf is not None else None)
        yield (torch.from_numpy(x[j]).to(device),
               torch.from_numpy(yl[j]).to(device),
               torch.from_numpy(y[j]).to(device),
               torch.from_numpy(mx[j]).to(device),
               torch.from_numpy(my[j]).to(device),
               torch.from_numpy(cens[j]).to(device),
               wfb)


def var_match(pred, obs, anchor, mask):
    """方差匹配正则: 逐(目标,时距)惩罚 log(std(pred-anchor)) 与 log(std(obs-anchor)) 之差.

    点预测 = 锚点 + delta, 在弱可预测条件下 MSE/pinball 会把 delta 收缩到很小,
    预测曲线因此退化为"观测右移 H 天". 该正则约束预测偏离锚点的幅度贴近真实
    变化幅度, 直接针对这种"贴锚点"式滞后.
    """
    eps = 1e-4
    a = anchor.unsqueeze(-1)                        # (B,N,T,1)
    dp = (pred - a) * mask
    do = (obs - a) * mask
    n = mask.sum((0, 1)).clamp(min=2)               # (T,H)
    mup = dp.sum((0, 1)) / n
    vup = (dp ** 2).sum((0, 1)) / n - mup ** 2
    muo = do.sum((0, 1)) / n
    vuo = (do ** 2).sum((0, 1)) / n - muo ** 2
    sp = torch.sqrt(vup.clamp(min=eps * eps))
    so = torch.sqrt(vuo.clamp(min=eps * eps))
    return ((sp.log() - so.log()) ** 2).mean()


def wxf_stats(wxf_tr):
    """未来气象归一化统计量: 通道0=t2m 标准化; 其余=降雨类 log1p 后标准化."""
    n_wxf = wxf_tr.shape[-1]
    mu = np.zeros(n_wxf, dtype=np.float32)
    sd = np.ones(n_wxf, dtype=np.float32)
    mu[0], sd[0] = wxf_tr[..., 0].mean(), wxf_tr[..., 0].std() + 1e-6
    for c in range(1, n_wxf):
        v = np.log1p(wxf_tr[..., c])
        mu[c], sd[c] = v.mean(), v.std() + 1e-6
    return mu, sd


def wxf_normalize(wxf, mu, sd):
    a = wxf.copy()
    a[..., 0] = (a[..., 0] - mu[0]) / sd[0]
    for c in range(1, a.shape[-1]):
        a[..., c] = (np.log1p(a[..., c]) - mu[c]) / sd[c]
    return a.astype(np.float32)


def pinball(pred, y, my, qs, w=None, cens=None):
    q = torch.as_tensor(qs, device=pred.device).view(1, 1, 1, -1, 1)
    u = y.unsqueeze(3) - pred
    loss = torch.maximum(q * u, (q - 1) * u)
    mask = my.unsqueeze(3)
    if cens is not None:
        mask = mask * (1 - cens.unsqueeze(3))
    if w is not None:
        wq = w.unsqueeze(3)
        return (loss * mask * wq).sum() / (mask * wq).sum().clamp(min=1)
    return (loss * mask).sum() / mask.sum().clamp(min=1)


def censored_nll(q_nq, c, T=0.1, eps=1e-6):
    """左删失负对数似然 -log F(c)。

    q_nq: (..., NQ)，最后一维为单调分位网格（归一化空间）；
    c: 标量，归一化空间中的删失点（如 log1p(检出限) 标准化后）。
    F(c) 用 19 个分位处的 soft 指示平均近似（无需显式单调头）。
    """
    F = torch.sigmoid((c - q_nq) / T).mean(dim=-1)
    return -torch.log(F.clamp(min=eps))


def cdf_eval(xp, u, qs, nq):
    """由单调分位网格线性插值求 P(Y<=u)。xp: (..., nq); u: (...)."""
    xp = np.maximum.accumulate(xp, axis=-1)
    k = np.sum(xp <= u[..., None], axis=-1) - 1
    k = np.clip(k, 0, nq - 2)
    i0, i1 = k, np.minimum(k + 1, nq - 1)
    x0 = np.take_along_axis(xp, i0[..., None], axis=-1)[..., 0]
    x1 = np.take_along_axis(xp, i1[..., None], axis=-1)[..., 0]
    qg = np.broadcast_to(qs, xp.shape)
    f0 = np.take_along_axis(qg, i0[..., None], axis=-1)[..., 0]
    f1 = np.take_along_axis(qg, i1[..., None], axis=-1)[..., 0]
    alpha = (u - x0) / np.maximum(x1 - x0, 1e-12)
    F = f0 + alpha * (f1 - f0)
    lo_slope = (qs[1] - qs[0]) / np.maximum(xp[..., 1] - xp[..., 0], 1e-12)
    hi_slope = (qs[-1] - qs[-2]) / np.maximum(xp[..., -1] - xp[..., -2], 1e-12)
    F = np.where(u < xp[..., 0], qs[0] + lo_slope * (u - xp[..., 0]), F)
    F = np.where(u > xp[..., -1], qs[-1] + hi_slope * (u - xp[..., -1]), F)
    return np.clip(F, 0.0, 1.0)


def conformal(q_v, q_t, yv, mv, qs, nq, min_cal=30):
    """逐 (站, 目标, 时距, 分位) conformal 校准. q_v/q_t: (n, n_st, n_t, nq, H)."""
    n = q_t.shape[0]
    n_st, n_t, _, H = q_t.shape[1:]
    adj = np.empty_like(q_t)
    pooled = {}
    for ti in range(n_t):
        for h in range(H):
            for qi in range(nq):
                sc = yv[:, :, h, ti] - q_v[:, :, ti, qi, h]
                m = mv[:, :, h, ti] > 0
                pooled[(ti, h, qi)] = np.sort(sc[m])
    fallback = 0
    for si in range(n_st):
        for ti in range(n_t):
            for h in range(H):
                for qi in range(nq):
                    sc = yv[:, si, h, ti] - q_v[:, si, ti, qi, h]
                    m = mv[:, si, h, ti] > 0
                    srt = np.sort(sc[m]) if m.sum() >= min_cal else pooled[(ti, h, qi)]
                    fallback += (m.sum() < min_cal)
                    k = min(int(np.ceil((len(srt) + 1) * qs[qi])) - 1, len(srt) - 1)
                    adj[:, si, ti, qi, h] = q_t[:, si, ti, qi, h] + srt[k]
    return np.maximum.accumulate(adj, axis=3), fallback


def inv_transform(p, ti, mean, std, log_tgt):
    p = p * std[ti] + mean[ti]
    if ti in log_tgt:
        p = np.expm1(p)
    return p


def p90_thresholds(ytr, mytr, n_st, n_t):
    p90 = np.zeros((n_st, n_t))
    for si in range(n_st):
        for ti in range(n_t):
            v = ytr[:, si, :, ti][mytr[:, si, :, ti] > 0]
            p90[si, ti] = np.percentile(v, 90) if len(v) else np.nan
    return p90


def event_rate(yraw, my, p90, hh):
    h = hh - 1
    m = my[:, :, h, :] > 0
    lab = (yraw[:, :, h, :] > p90[None, :, :]) * m
    return lab.sum(axis=(0, 1)) / m.sum(axis=(0, 1))


def point_metrics(pred, y, m):
    pv, yv = pred[m], y[m]
    mae = float(np.mean(np.abs(pv - yv)))
    rmse = float(np.sqrt(np.mean((pv - yv) ** 2)))
    smape = float(np.mean(2 * np.abs(pv - yv) /
                          (np.abs(pv) + np.abs(yv) + 1e-9))) * 100
    return mae, rmse, smape


def ridge_features(x, t_dates, n_st):
    """岭回归校正的特征: 最近7天全变量展平 + 年序谐波(2) + 站点哑变量.

    x: (n, S, L, V) 原始尺度窗口; 返回 (n*S, K*V+2+S), 与 y.reshape(n*S, T) 对齐.
    """
    K = 7
    n, S, L, V = x.shape
    feats = x[:, :, -K:, :].reshape(n, S, K * V)
    doy = t_dates.astype("datetime64[D]").astype(int) % 365
    ang = 2 * np.pi * doy / 365.0
    harm = np.stack([np.sin(ang), np.cos(ang)], axis=1)      # (n,2)
    oh = np.eye(S, dtype=np.float32)
    X = np.concatenate([feats,
                        np.broadcast_to(harm[:, None, :], (n, S, 2)),
                        np.broadcast_to(oh[None], (n, S, S))], axis=-1)
    return X.reshape(n * S, -1)
