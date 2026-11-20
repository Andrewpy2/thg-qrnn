# -*- coding: utf-8 -*-
"""QRNN-CQR main driver: data loading -> 3-seed training -> conformal calibration -> evaluation -> export."""
import os
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from .config import BASE, Config
from .model import QRNN
from .train import train_model, train_model_phys
from .utils import (build_adjacency, cdf_eval, conformal, event_rate, fit_stats,
                    inv_transform, normalize, p90_thresholds, point_metrics,
                    var_match, wxf_normalize, wxf_stats)

# 水文角色分组: 0=入湖口, 1=湖心, 2=出湖口(三胜大队渡口), 3=沿岸
GIDX = {"兆河入湖区": 0, "入湖口渡口": 0, "北闸渡口(杭埠河)": 0,
        "双桥河入湖口": 0, "新河入湖区": 0, "施口": 0,
        "东半湖湖心": 1, "西半湖湖心": 1,
        "三胜大队渡口": 2,
        "忠庙": 3, "湖滨": 3, "黄麓": 3, "龟山": 3}
NH3_DL = 0.025


def _load(split, d, cfg):
    x = d[f"x_{split}"][:, :, :, cfg.FEAT_IDX].astype(np.float32)
    y = d[f"y_{split}"][:, :, :, cfg.TGT_IDS].astype(np.float32)
    mx = d[f"mx_{split}"][:, :, :, cfg.FEAT_IDX].astype(np.float32)
    my = d[f"my_{split}"][:, :, :, cfg.TGT_IDS].astype(np.float32)
    cens = np.zeros_like(y)
    if cfg.cens:
        cens[:, :, :, 1] = (y[:, :, :, 1] <= NH3_DL + 1e-6) * (my[:, :, :, 1] > 0)
    if cfg.wx_on:
        wx = np.load(os.path.join(BASE, cfg.wx_npz), allow_pickle=True)[f"wx_{split}"]
        wx = wx.astype(np.float32)
        x = np.concatenate([x, wx], axis=-1)
        mx = np.concatenate([mx, np.ones_like(wx)], axis=-1)
    logpos = [cfg.FEAT_IDX.index(raw_i) for raw_i in (6, 7)]   # NH3/TP 在所选特征中的位置
    x[:, :, :, logpos[0]] = np.log1p(x[:, :, :, logpos[0]])
    x[:, :, :, logpos[1]] = np.log1p(x[:, :, :, logpos[1]])
    y[:, :, :, 1] = np.log1p(y[:, :, :, 1])
    y[:, :, :, 2] = np.log1p(y[:, :, :, 2])
    x[np.isnan(x)] = 0
    y[np.isnan(y)] = 0
    return x, y, mx, my, cens


def main(argv=None):
    cfg = Config.from_args(argv)
    os.makedirs(os.path.join(BASE, "tmp"), exist_ok=True)
    out_log = os.path.join(BASE, "tmp", "pro_net_%s.txt" % cfg.tag)
    sys.stdout = open(out_log, "w", encoding="utf-8")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if cfg.futwx and not cfg.dec:
        raise SystemExit("futwx 需要解码器 (dec=1): 未来气象按 horizon 注入自回归解码器")
    print("device:", device, "| tag:", cfg.tag, "| dec:", cfg.dec,
          "| attn:", cfg.attn, "| dim_attn:", cfg.dim_attn,
          "| dec_cell:", cfg.dec_cell, "| hid:", cfg.hid,
          "| futwx:", cfg.futwx, "| lambda_var:", cfg.lambda_var)

    d = np.load(os.path.join(BASE, cfg.npz), allow_pickle=True)
    stations = list(d["stations"])
    targets = list(d["targets"])
    n_st = len(stations)
    n_t = len(cfg.TGT_IDS)
    H, L, NQ = cfg.H, cfg.L, cfg.NQ

    x_tr, y_tr, mx_tr, my_tr, cens_tr = _load("train", d, cfg)
    x_va, y_va, mx_va, my_va, cens_va = _load("val", d, cfg)
    x_te, y_te, mx_te, my_te, cens_te = _load("test", d, cfg)

    xm, xs = fit_stats(x_tr, mx_tr)
    ym, ys = fit_stats(y_tr, my_tr)
    F = x_tr.shape[-1]
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

    # ---- v8: 未来 H 天已知气象 (known-future inputs) ----
    wxf_tr_n = wxf_va_n = wxf_te_n = None
    if cfg.futwx:
        wd = np.load(os.path.join(BASE, cfg.wxf_npz), allow_pickle=True)
        wxf_tr = wd["wxf_train"].astype(np.float32)
        wxf_va = wd["wxf_val"].astype(np.float32)
        wxf_te = wd["wxf_test"].astype(np.float32)
        assert wxf_tr.shape[2] == H, "未来窗口长度与 H 不一致"
        mu, sd = wxf_stats(wxf_tr)
        wxf_tr_n = wxf_normalize(wxf_tr, mu, sd)
        wxf_va_n = wxf_normalize(wxf_va, mu, sd)
        wxf_te_n = wxf_normalize(wxf_te, mu, sd)
        N_WXF = wxf_tr.shape[-1]
        print("futwx: wxf %s | t2m mu/sd %.2f/%.2f" % (wxf_tr.shape, mu[0], sd[0]))
    if not cfg.futwx:
        N_WXF = 0

    at = build_adjacency(stations, cfg.COORDS, scale=cfg.adj_scale,
                         cutoff=cfg.adj_cutoff).to(device)
    group_ids = torch.tensor([GIDX.get(s, 0) for s in stations],
                             device=device) if cfg.group_gate else None
    data = {"train": (x_tr_n, yl_tr_n, y_tr_n, mx_tr, my_tr, cens_tr, wxf_tr_n),
            "val": (x_va_n, yl_va_n, y_va_n, mx_va, my_va, cens_va, wxf_va_n)}

    n_wx_actual = F - cfg.f_main
    f_in = cfg.f_main if cfg.wxgate else F
    preds_tr, preds_va, preds_te = [], [], []
    pts_tr, pts_va, pts_te = [], [], []
    alphas, ap_collect, aw_collect = [], [], []
    print("\n训练 3 seeds ...")
    for seed in cfg.SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = QRNN(f_in, n_t, n_st, at, cfg.TGT_FEATPOS, H=H, NQ=NQ,
                     hid=cfg.hid, dual=cfg.dual, enc=cfg.enc, gated=cfg.gated,
                     pgate=cfg.pgate, group_gate=cfg.group_gate,
                     gmode=cfg.gmode, wxgate=cfg.wxgate, n_wx=n_wx_actual,
                     use_c=cfg.use_c, hemb=cfg.hemb, dec=cfg.dec,
                     attn=cfg.attn, dec_cell=cfg.dec_cell,
                     dim_attn=cfg.dim_attn, group_ids=group_ids,
                     gate_fix=cfg.gate_fix, gate_temp=cfg.gate_temp,
                     futwx=cfg.futwx, n_wxf=N_WXF).to(device)
        print("model params:", sum(p.numel() for p in model.parameters()))
        cens_c = float((np.log1p(NH3_DL) - ym[1]) / ys[1]) if cfg.cens else None
        use_phys = (cfg.lam_sp > 0) or (cfg.lam_decay > 0)             or (cfg.lam_nc > 0) or (cfg.lam_nn > 0)
        if use_phys:
            _phys_fn = os.path.join(BASE, cfg.npz[:-4] + "_phys.npz")
            if not os.path.exists(_phys_fn):
                _phys_fn = os.path.join(BASE, "dataset_phys.npz")
            ph = np.load(_phys_fn, allow_pickle=True)
            phys = {"stdD": float(ph["stdD"]),
                    "ym": torch.tensor(ph["ym"], device=device, dtype=torch.float32),
                    "ys": torch.tensor(ph["ys"], device=device, dtype=torch.float32),
                    "thr": torch.tensor(ph["thr"], device=device, dtype=torch.float32)}
            wz = np.load(os.path.join(BASE, "wx.npz"),
                         allow_pickle=True)
            rstd = float(np.nanstd(wz["wx_train"][..., 3, -1])) + 1e-6
            for s in ["train", "val"]:
                phys[s] = {k[:-(len(s) + 1)]: ph[k] for k in ph.files
                           if k.endswith("_" + s)}
                phys[s]["Rn"] = (wz[f"wx_{s}"][..., 3, -1].astype(np.float32)
                                 / rstd)
            model = train_model_phys(model, data, cfg, device, phys,
                                     cens_c=cens_c, cens_T=cfg.cens_T)
        else:
            model = train_model(model, data, cfg, device, cens_c=cens_c,
                                cens_T=cfg.cens_T)
        model.eval()
        torch.save(model.state_dict(),
                   os.path.join(BASE, "tmp", "weights_%s_seed%d.pt" % (cfg.tag, seed)))
        with torch.no_grad():
            for name, xa, yla in [("tr", x_tr_n, yl_tr_n),
                                  ("va", x_va_n, yl_va_n),
                                  ("te", x_te_n, yl_te_n)]:
                xb = torch.from_numpy(xa).to(device)
                ylb = torch.from_numpy(yla).to(device)
                wfb = (torch.from_numpy(wxf_te_n if name == "te" else
                                        wxf_tr_n if name == "tr" else wxf_va_n)
                       .to(device) if cfg.futwx else None)
                p = model(xb, yl0=ylb, wx_f=wfb).cpu().numpy()
                pt = (model(xb, yl0=ylb, need_point=True, wx_f=wfb)[1].cpu().numpy()
                      if cfg.dual else None)
                if cfg.gated and name == "te":
                    alphas.append(model.last_alpha.cpu().numpy())
                if cfg.pgate and not (cfg.gated and cfg.gmode == "q") and name == "te":
                    ap_collect.append(model.last_ap.cpu().numpy())
                if cfg.wxgate and name == "te":
                    aw_collect.append(model.last_alpha_w.cpu().numpy())
                if name == "tr":
                    preds_tr.append(p)
                    if cfg.dual:
                        pts_tr.append(pt)
                elif name == "va":
                    preds_va.append(p)
                    if cfg.dual:
                        pts_va.append(pt)
                else:
                    preds_te.append(p)
                    if cfg.dual:
                        pts_te.append(pt)
        print("seed", seed, "done")

    ptr = np.mean(preds_tr, axis=0)
    pva = np.mean(preds_va, axis=0)
    pte = np.mean(preds_te, axis=0)
    qtr_raw = np.stack([inv_transform(ptr[:, :, ti], ti, ym, ys, cfg.LOG_TGT)
                        for ti in range(n_t)], axis=2)
    qv_raw = np.stack([inv_transform(pva[:, :, ti], ti, ym, ys, cfg.LOG_TGT)
                       for ti in range(n_t)], axis=2)
    qte_raw = np.stack([inv_transform(pte[:, :, ti], ti, ym, ys, cfg.LOG_TGT)
                        for ti in range(n_t)], axis=2)
    if cfg.dual:
        ptr_pt = np.mean(pts_te, axis=0)
        pt_raw = np.stack([inv_transform(ptr_pt[:, :, ti], ti, ym, ys, cfg.LOG_TGT)
                           for ti in range(n_t)], axis=2)
        pt_seed = np.stack([np.stack([
            inv_transform(pts_te[s][:, :, ti], ti, ym, ys, cfg.LOG_TGT)
            for ti in range(n_t)], axis=2) for s in range(len(pts_te))], axis=0)
    else:
        pt_raw, pt_seed = None, None

    # ---- v8: 点预测线性校正集成 (QRNN + 岭回归, 权重验证集逐(目标,时距)选择) ----
    ptb_raw = blend_w = None
    if cfg.blend and cfg.dual:
        from sklearn.linear_model import Ridge
        from .utils import ridge_features
        print("\n线性校正集成 (岭回归 + QRNN, 权重验证集选择) ...")
        xr_tr = d["x_train"].astype(np.float32)
        xr_va = d["x_val"].astype(np.float32)
        xr_te = d["x_test"].astype(np.float32)
        yr_tr = d["y_train"][:, :, :, cfg.TGT_IDS].astype(np.float32)
        yr_va = d["y_val"][:, :, :, cfg.TGT_IDS].astype(np.float32)
        Xtr = ridge_features(xr_tr, d["t_train"], n_st)
        Xva = ridge_features(xr_va, d["t_val"], n_st)
        Xte = ridge_features(xr_te, d["t_test"], n_st)
        fmu = np.nanmean(np.where(np.isfinite(Xtr), Xtr, np.nan), axis=0)
        fsd = np.nanstd(np.where(np.isfinite(Xtr), Xtr, np.nan), axis=0) + 1e-6
        Xtr = np.nan_to_num((Xtr - fmu) / fsd)
        Xva = np.nan_to_num((Xva - fmu) / fsd)
        Xte = np.nan_to_num((Xte - fmu) / fsd)
        yr_te = d["y_test"][:, :, :, cfg.TGT_IDS].astype(np.float32)
        okr_tr = (np.isfinite(yr_tr) & (yr_tr != 0)).reshape(-1, H, n_t)
        okr_va = (np.isfinite(yr_va) & (yr_va != 0)).reshape(-1, H, n_t)
        okr_te = (np.isfinite(yr_te) & (yr_te != 0)).reshape(-1, H, n_t)
        pt_va_seed = np.stack([np.stack([
            inv_transform(pts_va[s][:, :, ti], ti, ym, ys, cfg.LOG_TGT)
            for ti in range(n_t)], axis=2) for s in range(len(pts_va))], axis=0)
        pt_va_m = pt_va_seed.mean(0)                     # (Nva,S,4,H)
        n_va = yr_va.shape[0]
        ptb_raw = np.empty_like(pt_raw)
        blend_w = np.zeros((n_t, H))
        for ti in range(n_t):
            for h in range(H):
                okt = okr_tr[:, h, ti]; okv = okr_va[:, h, ti]
                oke = okr_te[:, h, ti]
                rg = Ridge(alpha=100.0).fit(Xtr[okt],
                                            yr_tr[:, :, h, ti].reshape(-1)[okt])
                r_va = np.full(n_va * n_st, np.nan, np.float32)
                r_va[okv] = rg.predict(Xva[okv])
                r_te = np.full(len(Xte), np.nan, np.float32)
                r_te[oke] = rg.predict(Xte[oke])
                q_va = pt_va_m[:, :, ti, h].reshape(-1)
                q_te = pt_raw[:, :, ti, h].reshape(-1)
                m_v = okv & np.isfinite(q_va) & np.isfinite(r_va)
                obs_v = yr_va[:, :, h, ti].reshape(-1)
                wbest, vbest = 0.0, np.inf
                for w in np.arange(0, 1.01, 0.1):
                    bl = w * q_va[m_v] + (1 - w) * r_va[m_v]
                    v = np.sqrt(np.mean((bl - obs_v[m_v]) ** 2))
                    if v < vbest:
                        vbest, wbest = v, w
                blend_w[ti, h] = wbest
                bl = wbest * q_te + (1 - wbest) * r_te.reshape(-1)
                ptb_raw[:, :, ti, h] = bl.reshape(pt_raw[:, :, ti, h].shape)
        print("融合权重 w(QRNN) 逐目标均值: %s" %
              " ".join("%.2f" % blend_w[ti].mean() for ti in range(n_t)))
    elif cfg.blend and not cfg.dual:
        print("警告: blend 需要 dual=1 (验证集点预测), 已跳过")

    yv_raw = d["y_val"][:, :, :, cfg.TGT_IDS].astype(np.float32)
    myv = d["my_val"][:, :, :, cfg.TGT_IDS].astype(np.float32)
    yte_raw = d["y_test"][:, :, :, cfg.TGT_IDS].astype(np.float32)
    myte = d["my_test"][:, :, :, cfg.TGT_IDS].astype(np.float32)
    ytr_raw = d["y_train"][:, :, :, cfg.TGT_IDS].astype(np.float32)
    mytr = d["my_train"][:, :, :, cfg.TGT_IDS].astype(np.float32)

    print("\nconformal 校准 ...")
    adj, fallback = conformal(qv_raw, qte_raw, yv_raw, myv, cfg.QS, NQ, cfg.MIN_CAL)
    print("局部校准不足兜底次数:", fallback)
    p90 = p90_thresholds(ytr_raw, mytr, n_st, n_t)

    print("\n事件率(窗内任一天>训练p90):")
    for hh in cfg.HORIZONS:
        tr = event_rate(ytr_raw, mytr, p90, hh)
        va = event_rate(yv_raw, myv, p90, hh)
        te = event_rate(yte_raw, myte, p90, hh)
        print("  H=%2d train=%s val=%s test=%s" % (
            hh, " ".join("%.0f%%" % (v * 100) for v in tr),
            " ".join("%.0f%%" % (v * 100) for v in va),
            " ".join("%.0f%%" % (v * 100) for v in te)))

    xr = d["x_test"].astype(np.float32)
    print("\n" + "=" * 100)
    if cfg.blend and ptb_raw is not None:
        print("点预测 (MAE / RMSE / SMAPE%) + 区间  [点预测=线性校正集成]")
    else:
        print("点预测 (MAE / RMSE / SMAPE%) + 区间")
    print("=" * 100)
    pt_show = ptb_raw if (cfg.blend and ptb_raw is not None) else pt_raw
    for ti, nm in enumerate(targets):
        for Hh in cfg.HORIZONS:
            h = Hh - 1
            p = pt_show[:, :, ti, h] if cfg.dual else qte_raw[:, :, ti, 9, h]
            yh = yte_raw[:, :, h, ti]
            m = myte[:, :, h, ti] > 0
            mae, rmse, smape = point_metrics(p, yh, m)
            lo, hi = adj[:, :, ti, 0, h], adj[:, :, ti, -1, h]
            picp = float(np.mean((yh[m] >= lo[m]) & (yh[m] <= hi[m]))) * 100
            pimaw = float(np.mean((hi[m] - lo[m]) / (np.abs(yh[m]) + 1e-9)))
            print("%-14s H=%2d MAE=%.3f RMSE=%.3f SMAPE=%.1f%% "
                  "PICP=%.1f%% PINAW=%.2f" % (nm, Hh, mae, rmse, smape, picp, pimaw))

    print("\n" + "=" * 100)
    print("超标概率 (国标阈值): AUC")
    print("=" * 100)
    for ti, nm in enumerate(targets):
        for Hh in cfg.HORIZONS:
            h = Hh - 1
            u = cfg.STD_U[nm]
            xp = adj[:, :, ti, :, h]
            ub = np.broadcast_to(np.float32(u), xp.shape[:2])
            pe = 1.0 - cdf_eval(xp, ub, cfg.QS, NQ)
            m = myte[:, :, h, ti] > 0
            lab = (yte_raw[:, :, h, ti] > u) * m
            auc = roc_auc_score(lab[m].astype(int), pe[m]) \
                if lab[m].sum() >= 5 else np.nan
            print("%-14s H=%2d AUC=%.3f" % (nm, Hh, auc))

    med_seed = np.stack([np.stack([
        inv_transform(preds_te[s][:, :, ti, 9, :], ti, ym, ys, cfg.LOG_TGT)
        for ti in range(n_t)], axis=2) for s in range(len(preds_te))], axis=0)
    pt_seed_val = None
    if cfg.dual:
        pt_seed_val = np.stack([np.stack([
            inv_transform(pts_va[s][:, :, ti], ti, ym, ys, cfg.LOG_TGT)
            for ti in range(n_t)], axis=2) for s in range(len(pts_va))], axis=0)

    q_va_seed = np.stack([np.stack([
        inv_transform(preds_va[s][:, :, ti], ti, ym, ys, cfg.LOG_TGT)
        for ti in range(n_t)], axis=2) for s in range(len(preds_va))], axis=0)
    q_te_seed = np.stack([np.stack([
        inv_transform(preds_te[s][:, :, ti], ti, ym, ys, cfg.LOG_TGT)
        for ti in range(n_t)], axis=2) for s in range(len(preds_te))], axis=0)

    out_npz = os.path.join(BASE, "tmp", "preds_%s.npz" % cfg.tag)
    np.savez(out_npz, q_raw=qte_raw, med_raw=qte_raw[:, :, :, 9, :], adj=adj,
             med_seed=med_seed, pt_raw=pt_raw, pt_seed=pt_seed,
             pt_seed_val=pt_seed_val, q_va_seed=q_va_seed, q_te_seed=q_te_seed,
             pt_raw_blend=ptb_raw, blend_w=blend_w,
             alpha=(np.mean(alphas, axis=0).mean(axis=0) if cfg.gated else None),
             alpha_p=(np.mean(ap_collect, axis=0).mean(axis=0)
                      if (cfg.pgate and ap_collect) else None),
             alpha_w=(np.mean(aw_collect, axis=0).mean(axis=0)
                      if (cfg.wxgate and aw_collect) else None),
             qv_raw=qv_raw, t_val=d["t_val"], p90=p90,
             lab=(yte_raw > p90[None, :, None, :]) * (myte > 0),
             t_test=d["t_test"], stations=np.array(stations),
             targets=np.array(targets))
    print("\n完成! 输出:", out_npz)


if __name__ == "__main__":
    main()
