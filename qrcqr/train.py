# -*- coding: utf-8 -*-
"""Training loop for QRNN-CQR (includes the optional left-censoring loss for ammonia)."""
import numpy as np
import torch

from .utils import loader, pinball, censored_nll, var_match


def train_model(model, data, config, device, cens_c=None, cens_T=0.1):
    x_tr, yl_tr, y_tr, mx_tr, my_tr, cens_tr, wf_tr = data["train"]
    x_va, yl_va, y_va, mx_va, my_va, cens_va, wf_va = data["val"]
    opt = torch.optim.Adam(model.parameters(), lr=config.LR,
                           weight_decay=config.WD)
    best, best_state, patience = 1e9, None, 0
    for ep in range(config.EPOCHS):
        model.train()
        for xb, ylb, yb, mxb, myb, censb, wfb in loader(
                x_tr, yl_tr, y_tr, mx_tr, my_tr, cens_tr, device, shuffle=True,
                wf=wf_tr):
            opt.zero_grad()
            yp = yb.permute(0, 1, 3, 2)
            mp = myb.permute(0, 1, 3, 2)
            cp = censb.permute(0, 1, 3, 2)
            w = (1.0 + config.lambda_ev * torch.relu(yp)) \
                if config.lambda_ev > 0 else None
            wm = w if w is not None else torch.ones_like(mp)
            if config.dual:
                qpred, pt = model(xb, yl0=ylb, need_point=True, wx_f=wfb)
                l = pinball(qpred, yp, mp, config.QS, w, cens=cp)
                mse_mask = mp * (1 - cp) * wm
                l = l + config.lambda_mse * ((pt - yp) ** 2 * mse_mask).sum() / \
                    mse_mask.sum().clamp(min=1)
                if config.lambda_var > 0:
                    l = l + config.lambda_var * var_match(pt, yp, ylb, mse_mask)
                qt = qpred
            else:
                pred = model(xb, yl0=ylb, wx_f=wfb)
                l = pinball(pred, yp, mp, config.QS, w, cens=cp)
                if config.lambda_mse > 0:
                    med = pred[:, :, :, 9, :]
                    mse_mask = mp * (1 - cp) * wm
                    l = l + config.lambda_mse * ((med - yp) ** 2 * mse_mask).sum() / \
                        mse_mask.sum().clamp(min=1)
                    if config.lambda_var > 0:
                        l = l + config.lambda_var * var_match(med, yp, ylb,
                                                              mse_mask)
                qt = pred
            if cens_c is not None:
                Q_nh3 = qt[:, :, 1, :, :].transpose(2, 3)  # (B,N,H,NQ)
                l_cens = censored_nll(Q_nh3, cens_c, T=cens_T)
                cnm = cp[:, :, 1, :] * mp[:, :, 1, :]
                l = l + (l_cens * cnm).sum() / cnm.sum().clamp(min=1)
            l.backward()
            opt.step()
        model.eval()
        vl = 0.0
        with torch.no_grad():
            for xb, ylb, yb, mxb, myb, censb, wfb in loader(
                    x_va, yl_va, y_va, mx_va, my_va, cens_va, device, wf=wf_va):
                yp = yb.permute(0, 1, 3, 2)
                mp = myb.permute(0, 1, 3, 2)
                cp = censb.permute(0, 1, 3, 2)
                w = (1.0 + config.lambda_ev * torch.relu(yp)) \
                    if config.lambda_ev > 0 else None
                wm = w if w is not None else torch.ones_like(mp)
                if config.dual:
                    qpred, pt = model(xb, yl0=ylb, need_point=True, wx_f=wfb)
                    vv = pinball(qpred, yp, mp, config.QS, w, cens=cp)
                    mse_mask = mp * (1 - cp) * wm
                    vv = vv + config.lambda_mse * ((pt - yp) ** 2 * mse_mask).sum() / \
                        mse_mask.sum().clamp(min=1)
                    if config.lambda_var > 0:
                        vv = vv + config.lambda_var * var_match(pt, yp, ylb,
                                                                mse_mask)
                    qt = qpred
                else:
                    pred = model(xb, yl0=ylb, wx_f=wfb)
                    vv = pinball(pred, yp, mp, config.QS, w, cens=cp)
                    if config.lambda_mse > 0:
                        med = pred[:, :, :, 9, :]
                        mse_mask = mp * (1 - cp) * wm
                        vv = vv + config.lambda_mse * ((med - yp) ** 2 * mse_mask).sum() / \
                            mse_mask.sum().clamp(min=1)
                        if config.lambda_var > 0:
                            vv = vv + config.lambda_var * var_match(med, yp, ylb,
                                                                    mse_mask)
                    qt = pred
                if cens_c is not None:
                    Q_nh3 = qt[:, :, 1, :, :].transpose(2, 3)
                    l_cens = censored_nll(Q_nh3, cens_c, T=cens_T)
                    cnm = cp[:, :, 1, :] * mp[:, :, 1, :]
                    vv = vv + (l_cens * cnm).sum() / cnm.sum().clamp(min=1)
                vl += vv.item() * len(xb)
        vl /= len(x_va)
        if vl < best and np.isfinite(vl):
            best, best_state, patience = vl, \
                {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= 8:
                break
    model.load_state_dict(best_state)
    return model


def train_model_phys(model, data, cfg, device, phys, cens_c=None, cens_T=0.1):
    """物理一致损失训练 (方案 1+2+3a/3b): 基础双头损失 + SP 氧平衡 + 平静期物理收缩
    + 分位数非交叉 + 非负浓度. 物理参数 (k1/k2/k3/theta/tau) 可学习, softplus 保符号."""
    import torch.nn as nn
    softplus = nn.functional.softplus
    x_tr, yl_tr, y_tr, mx_tr, my_tr, cens_tr, wf_tr = data["train"]
    x_va, yl_va, y_va, mx_va, my_va, cens_va, wf_va = data["val"]
    H, NQ = cfg.H, cfg.NQ
    # 物理参数 (可学习): p -> softplus 保正
    pw = torch.tensor([-2.25, -1.85, -2.25], device=device, requires_grad=True)  # k1,k2_20,k3
    pth = torch.tensor([-2.44], device=device, requires_grad=True)               # Arrhenius theta
    ptau = torch.tensor([-1.17] * 4, device=device, requires_grad=True)          # tau per target
    ph_r = torch.tensor([-2.0] * 4, device=device, requires_grad=True)           # 降雨敏感度 per target
    sig = torch.sigmoid
    opt = torch.optim.Adam(list(model.parameters()) + [pw, pth, ptau, ph_r],
                           lr=cfg.LR, weight_decay=cfg.WD)
    thr = torch.as_tensor(phys["thr"], device=device, dtype=torch.float32)
    stdD = phys["stdD"]
    logmask = torch.tensor([False, True, True, False], device=device)
    ym_t = torch.as_tensor(phys["ym"], device=device, dtype=torch.float32)
    ys_t = torch.as_tensor(phys["ys"], device=device, dtype=torch.float32)
    hidx = torch.arange(H, device=device, dtype=torch.float32) + 1.0          # (H,)

    def inv_raw(pt_b, ti):
        v = pt_b * ys_t[ti] + ym_t[ti]
        return torch.where(logmask[ti], torch.expm1(v), v)

    def batch_terms(idx, split):
        ph = phys[split]
        to = lambda a: torch.as_tensor(a, device=device)
        xb = to(x_tr[idx] if split == "train" else x_va[idx]).float()
        ylb = to(yl_tr[idx] if split == "train" else yl_va[idx]).float()
        yb = to(y_tr[idx] if split == "train" else y_va[idx]).float()
        myb = to(my_tr[idx] if split == "train" else my_va[idx]).float()
        cp = to(cens_tr[idx] if split == "train" else cens_va[idx]).float().permute(0, 1, 3, 2)
        D0 = torch.nan_to_num(to(ph["D0"][idx]).float())
        Dtgt = torch.nan_to_num(to(ph["Dtgt"][idx]).float())
        Dbar = torch.nan_to_num(to(ph["Dbar"][idx]).float())
        Dok = to(ph["Dok"][idx]).bool()
        Ttgt = torch.nan_to_num(to(ph["Ttgt"][idx]).float(), nan=20.0)
        climn = to(ph["climn"][idx]).float()
        calm = to(ph["calm"][idx]).bool()
        anc = to(ph["ancraw"][idx]).float()
        return xb, ylb, yb, myb, cp, D0, Dtgt, Dbar, Dok, Ttgt, climn, calm, anc

    def loss_fn(model, idx, split):
        (xb, ylb, yb, myb, cp, D0, Dtgt, Dbar, Dok, Ttgt, climn, calm, anc) =             batch_terms(idx, split)
        yp = yb.permute(0, 1, 3, 2)
        mp = myb.permute(0, 1, 3, 2)
        qpred, pt = model(xb, yl0=ylb, need_point=True)
        l = pinball(qpred, yp, mp, cfg.QS)
        mse_mask = mp
        l = l + cfg.lambda_mse * ((pt - yp) ** 2 * mse_mask).sum() /             mse_mask.sum().clamp(min=1)
        # 3a 非交叉
        if cfg.lam_nc > 0:
            l = l + cfg.lam_nc * (torch.relu(qpred[..., :NQ - 1, :]
                                             - qpred[..., 1:, :]) ** 2).mean()
        # 3b 非负 (点预测头, 归一化空间阈值 thr)
        if cfg.lam_nn > 0:
            viol = torch.relu(thr[None, None, :, None] - pt) ** 2
            l = l + cfg.lam_nn * (viol * mse_mask).sum() / mse_mask.sum().clamp(min=1)
        # D1: 频域分解损失 (慢变高权重, 快变低权重)
        if cfg.lam_freq > 0:
            w_slow = cfg.freq_slow_w
            w_fast = 1.0 - w_slow
            kernel = torch.ones(1, 1, 5, device=pt.device) / 5
            B2, S2, T2, H2 = pt.shape
            pt_2d = pt.permute(0, 1, 2, 3).reshape(-1, 1, H2)
            yp_2d = yp.permute(0, 1, 2, 3).reshape(-1, 1, H2)
            pt_s = torch.nn.functional.conv1d(pt_2d, kernel, padding=2).squeeze(1).reshape(B2, S2, T2, H2)
            yp_s = torch.nn.functional.conv1d(yp_2d, kernel, padding=2).squeeze(1).reshape(B2, S2, T2, H2)
            pt_f = pt - pt_s
            yp_f = yp - yp_s
            l_slow = ((pt_s - yp_s) ** 2 * mse_mask).sum() / mse_mask.sum().clamp(min=1)
            l_fast = ((pt_f - yp_f) ** 2 * mse_mask).sum() / mse_mask.sum().clamp(min=1)
            l = l + cfg.lam_freq * (w_slow * l_slow + w_fast * l_fast)
        # 方案 1: SP 氧平衡 (COD/NH3 预测 vs 观测氧亏动态)
        if cfg.lam_sp > 0:
            k1 = 0.5 * sig(pw[0])          # 0~0.5 mg/L/天 每单位CODMn
            k2_20 = 2.0 * sig(pw[1])       # 0~2 /天 复氧系数
            k3 = 0.5 * sig(pw[2])          # 0~0.5 /天 每单位NH3-N
            theta = 1.0 + 0.3 * sig(pth)   # 1.0~1.3 Arrhenius
            cod = torch.clamp(inv_raw(pt[:, :, 0, :], 0), 0.0, 50.0)
            nh3 = torch.clamp(inv_raw(pt[:, :, 1, :], 1), 0.0, 20.0)
            k2 = k2_20 * torch.pow(theta, torch.clamp(Ttgt - 20.0, -15, 15))
            dD_obs = torch.nan_to_num((Dtgt - D0.unsqueeze(-1)) / (hidx * stdD),
                                      nan=0.0, posinf=0.0, neginf=0.0)
            dD_phys = torch.nan_to_num(
                (k1 * cod + k3 * nh3 - k2 * Dbar) / stdD,
                nan=0.0, posinf=0.0, neginf=0.0)
            msp = Dok & (myb[:, :, :, 0] > 0) & (myb[:, :, :, 1] > 0)
            l = l + cfg.lam_sp * ((dD_obs - dD_phys) ** 2 * msp).sum() /                 msp.sum().clamp(min=1)
        # 方案 4: 降雨-区间宽度下界 (物理驱动的异方差)
        if cfg.lam_het > 0:
            a_r = 2.0 * torch.sigmoid(ph_r)                       # (4,) 每目标
            W = qpred[..., -1, :] - qpred[..., 0, :]              # (B,S,4,H)
            Rn = torch.as_tensor(phys[split]["Rn"][idx], device=device).float()
            short = torch.relu(a_r[None, None, :, None]
                               * Rn[:, :, None, None] - W) ** 2
            l = l + cfg.lam_het * short.mean()
        # 方案 2: 平静期物理收缩 (锚点衰减 + 气候态)
        if cfg.lam_decay > 0:
            rdecay = torch.exp(-hidx[None, :] / (0.5 + 40 * sig(ptau))[:, None])
            base = rdecay[None, None, :, :] * ylb[:, :, :, None]                 + (1 - rdecay[None, None, :, :]) * climn.permute(0, 1, 3, 2)
            l = l + cfg.lam_decay * (((pt - base) ** 2) * calm.permute(0, 1, 3, 2)).sum() /                 calm.permute(0, 1, 3, 2).sum().clamp(min=1)
        if cens_c is not None:
            Q_nh3 = qpred[:, :, 1, :, :].transpose(2, 3)
            l_c = censored_nll(Q_nh3, cens_c, T=cens_T)
            cnm = cp[:, :, 1, :] * mp[:, :, 1, :]
            l = l + (l_c * cnm).sum() / cnm.sum().clamp(min=1)
        return l

    n = len(x_tr)
    best, best_state, patience = 1e9, None, 0
    for ep in range(cfg.EPOCHS):
        model.train()
        perm = torch.randperm(n)
        for s in range(0, n, 16):
            idx = perm[s:s + 16]
            opt.zero_grad()
            l = loss_fn(model, idx, "train")
            l.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = 0.0
            for s in range(0, len(x_va), 16):
                idx = torch.arange(s, min(s + 16, len(x_va)))
                vl += loss_fn(model, idx, "val").item() * len(idx)
        vl /= len(x_va)
        if vl < best:
            best, best_state, patience = vl,                 {k: v.clone() for k, v in model.state_dict().items()}, 0
            best_pw = (pw.detach().clone(), pth.detach().clone(),
                       ptau.detach().clone())
        else:
            patience += 1
            if patience >= 8:
                break
    model.load_state_dict(best_state)
    pw.data, pth.data, ptau.data = best_pw
    sig = torch.sigmoid
    print("物理参数(最优轮): k1=%.4f k2_20=%.3f theta=%.3f k3=%.4f tau=%s"
          % ((0.5 * sig(pw[0])).item(), (2.0 * sig(pw[1])).item(),
             (1.0 + 0.3 * sig(pth)).item(), (0.5 * sig(pw[2])).item(),
             np.round((0.5 + 40 * sig(ptau)).detach().cpu().numpy(), 1)))
    return model
