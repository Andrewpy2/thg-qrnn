# -*- coding: utf-8 -*-
"""QRNN-CQR model: encoder + dual paths + target/horizon gating + recursive shared-head decoder."""
import torch
import torch.nn as nn


class EncTransformer(nn.Module):
    def __init__(self, f, hid, nlayers=2):
        super().__init__()
        self.proj = nn.Linear(f, hid)
        layer = nn.TransformerEncoderLayer(hid, 4, dim_feedforward=2 * hid,
                                           batch_first=True, dropout=0.1)
        self.enc = nn.TransformerEncoder(layer, nlayers)

    def forward(self, x):
        pos = torch.zeros(1, x.size(1), self.proj.out_features, device=x.device)
        return self.enc(self.proj(x) + pos), None


class EncCNNLSTM(nn.Module):
    def __init__(self, f, hid, ch=16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(f, ch, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(ch, ch, kernel_size=3, padding=1), nn.ReLU())
        self.lstm = nn.LSTM(ch, hid, batch_first=True)

    def forward(self, x):
        xc = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return self.lstm(xc)


class QRNN(nn.Module):
    def __init__(self, f, n_t, n_st, at, tgt_featpos, H=30, NQ=19, hid=32,
                 dual=False, enc="gru", gated=False, pgate=False,
                 group_gate=False, gmode="full", wxgate=False, n_wx=0,
                 use_c=False, hemb=False, dec=False, attn=False,
                 dec_cell="gru", dim_attn=False, group_ids=None,
                 gate_fix=None, gate_temp=1.0, futwx=False, n_wxf=0):
        super().__init__()
        self.dual = dual
        self.gated = gated
        self.pgate = pgate
        self.group_gate = group_gate
        self.gmode = gmode
        self.wxgate = wxgate
        self.n_wx = n_wx
        self.use_c = use_c
        self.hemb = hemb
        self.gate_fix = gate_fix
        self.gate_temp = gate_temp
        self.dec = None
        self.attn = nn.Linear(hid, 1) if attn else None
        self.dim_attn = nn.Parameter(torch.randn(f)) if dim_attn else None
        self.dec_cell = dec_cell
        self.tgt_featpos = tgt_featpos
        self.futwx = futwx
        self.wx_f_fc = nn.Linear(n_wxf, hid) if futwx else None
        self.H = H
        self.NQ = NQ
        bidir = enc == "bilstm"
        self.bidir = bidir
        if enc == "transformer":
            self.enc = EncTransformer(f, hid)
        elif enc == "cnn_lstm":
            self.enc = EncCNNLSTM(f, hid)
        elif enc in ("lstm", "bilstm"):
            self.enc = nn.LSTM(f, hid, batch_first=True, bidirectional=bidir)
        else:
            self.enc = nn.GRU(f, hid, num_layers=1, batch_first=True)
        edim = hid * (2 if bidir else 1)
        enc_in = edim + (hid if use_c and enc in ("lstm", "bilstm") else 0)
        self.g0 = nn.Linear(enc_in, hid) if gated else None
        self.gcn1 = nn.Linear(enc_in, hid)
        self.gcn2 = nn.Linear(hid, hid)
        self.gate = nn.Linear(2 * hid, n_t) if gated else None
        self.gate_p = nn.Linear(2 * hid, n_t * H) if (gated and pgate) else None
        self.h_emb = nn.Parameter(torch.randn(H, hid)) if hemb else None
        if wxgate:
            self.wx_enc = nn.LSTM(n_wx, hid, batch_first=True)
            self.wx_gate = nn.Linear(3 * hid, n_t)
            self.wx_gate_p = nn.Linear(3 * hid, n_t * H) if pgate else None
        if group_gate:
            self.gq_bias = nn.Parameter(torch.zeros(4, n_t))
            self.gp_bias = nn.Parameter(torch.zeros(4, n_t, H))
            self.group_ids = group_ids
        if dec:
            self.heads = None
            self.point_head = None
        else:
            if gated:
                self.heads = nn.ModuleList(
                    [nn.Linear(hid, NQ * H) for _ in range(n_st)])
            else:
                self.heads = nn.ModuleList(
                    [nn.Linear(hid, n_t * NQ * H) for _ in range(n_st)])
            if dual:
                self.point_head = (nn.Linear(hid, 1)
                                   if (gated and pgate and gmode != "q")
                                   else nn.Linear(hid, H) if gated
                                   else nn.Linear(hid, n_t * H))
        if dec:
            self.dec = (nn.LSTMCell(hid, hid) if dec_cell == "lstm"
                        else nn.GRUCell(hid, hid))
            self.dec_h_emb = nn.Parameter(torch.randn(H, hid))
            self.prev_fc = nn.Linear(1, hid)
            self.out = nn.Linear(hid, 1 + NQ)
        self.n_t, self.n_st = n_t, n_st
        self.register_buffer("at", at)

    def _anchor(self, x):
        return x[:, :, -1, torch.tensor(self.tgt_featpos, device=x.device)]

    def forward(self, x, yl0=None, need_point=False, wx_f=None):
        B, N, Ln, F = x.shape
        wx_e = None
        if wx_f is not None and self.wx_f_fc is not None:
            wx_e = self.wx_f_fc(wx_f)                  # (B,N,H,hid)
        if self.wxgate:
            f_main = F - self.n_wx
            x_main = x[..., :f_main]
            x_wx = x[..., f_main:]
        else:
            x_main = x
            x_wx = None
        xf = x_main.reshape(B * N, Ln, -1)
        if self.dim_attn is not None:
            xf = xf * torch.softmax(self.dim_attn, dim=0)
        out, hc = self.enc(xf)
        if self.attn is not None:
            w = torch.softmax(self.attn(out), dim=1)
            h = (out * w).sum(dim=1)
        elif self.use_c and isinstance(self.enc, nn.LSTM):
            h = torch.cat([out[:, -1], hc[1][0]], dim=-1)
        else:
            h = out[:, -1]
        yl = yl0 if yl0 is not None else self._anchor(x)
        if self.gated:
            hv = h.view(B, N, -1)
            g0 = torch.relu(self.g0(hv))
            g1 = torch.relu(self.gcn1(torch.einsum("ij,bjf->bif", self.at, hv)))
            g1 = self.gcn2(torch.einsum("ij,bjf->bif", self.at, g1))
            alpha_logits = self.gate(torch.cat([g0, g1], dim=-1))
            if self.group_gate:
                alpha_logits = alpha_logits + self.gq_bias[self.group_ids][None]
            alpha = torch.sigmoid(alpha_logits / self.gate_temp)
            if self.gate_fix is not None:
                alpha = torch.full_like(alpha, self.gate_fix)
            self.last_alpha = alpha.detach()
            gf = (alpha.unsqueeze(-1) * g1.unsqueeze(-2) +
                  (1 - alpha).unsqueeze(-1) * g0.unsqueeze(-2))
            if self.wxgate:
                wx_out, _ = self.wx_enc(x_wx.reshape(B * N, Ln, self.n_wx))
                wv = wx_out[:, -1].view(B, N, -1)
                gw_in = torch.cat([g0, g1, wv], dim=-1)
                alpha_w = torch.sigmoid(self.wx_gate(gw_in))
                self.last_alpha_w = alpha_w.detach()
                gf = gf + alpha_w.unsqueeze(-1) * wv.unsqueeze(-2)
            if self.dec is not None:
                M = B * N * self.n_t
                state = gf.reshape(M, -1)
                if self.dec_cell == "lstm":
                    state = (state, torch.zeros_like(state))
                ylf = yl.reshape(M, 1)
                prev = ylf
                ap = None
                if self.pgate and self.gate_p is not None:
                    ap = torch.sigmoid(self.gate_p(
                        torch.cat([g0, g1], dim=-1)).view(B, N, self.n_t, self.H)
                        / self.gate_temp)
                    if self.gate_fix is not None:
                        ap = torch.full_like(ap, self.gate_fix)
                    self.last_ap = ap.detach()
                pts, qs = [], []
                for hh in range(self.H):
                    ctx_h = 0.0
                    if ap is not None:
                        ctx_h = (ap[:, :, :, hh].unsqueeze(-1) * g1.unsqueeze(-2) +
                                 (1 - ap[:, :, :, hh]).unsqueeze(-1) * g0.unsqueeze(-2))
                        ctx_h = ctx_h.reshape(M, -1)
                    e = self.dec_h_emb[hh] + self.prev_fc(prev) + ctx_h
                    if wx_e is not None:
                        e = e + wx_e[:, :, hh].unsqueeze(2) \
                            .expand(B, N, self.n_t, -1).reshape(M, -1)
                    state = self.dec(e, state)
                    o = self.out(state[0] if self.dec_cell == "lstm" else state)
                    delta_p = o[:, 0]
                    delta_q = o[:, 1:]
                    full_p = ylf[:, 0] + delta_p
                    prev = full_p[:, None]
                    pts.append(full_p)
                    qs.append(ylf + delta_q)
                pt = torch.stack(pts, dim=-1).view(B, N, self.n_t, self.H)
                q = torch.stack(qs, dim=-1).view(B, N, self.n_t, self.NQ, self.H)
                if need_point and self.dual:
                    return q, pt
                return q
            if self.gmode == "p":
                gf = g0.unsqueeze(-2).expand(B, N, self.n_t, -1)
            delta = torch.stack([self.heads[s](gf[:, s]) for s in range(N)], dim=1)
            q = delta.view(B, N, self.n_t, self.NQ, self.H)
            res = yl.unsqueeze(-1).unsqueeze(-1) + q
            if need_point and self.dual:
                if self.gmode == "q":
                    g0t = g0.unsqueeze(-2).expand(B, N, self.n_t, -1)
                    pt = self.point_head(g0t)
                elif self.pgate:
                    ap = self.gate_p(torch.cat([g0, g1], dim=-1)).view(
                        B, N, self.n_t, self.H)
                    if self.group_gate:
                        ap = ap + self.gp_bias[self.group_ids][None]
                    ap = torch.sigmoid(ap)
                    self.last_ap = ap.detach()
                    gf_p = (ap.unsqueeze(-1) * g1.unsqueeze(-2).unsqueeze(-2) +
                            (1 - ap).unsqueeze(-1) * g0.unsqueeze(-2).unsqueeze(-2))
                    if self.wxgate:
                        alpha_wp = torch.sigmoid(self.wx_gate_p(gw_in)).view(
                            B, N, self.n_t, self.H)
                        gf_p = gf_p + alpha_wp.unsqueeze(-1) * wv.unsqueeze(-2).unsqueeze(-2)
                    if self.hemb:
                        gf_p = gf_p + self.h_emb.view(1, 1, 1, self.H, -1)
                    pt = self.point_head(gf_p).squeeze(-1)
                else:
                    pt = self.point_head(gf)
                return res, yl.unsqueeze(-1) + pt
            return res
        g = torch.relu(self.gcn1(torch.einsum("ij,bjf->bif", self.at,
                                              h.view(B, N, -1))))
        g = self.gcn2(torch.einsum("ij,bjf->bif", self.at, g))
        delta = torch.stack([self.heads[s](g[:, s]) for s in range(N)], dim=1)
        q = delta.view(B, N, self.n_t, self.NQ, self.H)
        res = yl.unsqueeze(-1).unsqueeze(-1) + q
        if need_point and self.dual:
            pt = self.point_head(g).view(B, N, self.n_t, self.H)
            return res, yl.unsqueeze(-1) + pt
        return res
