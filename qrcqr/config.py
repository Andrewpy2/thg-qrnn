# -*- coding: utf-8 -*-
"""Configuration for QRNN-CQR. All tunable and fixed settings live here."""
import os
from dataclasses import dataclass, field

import numpy as np

# Project root (parent of the qrcqr package); data and outputs are located relative to it
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

COORDS = {}  # station coordinates are injected by the experiment runner (see run_neimenggu_main.py)

# GB 3838-2002 lake/reservoir thresholds (early-warning limits)
STD_U = {"高锰酸盐指数(mg/L)": 6.0, "氨氮(mg/L)": 1.0,
         "总磷(mg/L)": 0.10, "总氮(mg/L)": 1.5}


@dataclass
class Config:
    # 命令行可调 (默认值 = 论文最终模型 v8blendfix2 的配置)
    lambda_mse: float = 1.0
    tag: str = "final"
    dual: bool = True
    enc: str = "lstm"
    gated: bool = True
    pgate: bool = True
    group_gate: bool = False
    npz: str = "dataset_neimenggu_9st.npz"
    gmode: str = "full"
    wx_npz: str = "none"
    wxgate: bool = False
    lambda_ev: float = 0.0
    use_c: bool = False
    hemb: bool = False
    dec: bool = True
    hid: int = 32
    attn: bool = False
    dec_cell: str = "gru"
    dim_attn: bool = False
    gate_fix: object = None
    adj_scale: float = 20.0
    adj_cutoff: float = 50.0
    gate_temp: float = 0.2
    cens: bool = False
    cens_T: float = 0.1
    # 可选扩展 (默认关闭; 论文最终模型未使用)
    futwx: bool = False
    wxf_npz: str = "none"
    lambda_var: float = 0.0
    # 点预测线性校正集成 (QRNN + 岭回归, 权重验证集逐目标逐时距选择)
    # 论文最终模型开启; 置 False 可得纯深度头 (架构层消融口径)
    blend: bool = True
    # 物理一致损失 (默认 0 = 关闭; 方案 1/2/3a/3b)
    lam_sp: float = 0.0        # Streeter-Phelps 氧平衡软约束
    lam_decay: float = 0.0     # 平静期物理收缩 (锚点衰减+气候态)
    lam_nc: float = 0.0        # 分位数非交叉
    lam_nn: float = 0.0        # 非负浓度
    lam_het: float = 0.0       # 降雨-区间宽度下界 (方案4)
    lam_freq: float = 0.0      # 频域分解损失权重 (D1)
    freq_slow_w: float = 0.7   # 慢变分量权重 (快变 = 1 - slow_w)

    # 固定设置
    L: int = 30
    H: int = 30
    NQ: int = 19
    EPOCHS: int = 60
    LR: float = 1e-3
    WD: float = 1e-4
    SEEDS: tuple = (0, 1, 2)
    HORIZONS: tuple = (7, 14, 30)
    ALPHA: float = 0.10
    MIN_CAL: int = 30
    FEAT_IDX: tuple = (0, 1, 2, 3, 5, 6, 7, 8)
    TGT_IDS: tuple = (0, 1, 2, 3)
    TGT_FEATPOS: tuple = (4, 5, 6, 7)
    TGT_FEAT_RAW: tuple = (5, 6, 7, 8)
    LOG_TGT: set = field(default_factory=lambda: {1, 2})
    QS: np.ndarray = field(default_factory=lambda: np.arange(0.05, 1.0, 0.05))
    COORDS: dict = field(default_factory=lambda: COORDS)
    STD_U: dict = field(default_factory=lambda: STD_U)

    @property
    def wx_on(self) -> bool:
        return bool(self.wx_npz) and self.wx_npz not in ("none", "None", "-")

    @property
    def n_wx(self) -> int:
        return 4 if self.wx_on else 0

    @property
    def f_main(self) -> int:
        return len(self.FEAT_IDX)

    @classmethod
    def from_args(cls, argv=None):
        import sys
        a = list(sys.argv[1:] if argv is None else argv)

        def g(i, default):
            return a[i] if len(a) > i else default

        cfg = cls(
            lambda_mse=float(g(0, 1.0)),
            tag=g(1, "final"),
            dual=g(2, "1") == "1",
            enc=g(3, "lstm"),
            gated=g(4, "1") == "1",
            pgate=g(5, "1") == "1",
            group_gate=g(6, "0") == "1",
            npz=g(7, "dataset_neimenggu_9st.npz"),
            gmode=g(8, "full"),
            wx_npz=g(9, "none"),
            wxgate=g(10, "0") == "1",
            lambda_ev=float(g(11, 0.0)),
            use_c=g(12, "0") == "1",
            hemb=g(13, "0") == "1",
            dec=g(14, "1") == "1",
            hid=int(g(15, 32)),
            attn=g(16, "0") == "1",
            dec_cell=g(17, "gru"),
            dim_attn=g(18, "0") == "1",
            gate_fix=(None if g(19, "none") in ("none", "None", "-")
                      else float(g(19, "none"))),
            adj_scale=float(g(20, 20.0)),
            adj_cutoff=float(g(21, 50.0)),
            gate_temp=float(g(22, 0.2)),
            cens=g(23, "0") == "1",
            cens_T=float(g(24, 0.1)),
            futwx=g(25, "0") == "1",
            wxf_npz=g(26, "none"),
            lambda_var=float(g(27, 0.0)),
            blend=g(28, "1") == "1",
            lam_sp=float(g(29, 0.0)),
            lam_decay=float(g(30, 0.0)),
            lam_nc=float(g(31, 0.0)),
            lam_nn=float(g(32, 0.0)),
            lam_het=float(g(33, 0.0)),
            lam_freq=float(g(34, 0.0)),
            freq_slow_w=float(g(35, 0.7)),
        )
        # 特征扩展: g(36)=1 加入浊度(9特征); g(37)=1 再加入叶绿素(10特征)
        # (原实现误用 34/35, 与 lam_freq/freq_slow_w 位置冲突, 已修正)
        if g(36, "0") == "1":
            cfg.FEAT_IDX = (0, 1, 2, 3, 4, 5, 6, 7, 8)
            cfg.TGT_FEATPOS = (5, 6, 7, 8)
        if g(37, "0") == "1":
            cfg.FEAT_IDX = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
            cfg.TGT_FEATPOS = (5, 6, 7, 8)
        return cfg
