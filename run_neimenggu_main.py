# -*- coding: utf-8 -*-
"""
Launcher for the Inner Mongolia reach (Yellow River) main model: injects the
station coordinates and GB 3838-2002 early-warning thresholds, then calls qrcqr.main without
modifying the shared configuration. Coordinates are approximate geographic estimates (about
5-15 km uncertainty) used only by the distance-kernel adjacency.
Usage: python run_neimenggu_main.py [dataset_neimenggu_9st.npz] [tag]
"""
import sys

import qrcqr.config as C

NMG_COORDS = {
    "磴口": (106.78, 40.33),          # 干流, 三盛公枢纽上游侧
    "三盛公": (106.87, 40.32),        # 干流, 三盛公水利枢纽
    "总排干入黄口": (108.60, 40.75),   # 排干入黄(乌梁素海退水, 三湖河口附近)
    "乌梁素海湖心": (108.68, 40.98),   # 乌梁素海湖心
    "黑柳子": (109.15, 40.82),        # 干流, 乌拉特前旗
    "大黑河入黄口": (111.25, 40.35),   # 大黑河入黄口(托克托)
    "卧佛下": (112.35, 40.93),        # 大黑河上游(乌兰察布)
    "喇嘛湾": (111.43, 40.40),        # 干流, 托克托喇嘛湾
    "头道拐": (111.50, 40.34),        # 干流, 出境控制断面
}
NMG_STD_U = {"高锰酸盐指数(mg/L)": 6.0, "氨氮(mg/L)": 0.15,
             "总磷(mg/L)": 0.10, "总氮(mg/L)": 2.0}

C.COORDS.update(NMG_COORDS)
C.STD_U.clear()
C.STD_U.update(NMG_STD_U)

from qrcqr.main import main  # noqa: E402

npz = sys.argv[1] if len(sys.argv) > 1 else "dataset_neimenggu.npz"
tag = sys.argv[2] if len(sys.argv) > 2 else "nm_final"

argv = [
    "1.0",                 # 0 lambda_mse
    tag,                   # 1 tag
    "1", "lstm", "1", "1", "0",   # 2-6 dual enc gated pgate group_gate
    npz,                   # 7 npz
    "full", "none", "0",   # 8-10 gmode wx_npz wxgate
    "0", "0", "0",         # 11-13 lambda_ev use_c hemb
    "1", "32",             # 14-15 dec hid
    "0", "gru", "0",       # 16-18 attn dec_cell dim_attn
    "none",                # 19 gate_fix
    "100", "450",          # 20-21 adj_scale adj_cutoff (河网尺度)
    "0.2",                 # 22 gate_temp
    "0", "0.1",            # 23-24 cens cens_T
    "0", "none", "0.0",  # 25-27 futwx wxf_npz lambda_var
    "1",                   # 28 blend (点预测岭回归集成, 论文最终口径)
    "0", "0", "0", "0", "0", "0",  # 29-34 物理损失(默认关, 与论文最终模型同)
    "0", "0",              # 35-36 频域损失(关)
]
main(argv)
