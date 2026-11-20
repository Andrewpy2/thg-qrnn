# THG-QRNN: Adaptive Spatiotemporal Gating for Probabilistic Water Quality Forecasting

Reference implementation accompanying the manuscript. THG-QRNN is a quantile recurrent
neural network with a two-level, target- and forecast-horizon-conditioned spatial gating
mechanism, trained with a pinball loss plus physics-consistency and non-crossing
regularizers, and calibrated with split conformalized quantile regression.

## Repository layout

| Path | Role (manuscript section) |
|---|---|
| `qrcqr/model.py` | Model: encoder, dual paths, two-level target/horizon gating, recursive shared-head quantile decoder (Methods, model section) |
| `qrcqr/train.py` | Training loop: pinball + MSE + physics-consistency + non-crossing losses (Methods, training) |
| `qrcqr/main.py` | Driver: load -> 3-seed training -> split-conformal calibration -> evaluation -> export |
| `qrcqr/config.py` | All tunable and fixed settings (defaults = paper model) |
| `qrcqr/utils.py` | Distance-kernel adjacency, normalization, conformal calibration, CDF, metrics, deployed ridge blend |
| `qrcqr/interpret.py` | Gating-weight extraction and interpretation helpers |
| `build_nm9_phys.py` | Physics-augmented features: dissolved-oxygen equilibrium deficit (Methods, physics regularizer) |
| `run_neimenggu_main.py` | Main-model launcher (station coordinates + GB 3838-2002 thresholds injection) |
| `loso_nm_phys.py` | Leave-one-station-out generalization (Results, LOSO) |
| `transfer_infer_yr.py` | Zero-shot transfer to 17 mainstem stations (Results, transfer) |
| `nm_perstation_full.py` | Per-station, per-target, per-horizon evaluation protocol (Results) |

## Environment

Python >= 3.10 with `numpy pandas torch scikit-learn matplotlib`
(see `requirements.txt`). A single consumer-grade GPU is sufficient; CPU works but slower.

## Intended pipeline order

1. build the base dataset npz from the cleaned monitoring panel;
2. `python build_nm9_phys.py` -> physics-augmented `*_phys.npz`;
3. `python run_neimenggu_main.py dataset_neimenggu_9st.npz nm9_phys` -> training,
   conformal calibration, predictions `tmp/preds_nm9_phys.npz`, weights
   `tmp/weights_nm9_phys_seed{0,1,2}.pt`;
4. evaluation and analysis scripts (`nm_perstation_full.py`, `loso_nm_phys.py`, `transfer_infer_yr.py`).

Determinism: fixed seeds (0, 1, 2) and `cudnn.deterministic = True`.

## Citation

If you use this code, please cite the accompanying paper (in submission).
