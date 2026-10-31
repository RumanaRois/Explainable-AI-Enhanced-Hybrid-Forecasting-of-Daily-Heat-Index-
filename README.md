# Explainable AI-Enhanced Hybrid Forecasting of Daily Heat Index for Heat Early Warning in Southwestern Bangladesh

Code, result tables and figures for the manuscript:

> Atoshi, G.A., Khan, M.M.H., Rois, R. *Explainable AI-Enhanced Hybrid Forecasting of Daily Heat Index for Heat Early Warning in Southwestern Bangladesh.* Submitted to *Advances in Climate Change Research*.

This repository forecasts the daily **Heat Index (HI)** at the Jessore (Jashore) station of the Bangladesh Meteorological Department (BMD) 1, 3 and 7 days ahead. It combines:

- an **STL** seasonal profile,
- a **TBATS-type** trend–Fourier–ARMA model,
- **machine-learning (ML) residual learners**: random forest, SVR, LightGBM, ANN and LSTM.

It then turns the forecasts into **explainable Danger-day warnings** (HI ≥ 41 °C) using conformal prediction intervals and SHAP.

![Framework](figures/fig_framework.png)

---

## Key results (independent test period 2014–2024)

| Lead time | Best model | MAE (°C) | Skill vs persistence | 90% interval coverage |
|---|---|---|---|---|
| 1 day | STL-TBATS-ENS | 1.58 | 7.5% | 89.1% |
| 3 days | STL-TBATS-ENS | 2.18 | 17.6% | 89.2% |
| 7 days | STL-TBATS-ENS | 2.29 | 26.9% | 89.7% |

- **ML adds a small gain.** ML residual correction is significant at 1 day (Diebold–Mariano p = 0.001) but not at 3–7 days. Most of the skill comes from the explicit trend and seasonal structure.
- **Danger-day warnings work best as probabilities.** At 1 day they reach a Heidke skill score of 0.56 (AUC 0.96), and they beat persistence at 3 and 7 days.
- **SHAP shows what drives the forecasts.** Recent forecast errors and same-day temperature and humidity anomalies drive the ML correction.
- **Heat stress is rising.** Danger days at Jessore increased by **3.1 days per decade** over 1981–2024 (Mann–Kendall p < 0.001).

---

## Repository structure

```
heat-index-xai-hybrid/
├── data/                  # input data go here (not distributed; see data/README.md)
├── src/
│   ├── stage1.py          # data, training-only STL profile, ARIMA / ETS / TBATS-type models, rolling h-step forecasts
│   ├── stage2.py          # ML learners (RF, SVR, LightGBM, ANN, LSTM): single, STL-adjusted and residual hybrids
│   ├── stage3.py          # metrics, ensemble, Diebold–Mariano tests, conformal intervals, warning skill, trends, outlook
│   ├── stage4.py          # explainable warning classifier, TreeSHAP, manuscript figures
│   └── fig_framework.py   # framework schematic (Figure 2)
├── results/               # result tables (CSV/JSON) written by the scripts
├── figures/               # figures written by the scripts
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Data

The daily station data (1981–2024) were provided by the Climate Division of the Bangladesh Meteorological Department. They are **not redistributed** here; see [`data/README.md`](data/README.md) for how to request them and for the expected file format.

Place the file at `data/Heat_Index_Data_For_Jessore.xlsx`, or point to it with an environment variable:

```bash
export HI_DATA=/path/to/Heat_Index_Data_For_Jessore.xlsx
```

---

## Installation

You need Python ≥ 3.11. The analysis was run with Python 3.13 on CPU.

```bash
git clone https://github.com/<your-username>/heat-index-xai-hybrid.git
cd heat-index-xai-hybrid
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

A GPU is not needed; the LSTM is small and trains on CPU.

---

## Usage

Run the stages in order from the repository root:

```bash
python src/stage1.py         # ~3 min  statistical models and seasonal profile
python src/stage2.py         # ~10 min ML learners for leads 1, 3, 7 days (2-core CPU)
python src/stage3.py         # ~1 min  evaluation, tests, intervals, warnings, trends, outlook
python src/stage4.py         # ~4 min  warning classifier, SHAP, figures
python src/fig_framework.py  # framework schematic
```

Intermediate files (`*.pkl`) are written to `results/`. They are not tracked by git.

### Outputs and the manuscript

| Output file | Manuscript item |
|---|---|
| `results/trends.csv`, `figures/fig_trends.png` | Table 1, Figure 10 – long-term trends |
| `results/metrics_all.csv`, `figures/fig_lead_skill.png` | Table 2, Figure 11 – accuracy of all 44 configurations |
| `results/dm_tests.csv`, `results/conformal.csv` | Table 3 and Section 3.3 – significance tests and intervals |
| `results/warning_classifier.csv`, `figures/fig_case_2024.png` | Table 4, Figure 12 – Danger-day warning skill, 2024 heatwave |
| `figures/fig_decomposition.png` | Figure 13 – additive explanation of forecasts |
| `results/shap_groups_reg.csv`, `figures/fig_shap_regression.png` | Figure 14 – SHAP for ML residual learner |
| `results/shap_groups_clf.csv`, `figures/fig_shap_classifier.png` | Figure 15 – SHAP for warning classifier |
| `results/outlook.csv`, `figures/fig_outlook.png` | Table 5, Figure 16 – trend-adjusted outlook 2025–2026 |
| `figures/fig_framework.png` / `.pdf` | Figure 2 – framework |

---

## Methods in brief

- **Heat Index.** HI is computed with the US NWS algorithm (Steadman / Rothfusz with adjustments) from daily mean T and RH. The single missing day is interpolated, and no outlier smoothing is applied.
- **Leakage-free design.** The data are split chronologically:
  - training: 1981–2010
  - validation: 2011–2013, used for early stopping, warning thresholds and conformal calibration
  - test: 2014–2024, 4,018 days never used in fitting

  STL is fitted on training years only. Forecasts are issued from rolling origins using data available up to that day only.
- **Hybrid chain.** The hybrid ensemble is built as follows:

  ŷ(t+h) = S(d) + L̂(t+h | t) + mean over m of ĝₘ(xₜ)

  - S(d) is the training-only STL day-of-year profile.
  - L̂ is the TBATS-type forecast: trend + Fourier terms + ARMA errors.
  - ĝₘ are five ML learners trained on the stage-1 errors, using residual lags, HI/T/RH anomalies and day of year.
- **Baselines and comparisons.** The study compares 44 configurations, including persistence, climatology, ARIMA, ETS, single ML, STL-adjusted and hybrid models.
- **Uncertainty.** Season-specific split-conformal 90% prediction intervals.
- **Explainability.** An additive component breakdown, plus exact TreeSHAP for the LightGBM residual learner and the warning classifier.
- **Warning verification.** POD, FAR, CSI, HSS, AUC and Brier skill score for HI ≥ 41 °C.
- **Trends.** Mann–Kendall test with the Hamed–Rao correction, and Sen's slope.

---

## Reproducibility notes

- Random seeds are fixed for RF, LightGBM, ANN and LSTM.
- LSTM results can differ slightly (typically < 0.01 °C MAE) across hardware and PyTorch versions.
- The tables in `results/` were produced with the package versions in `requirements.txt`.
- The exploratory figures in Section 3.1 of the manuscript (Figures 1 and 3–9) were produced separately in R 4.5.1.

---

## Citation

If you use this code, please cite the manuscript above. The citation will be updated with volume, pages and DOI after publication.

```bibtex
@article{Atoshi_HI_XAI_Hybrid,
  author  = {Atoshi, Gulshanara Afrose and Khan, Mohammad Mahboob Hussain and Rois, Rumana},
  title   = {Explainable AI-Enhanced Hybrid Forecasting of Daily Heat Index for Heat Early Warning in Southwestern Bangladesh},
  journal = {Advances in Climate Change Research},
  year    = {2026},
  note    = {Submitted}
}
```

---

## Authors and contact

- **Gulshanara Afrose Atoshi** – Department of Statistics and Data Science, Jahangirnagar University, Dhaka-1342, Bangladesh (gulshanara.stu2018@juniv.edu)
- **Mohammad Mahboob Hussain Khan** – Bangladesh Meteorological Department (BMD), Dhaka-1207, Bangladesh
- **Rumana Rois** – Department of Statistics and Data Science, Jahangirnagar University, Dhaka-1342, Bangladesh (rois@juniv.edu)

Questions and issues are welcome through the GitHub issue tracker or by email to the corresponding authors.

## License

The code is released under the MIT License (see `LICENSE`). The BMD station data are subject to BMD's data policy and are not covered by this license.
