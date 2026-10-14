# Groundwater forecast difficulty across contrasting models — code (v1, partial release)

MATLAB code accompanying the manuscript *Groundwater forecast difficulty
across contrasting models: shared patterns and diagnosis from antecedent
hydrological information*.

**This is version 1 (v1) of the code release.** It covers the two
computational cores of the study:

- **Forecasting experiments** — nine 12-month forecasting configurations
  of monthly groundwater-storage change (dGWS) on a global 1° grid;
- **Forecastability analysis** — pixel × forecast-year performance
  database, meta-features, three random-forest (RF) meta-models
  (M1 / M2 / M3) that diagnose forecast error from pre-forecast
  information, exact TreeSHAP attribution, and robustness checks.

> **Intentionally withheld in v1:** the data-preparation scripts
> (raw GLDAS download → 1° monthly target/predictor series) and the
> figure-generation scripts. These will be released together with the
> raw data after the manuscript is accepted for publication.
> No raw or derived data are included in this repository.

---

## 1. Scope of this release

```
GitHub_v1/
├── 02_forecasting_models/        # nine forecasting configurations
│   ├── linear_models/            #   AR, ARIMA, SARIMA
│   ├── nonlinear_models/         #   SVM, RF, LSTM
│   └── nonlinear_x_models/       #   SVMX, RFX, LSTMX (antecedent hydroclimate)
└── 03_forecastability_analysis/  # master database, features, meta-models, SHAP, robustness
```

Folder numbering follows the numbering of the complete pipeline
(`01_data_preparation` → `04_figures`); only stages **02** and **03** are
released in v1. The withheld stages will be added in a later release.

### What is included (`03_forecastability_analysis/`)

| Script | Purpose |
|---|---|
| `s1_master_database.m` | [legacy] lead-level master database |
| `s1_master_database_pixel_year.m` | pixel × forecast-year master database (main input of the analysis) |
| `s2_dgws_features.m` | 13 antecedent dGWS temporal features |
| `s2_dgws_features_hydro.m` | 24 antecedent hydroclimatic features |
| `s3_feature_diagnostics.m` | [QA] feature diagnostics |
| `s4_target_diagnostics.m` | [QA] target diagnostics |
| `s5_RF_meta_model.m` | **M1**: dGWS features → 27 RF meta-models |
| `s5_RF_meta_model_M2_hydro.m` | **M2**: hydroclimatic features |
| `s5_RF_meta_model_M3_dgws_hydro.m` | **M3**: 37 combined features |
| `s6_forecastability_target_correlation.m` | cross-model commonality (Spearman) + amplitude checks |
| `s6_M1_M2_M3_comparison.m` | paired M3 vs M1/M2 + pixel-cluster bootstrap |
| `s7_M1_TreeSHAP_compute.m` | exact TreeSHAP, all 27 M1 models |
| `s7_M2_TreeSHAP_compute.m` | exact TreeSHAP, all 27 M2 models |
| `s7_M3_TreeSHAP_compute.m` | exact TreeSHAP, all 27 M3 models |
| `s7a_M1_SHAP_TreeSHAP_validation.m` | [validation] tree-wise SHAP additivity (App. S1.5) |
| `s7b_M1_TreeSHAP_optimization.m` | [sensitivity] background/settings (App. S1.5) |
| `s8_M1_M2_M3_SHAP_analysis.m` | integrated SHAP analysis (fig.-ready outputs) |
| `s10_forecastability_robustness_checks.m` | amplitude control, low-amplitude, spatial-block bootstrap (App. S2.5–S2.6) |

---

## 2. Requirements

- **Developed and tested in MATLAB R2024a.** Compatibility with other
  versions has not been systematically tested.
- Toolboxes:
  - Statistics and Machine Learning Toolbox (`TreeBagger`,
    `fitrensemble`, `fitrsvm`, `shapley`)
  - Econometrics Toolbox (`arima`, `estimate`, `forecast`)
  - System Identification Toolbox (`ar`, `forecast` for AR models)
  - Deep Learning Toolbox (`sequenceInputLayer`, `lstmLayer`, ...)
  - Parallel Computing Toolbox (`parpool`, `parfor`,
    `parallel.pool.Constant`) — scripts fall back to serial loops if the
    pool is unavailable.

---

## 3. Data note

**No data files are shipped with this release.** The released scripts
consume the prepared 1° GLDAS series for the study period of the
manuscript — December 2003 to December 2024 (monthly dGWS from January
2004) — covering precipitation, evapotranspiration, runoff, surface
temperature, soil-moisture change, and groundwater storage. All analyses
stop at December 2024; no 2025 data are used.

The data-preparation scripts that generate these inputs, and the raw
NASA GLDAS products they read, are intentionally not part of v1; they
will be provided in the full release after publication.

---

## 4. Run order

The pipeline is sequential — every script consumes files produced by the
previous steps.

### Stage 1 — Forecasting experiments (`02_forecasting_models/`)

Each script trains/validates/forecasts per grid cell in parallel blocks
with checkpoints, and saves `<MODEL>_univariate_global_2019_2021.mat`
(or `_multivariate_` for the X models) into a `s1_data_results` folder.

| # | Script | Configuration (see manuscript App. S1.1) |
|---|---|---|
| 1 | `linear_models/AR_global_forecast_2019_2021.m` | AR(p), p ∈ {1,6,12,24,36,48,60}, selected by validation-year NRMSE |
| 2 | `linear_models/ARIMA_global_forecast_2019_2021.m` | ARIMA(p,0,1) |
| 3 | `linear_models/SARIMA_global_forecast_2019_2021.m` | SARIMA(p,0,1)×(1,1,1)₁₂ |
| 4 | `nonlinear_models/SVM_global_forecast_2019_2021.m` | RBF SVM, BoxConstraint ∈ {1,…,100}, horizon-specific direct |
| 5 | `nonlinear_models/RF_global_forecast_2019_2021.m` | bagged trees, 50 trees, MinLeafSize 5, horizon-specific direct |
| 6 | `nonlinear_models/LSTM_global_forecast_2019_2021.m` | LSTM (32 units) seq-to-vector, 12-month output |
| 7 | `nonlinear_x_models/SVMX_global_forecast_2019_2021.m` | SVM + antecedent P/ET/R/T/dSM (x ∈ {1,3,6} months) |
| 8 | `nonlinear_x_models/RFX_global_forecast_2019_2021.m` | RF + antecedent hydroclimate |
| 9 | `nonlinear_x_models/LSTMX_global_forecast_2019_2021.m` | LSTM with multivariate input sequences |

**2019–2024 coverage.** The released scripts run the 2019–2021 test years
(rolling rounds: train 2004–2017/2018/2019, validate the following year,
test 2019/2020/2021). The 2022–2024 forecasts used in the manuscript were
produced with the same scripts by extending `testYears` to `[2022 2023
2024]` and renaming the output/checkpoint files accordingly (e.g.
`AR_univariate_global_200401_202412.mat`), exactly as consumed by
`s1_master_database_pixel_year.m`.

### Stage 2 — Forecastability analysis (`03_forecastability_analysis/`)

| # | Script | Purpose | Key outputs |
|---|---|---|---|
| 10 | `s1_master_database.m` | [legacy] lead-level master database | `s1_master_database.mat` |
| 11 | `s1_master_database_pixel_year.m` | **main input:** pixel × forecast-year database with NRMSE_Q10Q90 targets | `s1_master_database_pixel_year.mat` |
| 12 | `s2_dgws_features.m` | 13 antecedent dGWS temporal features (4 information groups) | `s2_dgws_features.mat` |
| 13 | `s2_dgws_features_hydro.m` | 24 antecedent hydroclimatic features | `s2_dgws_features_hydro.mat` |
| 14 | `s3_feature_diagnostics.m` | [QA] feature distributions, correlations, redundancy | `s3_feature_diagnostics.mat` |
| 15 | `s4_target_diagnostics.m` | [QA] raw vs log1p target distributions | `s4_target_diagnostics.mat` |
| 16 | `s5_RF_meta_model.m` | **M1**: 27 RF meta-models (9 targets × 3 holdouts), 13 dGWS features | `s5_RF_meta_model.mat` |
| 17 | `s5_RF_meta_model_M2_hydro.m` | **M2**: 27 RF meta-models, 24 hydro features | `s5_RF_meta_model_M2_hydro.mat` |
| 18 | `s5_RF_meta_model_M3_dgws_hydro.m` | **M3**: 27 RF meta-models, 37 combined features | `s5_RF_meta_model_M3_dgws_hydro.mat` |
| 19 | `s6_forecastability_target_correlation.m` | cross-model commonality: Spearman NRMSE/RMSE, amplitude adjustment | `s6_forecastability_target_correlation.mat` |
| 20 | `s6_M1_M2_M3_comparison.m` | paired ΔRMSE/improvement of M3 vs M1 & M2; 2,000-pixel-cluster bootstrap | `s6_M1_M2_M3_comparison.mat` |
| 21 | `s7a_M1_SHAP_TreeSHAP_validation.m` | [validation] tree-wise SHAP additivity & Kernel-SHAP benchmark (App. S1.5) | `M1_AR2022_TreeSHAP_validation.mat` |
| 22 | `s7b_M1_TreeSHAP_optimization.m` | [sensitivity] worker/query/BG-size & sampling robustness (App. S1.5) | `s7b_M1_TreeSHAP_optimization.mat` |
| 23 | `s7_M1_TreeSHAP_compute.m` | exact interventional TreeSHAP, all 27 M1 models, BG300 | `M1_TreeSHAP_compute_summary.mat` |
| 24 | `s7_M2_TreeSHAP_compute.m` | exact TreeSHAP, all 27 M2 models | `M2_TreeSHAP_compute_summary.mat` |
| 25 | `s7_M3_TreeSHAP_compute.m` | exact TreeSHAP, all 27 M3 models | `M3_TreeSHAP_compute_summary.mat` |
| 26 | `s8_M1_M2_M3_SHAP_analysis.m` | integrated attribution: profiles, restructuring, stability, directional, spatial | `s8_M1_M2_M3_SHAP_analysis.mat` |
| 27 | `s10_forecastability_robustness_checks.m` | amplitude control, low-amplitude exclusion, 2°/5°/10° spatial-block bootstrap (App. S2.5–S2.6) | `s10_forecastability_robustness.mat` |

Steps 14–15 and 21–22 are quality-control / validation scripts; they are
not required to reproduce the headline results but support the
diagnostics reported in the appendices.

---

## 5. Notes

- **Hard-coded paths.** All scripts contain absolute Windows paths
  (e.g. `E:\P1_SKLGB\...`) in their "User settings" blocks at the top.
  Update these paths to your local directory layout before running. No
  script was modified for this release.
- **Parallel settings.** `numWorkers` and block sizes are set at the top
  of each script and can be adjusted to your machine.
- **Generated outputs are not committed.** Every `.mat` / `.csv` /
  `.png` / `.fig` / log file is produced by running the scripts (see
  `.gitignore`).
- **Internal header names.** Some file headers still carry older names
  (e.g. the header of `s5_RF_meta_model.m` references development
  versions). The files themselves are the final versions used for the
  manuscript.

---

## 6. Citation

If you use this code, please cite:

> Hu, K., Zhang, Q., Xiao, X. Groundwater forecast difficulty across
> contrasting models: shared patterns and diagnosis from antecedent
> hydrological information. Manuscript under review; full citation to be
> added upon publication.

---

## 7. License & contact

This code is provided for research and reproducibility purposes. The
data-preparation and figure-generation scripts, together with the raw
data, will be added to this repository after the manuscript is accepted
for publication. For questions, contact the corresponding author:
Kexiang Hu (hukx@cdut.edu.cn).
