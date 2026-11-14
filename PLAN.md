# PLAN — Pakistan ClimateTwin

Living plan. The full design is in `PROJECT MASTER DOCUMENT.docx`; this file records
**decisions taken since then** and **progress**. Update it as work happens.

Deadline: **30 November 2026**

---

## 1. Decisions that update the master document

| # | Change | Why |
|---|--------|-----|
| D1 | **Primary target = next-month Tmax *anomaly*** (Tmax minus that district's average for that calendar month, computed from training years 2000–2019 only). Raw Tmax is still reported for interpretation. | Raw Tmax is dominated by the seasonal cycle, so every model scores R² ≈ 0.99 and the comparison says nothing. The anomaly isolates the part that is actually hard to predict — departures from normal. |
| D2 | **Two naive baselines**: *climatology* (predict anomaly = 0) and *persistence* (next month's anomaly = this month's). Report **skill score = 1 − RMSE_model / RMSE_baseline**. | Shows whether ML adds anything beyond trivial rules. Strongest "reasoning, not just accuracy" point. |
| D3 | **Secondary classification task**: "extreme-heat month" = anomaly above the 90th percentile of the training period. Report Precision, Recall, F1. | Ties to heatwave risk; covers the classification metrics named in the course brief. |
| D4 | **Monthly Tmax = mean of daily maxima** from `ECMWF/ERA5_LAND/DAILY_AGGR`, not the `MONTHLY_AGGR` max band. | The monthly product's max band may be the hottest single hour of the month, which is a different quantity. Verified with `gee/extract_data.py --check`. |
| D5 | **Elevation (SRTM) is a core feature**, not optional. | Pakistan spans sea level to 8,611 m; elevation strongly controls temperature. One line in GEE. |
| D6 | **Spatial validation (leave-one-province-out) is a core result**, not "if time permits". | Tests whether models generalise to unseen regions — a genuine research question and a source of novelty. |
| D7 | **Dashboard cut to 3 pages**: (1) Map + warming hotspots, (2) Prediction + model comparison, (3) Scenario explorer. | Fits the deadline. |
| D8 | **MODIS dropped** unless the core pipeline finishes early. | Scope. |
| D10 | **Tuning = 4-fold expanding-window time-series CV** inside 2000–2022 (validate 2011–13, 2014–16, 2017–19, 2020–22), then refit on 2000–2022 and a single evaluation on 2023–2025. Replaces tuning on the single 2020–22 block. | Anomalies are spatially correlated, so one 3-year block ≈ 36 effective samples. Tuning on it was unstable (first experiment: models beat climatology in only 19/36 validation months). |
| D11 | **Each CV fold recomputes the climatology from its own training years** (`models.make_cv_folds`). | Found during development: with the global 2000–2019 climatology, folds 1–3 leaked the validation-period mean into training anomalies. The test set was never affected. |
| D12 | **Third baseline: damped persistence (AR(1))**, plus **"Seasonal persistence" interaction features** (anomaly × sin/cos of the target month). | Damped persistence beats climatology and, on CV, every untuned ML model, so it is the honest bar. The interactions let linear models use persistence that varies by season (EDA F7). |
| D13 | **Search ranges widened** after the first Ridge/RF optima sat on the grid edge; notebook 05 checks that selected values are interior. | A boundary optimum means the true optimum may lie outside the range. |
| D9 | **Boundaries = geoBoundaries v6.0.0 ADM2** (`WM/geoLab/geoBoundaries/600/ADM2`): **126 districts**, labelled with 7 provinces/territories. This is an older delimitation (Pakistan now has ~160 districts after later splits), and Azad Kashmir is a single unit. | It contains real districts, not divisions; one consistent map for all 26 years; already in GEE. State the version and the 126-unit count in the paper's limitations. |

## 2. Schedule

| Days | Dates | Work | Output |
|------|-------|------|--------|
| 1–2 | 12–13 Nov | Project setup, GEE access, boundary check, sample extraction | `data/raw/` sample, verified bands |
| 3–5 | 14–16 Nov | Full extraction, cleaning, EDA, Mann-Kendall/Sen's slope hotspots | `data/processed/`, EDA figures |
| 6–10 | 17–21 Nov | Features, baselines, 4 models, tuning, temporal + spatial evaluation, SHAP | Results tables, figures |
| 11–13 | 22–24 Nov | Streamlit dashboard (3 pages) | `dashboard/app.py` |
| 14–18 | 25–29 Nov | IEEE paper, slides, final review | PDF, slides |
| — | 30 Nov | Submit | |

## 3. Progress

- [x] Project structure, README, PLAN, requirements
- [x] GEE extraction script written (`gee/extract_data.py`)
- [x] GEE account + Cloud project registered (noncommercial, Community tier), authentication done
- [x] `--check` run: all ERA5 bands present, CHIRPS band = `precipitation`, 126 districts
- [x] Sample year extracted and sanity-checked (Jacobabad hottest in June, monsoon peak Jul–Aug)
- [x] Full 2000–2025 extraction: 39,312 rows (126 districts × 312 months), 0 missing, 0 duplicates, no incomplete months
- [x] Preprocessing notebook (`notebooks/02_preprocessing.ipynb`, `src/preprocessing.py`): quality checks all 0, training-only climatology + anomalies → `data/processed/climate_monthly.parquet`
- [x] EDA + hotspot analysis (`notebooks/03_eda.ipynb`, `src/trends.py`): 13 figures in `research/figures/`, district trend table → `data/processed/district_trends.csv`, findings F1–F7 at the end of the notebook
- [x] Feature engineering + baselines (`notebooks/04_feature_engineering.ipynb`, `src/features.py`: 23 features in 7 groups, leakage checks pass)
- Development finding (leak-free CV, before final tuning): damped persistence CV RMSE 1.520 vs climatology 1.631. No feature set or model variant tried beat damped persistence on the 4-fold average; the best was Ridge + seasonal-persistence interactions (1.521, a tie).
- [x] Models: Ridge, Random Forest, XGBoost, SVR tuned with leak-free 4-fold CV (`notebooks/05_model_training.ipynb`, `src/models.py`). No selected hyperparameter sits on a grid edge. Best params are in `models/best_params.json`.
- [x] Temporal test + spatial (leave-one-province-out) evaluation, extreme-heat F1, permutation importance, SHAP (`notebooks/06_model_evaluation.ipynb`). Tables are in `reports/`, figures `ml_*.png` in `research/figures/`. Findings M1–M8 are at the end of notebook 06.
- [x] Dashboard (`dashboard/`): Streamlit, 4 pages (overview, map & hotspots, forecast & models, scenario explorer). Data packed by `dashboard/prepare_data.py`, including a real Jan 2026 forecast from Ridge + ocean refitted on 2000–2025. All pages pass a headless smoke test.
- [ ] Paper
- [ ] Slides

## 4. Key EDA findings (full text: end of `notebooks/03_eda.ipynb`)

- **Nights warm, days don't:** Tmin +0.19 °C/decade nationally; Tmax ≈ 0 (p = 0.93). DTR shrank in 125/126 districts (44 significant after FDR).
- **Hotspots:** 12 significant Tmin-warming districts (Sindh 6, Balochistan 4, KP 2); 0 for Tmax after FDR.
- **Monsoon-season Tmin** is the strongest signal: +0.34 °C/decade, 85 districts significant.
- **Elevation-dependent warming:** Tmax trend vs elevation ρ = 0.69. Mountains warm, while the irrigated plains show daytime cooling (irrigation is a hypothesis, to support with literature).
- **ML implication:** lag-1 autocorrelation of the Tmax anomaly is only 0.28, so expect modest skill and judge models against the baselines. The test period has 17.8% extreme-heat months vs 10% in training (distribution shift).

## 5. Key modelling results (test 2023–2025; full text: end of `notebooks/06_model_evaluation.ipynb`)

| | RMSE (°C) | Skill vs climatology | Skill vs damped persistence | Extreme-heat F1 |
|---|---|---|---|---|
| Climatology | 1.756 | 0 | −8.0% | – |
| Damped persistence | 1.689 | +7.4% | 0 | 0.31 |
| **Ridge** | **1.673** | **+9.1%** | **+1.8%** | **0.37** |
| XGBoost | 1.695 | +6.8% | −0.7% | 0.35 |
| SVR | 1.706 | +5.6% | −2.0% | 0.37 |
| Random Forest | 1.711 | +5.0% | −2.6% | 0.35 |

- None of the ML-vs-baseline differences is significant (95% month-bootstrap CIs include 0; only 36 test months).
- Strongest predictor: solar-radiation anomaly, then temperature memory. Geography adds nothing.
- All models under-predict the warm test period by 0.26–0.33 °C (bias from a warming climate).
- Leave-one-province-out skill drop ≤ 0.015: the learned relations transfer across regions.

### Extension: large-scale ocean drivers (`notebooks/07_ocean_drivers.ipynb`, `gee/extract_indices.py`)

Niño-3.4 (ENSO), DMI (IOD) and tropical-mean SST from NOAA OISST v2.1 in GEE, as anomalies vs a fixed 1982–2010 base (leakage-free in every fold). Each model was re-tuned with these features.

| Test 2023–2025 | RMSE local | RMSE + ocean | Skill vs clim (+ocean) | Bias local → +ocean |
|---|---|---|---|---|
| **Ridge** | 1.673 | **1.650** | **+11.7%** | −0.29 → −0.23 |
| Random Forest | 1.711 | 1.679 | +8.5% | −0.27 → −0.20 |
| XGBoost | 1.695 | 1.693 | +7.0% | −0.26 → −0.13 |
| SVR | 1.706 | 1.700 | +6.2% | −0.33 → −0.16 |

- Ridge + ocean beats local Ridge significantly (+2.8% skill, 95% CI [+0.04%, +5.8%]). It beats damped persistence by +4.6%, which is not significant.
- On CV (2011–2022) there is no robust gain; for RF, XGBoost and SVR it is slightly worse. The gain appears in the warm recent years, mainly through bias reduction.
- **Best configuration: Ridge + ocean drivers.** Use it in the dashboard, labelled with its uncertainty.

## 6. Paper narrative

"Pakistan's warming between 2000 and 2025 is mainly a *night-time and monsoon-season* phenomenon, strongest in the mountains and the south-west coast; the irrigated plains show daytime cooling. One month ahead, district Tmax anomalies are only weakly predictable from local reanalysis predictors. A regularised linear model is best (+9% skill over climatology), while more complex models (RF, XGBoost, SVR) do not beat a one-parameter damped-persistence baseline. Solar-radiation and temperature anomalies carry the predictable signal, and the relationships transfer across provinces. ML adds most value for flagging extreme-heat months. Adding ENSO, IOD and tropical-SST indices from GEE gives the best configuration (Ridge, +11.7% skill over climatology on 2023–2025), mainly by reducing a warming-induced bias. This gain is not robust in cross-validation, which marks the information limit of one-month-ahead district forecasts."
