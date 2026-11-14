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
- [ ] Feature engineering + baselines
- [ ] Models: Ridge, Random Forest, XGBoost, SVR (tuned)
- [ ] Temporal test + spatial (leave-one-province-out) evaluation
- [ ] Dashboard
- [ ] Paper
- [ ] Slides

## 4. Key EDA findings (full text: end of `notebooks/03_eda.ipynb`)

- **Nights warm, days don't:** Tmin +0.19 °C/decade nationally; Tmax ≈ 0 (p = 0.93). DTR shrank in 125/126 districts (44 significant after FDR).
- **Hotspots:** 12 significant Tmin-warming districts (Sindh 6, Balochistan 4, KP 2); 0 for Tmax after FDR.
- **Monsoon-season Tmin** is the strongest signal: +0.34 °C/decade, 85 districts significant.
- **Elevation-dependent warming:** Tmax trend vs elevation ρ = 0.69. Mountains warm, while the irrigated plains show daytime cooling (irrigation is a hypothesis, to support with literature).
- **ML implication:** lag-1 autocorrelation of the Tmax anomaly is only 0.28, so expect modest skill and judge models against the baselines. The test period has 17.8% extreme-heat months vs 10% in training (distribution shift).

## 5. Paper narrative (emerging)

"Pakistan's warming between 2000 and 2025 is mainly a *night-time and monsoon-season* phenomenon, strongest in the mountains and the south-west coast. The irrigated plains show daytime cooling. ML models predict next-month Tmax anomalies with [result] skill over persistence/climatology, and the strongest predictors are antecedent soil moisture and [result]."
Fill in the [result] parts only from experiments.
