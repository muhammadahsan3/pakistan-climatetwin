# Submission checklist: Pakistan ClimateTwin

Deadline: **30 November 2026**

## Deliverables

| # | Deliverable (course brief) | Where | Status |
|---|---|---|---|
| 1 | **Code** (notebooks + Python files) | `notebooks/02`–`07`, `src/`, `gee/`, `dashboard/` | ✅ Done |
| 2 | **Dataset (or link)** | `release/ClimateTwin_dataset.zip` (17 MB) + `data/DATA_README.md` | ⚠️ Upload the zip to Google Drive, set “Anyone with the link”, paste the link below |
| 3 | **Research paper (PDF, IEEE)** | `research/paper/main.tex` → `release/paper_overleaf.zip` | ⚠️ Compile on Overleaf (see `research/paper/README.md`), then finish your team's part |
| 4 | **Presentation (slides)** | Online slide deck (Slides link), with speaker notes on every slide | ⚠️ Replace `[Member]` names and links; download as PowerPoint or PDF from the deck's menu |
| + | Working prototype (digital twin) | `streamlit run dashboard/app.py` | ✅ Done (demo it live, or use screenshots as a backup) |

Links to fill in, then use them in the paper ("Code and Data Availability") and on the last slide:
- GitHub repository: `[ ]`
- Dataset (Google Drive): `[ ]`

## Your team's part (it can't be delegated)

- [ ] **Names** on the paper and the slides.
- [ ] **Read the papers you cite**, at least del Río 2013, Iqbal 2016, Hussain 2024, Mishra 2020, Pepin 2015. Be ready to say why each is cited.
- [ ] **Add 3–5 papers you found yourselves** to Related Work.
- [ ] **Rewrite in your own words** any sentence you couldn't defend in the viva. The plagiarism check looks at text.
- [ ] **Each member rehearses** the whole story: problem → data → anomalies and leakage → findings F1–F3 → models and baselines → why simple wins → ocean drivers → dashboard.
- [ ] Run the dashboard once on the presenting laptop (`python dashboard/prepare_data.py`, then `streamlit run dashboard/app.py`).

## Viva questions to prepare (with short answers)

1. **Why predict anomalies, not temperature?** Raw temperature is mostly the seasonal cycle; even "predict normal" gets R² ≈ 0.98.
2. **How did you avoid data leakage?** Time-ordered splits, climatology from training years only (recomputed per CV fold), scaling inside pipelines, test set used once.
3. **Why does Ridge beat Random Forest/XGBoost?** Weak signal: regularisation matters more than flexibility, and trees can't extrapolate a warming trend.
4. **Are the differences significant?** No. The month-bootstrap 95% CIs include zero (36 test months).
5. **Why is Tmax not warming while Tmin is?** Consistent with global DTR decline. In the plains, irrigation (cooler days, humid nights) is a plausible but untested cause.
6. **What is "digital twin" here?** A data-driven virtual representation that ingests GEE data, analyses trends, forecasts next month and supports what-if exploration. It is not a physics model.
7. **Why false-discovery correction?** 126 tests at 5% give about 6 false hotspots by chance: 42 → 12.

## Reproduce everything from scratch

```bash
python gee/extract_data.py
python gee/extract_indices.py
jupyter nbconvert --to notebook --execute --inplace notebooks/0*.ipynb
python dashboard/prepare_data.py
```
