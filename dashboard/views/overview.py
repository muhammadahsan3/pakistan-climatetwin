import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import common as cm
from trend_utils import sen_slope_per_decade

meta = cm.meta()
d = cm.districts()
m = cm.monthly()
fc = cm.forecast()

st.title("Pakistan ClimateTwin")
st.markdown(
    "A research-scale **climate digital twin** of Pakistan's 126 districts. It combines satellite and "
    "reanalysis data from Google Earth Engine with machine learning to show **how the climate has changed "
    "(2000–2025)**, **where warming is concentrated**, **what next month may bring**, and **how heat risk "
    "responds to hypothetical warming**.")

nat = m.groupby("year")[["tmax_c_anom", "tmin_c_anom"]].mean()
tmin_trend = sen_slope_per_decade(nat["tmin_c_anom"].to_numpy())
tmax_trend = sen_slope_per_decade(nat["tmax_c_anom"].to_numpy())
n_hot = int((d["Tmin_trend_class"] == "Significant warming").sum())

c = st.columns(4)
c[0].metric("Night-time warming (Tmin)", f"{tmin_trend:+.2f} °C/dec",
            help="Sen's slope of the national annual mean Tmin anomaly, 2000–2025")
c[1].metric("Daytime warming (Tmax)", f"{tmax_trend:+.2f} °C/dec",
            help="Sen's slope of the national annual mean Tmax anomaly, 2000–2025 (not significant)")
c[2].metric("Warming hotspot districts", f"{n_hot} / {meta['n_districts']}",
            help="Significant Tmin warming after Mann-Kendall test with Benjamini–Hochberg FDR correction")
c[3].metric(f"Forecast, {pd.Timestamp(meta['forecast_month']):%b %Y}",
            f"{fc['pred_anom'].mean():+.2f} °C", help="National mean predicted Tmax anomaly vs 2000–2019 (best model)")

left, right = st.columns([3, 2], gap="large")
with left:
    fig = go.Figure()
    for col, name, color in [("tmin_c_anom", "Tmin (night)", cm.BLUE), ("tmax_c_anom", "Tmax (day)", cm.ORANGE)]:
        fig.add_scatter(x=nat.index, y=nat[col], name=name, mode="lines+markers", line=dict(color=color, width=2),
                        marker=dict(size=6), hovertemplate="%{x}: %{y:+.2f} °C")
    fig.add_hline(y=0, line_color=cm.MUTED, line_width=1)
    cm.layout(fig, 330, title="National annual temperature anomaly (vs 2000–2019)", yaxis_title="°C")
    st.plotly_chart(fig, width="stretch")
with right:
    st.subheader("Key findings")
    st.markdown(f"""
- **Nights warm, days don't.** Tmin rises {tmin_trend:+.2f} °C/decade, while Tmax shows no trend. The day–night range shrank in 125 of 126 districts.
- **Hotspots** are concentrated in Sindh and coastal Balochistan, and warming is strongest in the **monsoon season**.
- **Mountains warm by day, the irrigated plains cool by day** (elevation-dependent warming).
- **One month ahead, temperature is only weakly predictable.** The best model ({meta['best_model']}) improves on climatology by {meta['best_test_skill_vs_clim']:.0%} on 2023–2025.
""")

st.divider()
st.subheader("How the twin works")
steps = st.columns(4)
for col, (title, body) in zip(steps, [
        ("1 · Sense", "ERA5-Land temperature, soil moisture and radiation, CHIRPS rainfall and NOAA ocean temperatures, pulled from **Google Earth Engine** and averaged per district and month."),
        ("2 · Analyse", "Anomalies against a 2000–2019 baseline; **Mann-Kendall + Sen's slope** trend tests with false-discovery control identify hotspots."),
        ("3 · Predict", "Ridge, Random Forest, XGBoost and SVR, tuned with time-series cross-validation, forecast next month's Tmax anomaly and are compared against statistical baselines."),
        ("4 · Explore", "Interactive maps, district histories, forecasts with error bands, and **what-if** heat-risk scenarios.")]):
    col.markdown(f"**{title}**\n\n{body}")

cm.caveat(f"Research prototype. ERA5-Land is a reanalysis (model + observations), not station data. Trends use {meta['data_start'][:4]}–"
          f"{meta['data_end'][:4]}; live data through {cm.latest_data_month():%B %Y}"
          + (f" (last update {meta['updated_at']})" if "updated_at" in meta else "")
          + ". Forecasts are one-month-ahead statistical predictions, not climate projections.")
