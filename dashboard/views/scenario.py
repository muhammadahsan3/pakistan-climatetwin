import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import common as cm

meta = cm.meta()
d = cm.districts()
m = cm.monthly()
clim = cm.clim_monthly()
THR = meta["extreme_threshold_anom"]
BASE = (2016, 2025)

st.title("Scenario explorer")
st.warning("**Hypothetical what-if experiments, not climate projections.** The recent decade's observed months are shifted by "
           "the amounts you choose and the heat-risk indicators are recounted. The temperature and rainfall changes are applied "
           "independently; no causal link between them is modelled.", icon=":material/science:")

INDICATORS = {
    "Dangerous-heat months (Tmax ≥ 40 °C)":
        "Months per year whose **mean daily maximum** reaches 40 °C or more.",
    "Extreme-heat months (vs local normal)":
        f"Months per year whose Tmax anomaly exceeds **{THR:.2f} °C**, the 90th percentile of 2000–2019 anomalies. "
        "Local by design: it also flags unusual heat in cool mountain districts.",
    "Hot-and-dry months":
        "Months per year with Tmax ≥ 40 °C **and** rainfall below that district's 20th percentile for the calendar month (2000–2019). "
        "Compound heat-and-drought stress.",
}
c1, c2, c3 = st.columns([2, 2, 3])
dT = c1.select_slider("Temperature change (°C)", options=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0], value=1.5)
dR = c2.select_slider("Rainfall change (%)", options=[-30, -20, -10, 0, 10, 20], value=-10)
ind = c3.selectbox("Heat-risk indicator", list(INDICATORS))
st.caption(INDICATORS[ind] + f" Baseline: observed months {BASE[0]}–{BASE[1]}.")


@st.cache_data
def baseline_frame() -> pd.DataFrame:
    b = m[m["year"].between(*BASE)].merge(clim[["district_id", "month", "rain_p20"]], on=["district_id", "month"])
    return b


def count(b: pd.DataFrame, dT: float, dR: float, indicator: str) -> pd.Series:
    tmax = b["tmax_c"] + dT
    if indicator.startswith("Dangerous"):
        flag = tmax >= 40
    elif indicator.startswith("Extreme"):
        flag = (b["tmax_c_anom"] + dT) > THR
    else:
        flag = (tmax >= 40) & (b["rain_mm"] * (1 + dR / 100) < b["rain_p20"])
    n_years = BASE[1] - BASE[0] + 1
    return flag.groupby(b["district_id"]).sum() / n_years


b = baseline_frame()
base = count(b, 0.0, 0, ind).rename("baseline")
scen = count(b, dT, dR, ind).rename("scenario")
res = d[["district_id", "district", "province", "elevation_m"]].merge(pd.concat([base, scen], axis=1).reset_index(), on="district_id")
res["change"] = res["scenario"] - res["baseline"]

k = st.columns(4)
k[0].metric("National mean, baseline", f"{res['baseline'].mean():.2f} months / yr")
k[1].metric("National mean, scenario", f"{res['scenario'].mean():.2f} months / yr", f"{res['change'].mean():+.2f}", delta_color="inverse")
k[2].metric("Districts affected", f"{int((res['change'] > 0).sum())} / {len(res)}", help="Districts where the indicator increases")
k[3].metric("Largest increase", f"{res['change'].max():+.2f} months / yr", res.loc[res["change"].idxmax(), "district"], delta_color="off")

left, right = st.columns([3, 2], gap="large")
with left:
    view = st.radio("Map shows", ["Change vs baseline", "Scenario value"], horizontal=True)
    col = "change" if view.startswith("Change") else "scenario"
    show = res.assign(**{"Baseline": res["baseline"].round(2), "Scenario": res["scenario"].round(2), "value": res[col].round(2)})
    fig = cm.choropleth(show, "value", cm.SEQ_ORANGE, "months / yr", hover=["province", "Baseline", "Scenario"], height=520)
    st.plotly_chart(fig, width="stretch")
with right:
    prov = res.groupby("province")[["baseline", "scenario"]].mean().reindex([p for p in cm.PROVINCE_ORDER if p in res["province"].unique()])
    prov.index = prov.index.str.replace("Islamabad Capital Territory", "ICT").str.replace("Khyber Pakhtunkhwa", "KP")
    fig = go.Figure()
    fig.add_bar(y=prov.index, x=prov["baseline"], name="Baseline", orientation="h", marker_color="#c3c2b7",
                hovertemplate="%{y}: %{x:.2f}")
    fig.add_bar(y=prov.index, x=prov["scenario"], name="Scenario", orientation="h", marker_color=cm.ORANGE,
                hovertemplate="%{y}: %{x:.2f}")
    fig.update_layout(barmode="group", yaxis=dict(autorange="reversed"))
    st.plotly_chart(cm.layout(fig, 420, title="Province average (months / year)", hovermode="closest"), width="stretch")

# Sensitivity curve: how the national indicator grows with warming (at the chosen rainfall change)
curve = pd.DataFrame({"dT": np.arange(0, 3.01, 0.25)})
curve["value"] = [count(b, t, dR, ind).mean() for t in curve["dT"]]
fig = go.Figure(go.Scatter(x=curve["dT"], y=curve["value"], mode="lines+markers", line=dict(color=cm.ORANGE, width=2),
                           hovertemplate="+%{x:.2f} °C: %{y:.2f} months/yr<extra></extra>"))
fig.add_vline(x=dT, line_color=cm.INK_2, line_dash="dot")
st.plotly_chart(cm.layout(fig, 300, title=f"National mean '{ind}' vs temperature change (rainfall {dR:+d}%)",
                          xaxis_title="Temperature change (°C)", yaxis_title="months / year", hovermode="closest"),
                width="stretch")

st.subheader("Most affected districts")
top = res.sort_values("change", ascending=False).head(15)[["district", "province", "elevation_m", "baseline", "scenario", "change"]]
top.columns = ["District", "Province", "Elevation (m)", "Baseline (months/yr)", "Scenario (months/yr)", "Change"]
st.dataframe(top.round(2), hide_index=True, width="stretch")
