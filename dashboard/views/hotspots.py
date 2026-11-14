import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import common as cm

d = cm.districts()
m = cm.monthly()
clim = cm.clim_monthly()

st.title("Climate map & warming hotspots")
st.markdown("Explore 26 years of district climate. Trend layers use the **Mann-Kendall test** and **Sen's slope** on "
            "annual values. A district counts as significant only after Benjamini–Hochberg false-discovery correction (q < 0.05). "
            "**Choose a district** in the list on the right (or hover over the map) to see its history.")

annual = m.groupby(["district_id", "year"]).agg(tmax=("tmax_c", "mean"), tmin=("tmin_c", "mean"), rain=("rain_mm", "sum")).reset_index()
means = annual.groupby("district_id")[["tmax", "tmin", "rain"]].mean().reset_index()
d = d.merge(means, on="district_id")

LAYERS = {
    "Tmin trend (night-time warming)": ("Tmin_sen_slope_per_decade", "Tmin_q_value", cm.DIVERGING, 0.0, "°C / decade"),
    "Tmax trend (daytime warming)": ("Tmax_sen_slope_per_decade", "Tmax_q_value", cm.DIVERGING, 0.0, "°C / decade"),
    "Day–night range trend (DTR)": ("DTR_sen_slope_per_decade", "DTR_q_value", cm.DIVERGING, 0.0, "°C / decade"),
    "Rainfall trend": ("Rain_sen_slope_per_decade", "Rain_q_value", cm.DIVERGING_R, 0.0, "mm/yr per decade"),
    "Mean Tmax 2000–2025": ("tmax", None, cm.SEQ_ORANGE, None, "°C"),
    "Mean Tmin 2000–2025": ("tmin", None, cm.SEQ_ORANGE, None, "°C"),
    "Mean annual rainfall": ("rain", None, cm.SEQ_BLUE, None, "mm / year"),
}
c1, c2 = st.columns([3, 2])
layer = c1.selectbox("Map layer", list(LAYERS), index=0)
col, qcol, scale, mid, unit = LAYERS[layer]
only_sig = c2.toggle("Show only significant districts (q < 0.05)", value=False, disabled=qcol is None,
                     help="Grey out districts whose trend is not significant after FDR correction")

show = d.copy()
show["value"] = show[col]
hover = ["province"]
if qcol:
    show["q-value"] = show[qcol].round(3)
    hover.append("q-value")
    if only_sig:
        show.loc[show[qcol] >= 0.05, "value"] = np.nan
show["value"] = show["value"].round(3)

map_col, info_col = st.columns([3, 2], gap="large")
with map_col:
    fig = cm.choropleth(show, "value", scale, unit, hover=hover, midpoint=mid)
    fig.update_traces(selector=dict(type="choropleth"), marker_line_color="white")
    ev = st.plotly_chart(fig, width="stretch", on_select="rerun", selection_mode="points", key="hotmap")
    if qcol and only_sig:
        cm.caveat(f"{int(show['value'].notna().sum())} of 126 districts significant for this layer; others are blank.")

lookup = cm.district_lookup()
ids = list(lookup.values())
clicked = cm.clicked_district(ev)
if clicked:
    st.session_state["district_id"] = clicked
default_id = st.session_state.get("district_id", d.sort_values("Tmin_sen_slope_per_decade").iloc[-1]["district_id"])

with info_col:
    label = st.selectbox("District", list(lookup), index=ids.index(default_id))
    did = lookup[label]
    st.session_state["district_id"] = did
    r = d.set_index("district_id").loc[did]
    k = st.columns(3)
    k[0].metric("Elevation (m)", f"{r['elevation_m']:,.0f}")
    k[1].metric("Tmin trend", f"{r['Tmin_sen_slope_per_decade']:+.2f}", help=f"°C per decade · q = {r['Tmin_q_value']:.3f} · {r['Tmin_trend_class']}")
    k[2].metric("Tmax trend", f"{r['Tmax_sen_slope_per_decade']:+.2f}", help=f"°C per decade · q = {r['Tmax_q_value']:.3f} · {r['Tmax_trend_class']}")
    st.markdown(f"**{r['district']}**, {r['province']} · Tmin: *{r['Tmin_trend_class']}* · "
                f"Rainfall trend {r['Rain_sen_slope_per_decade']:+.0f} mm/yr per decade")

    a = annual[annual["district_id"] == did]
    an = m[m["district_id"] == did].groupby("year")[["tmax_c_anom", "tmin_c_anom"]].mean()
    fig = go.Figure()
    for c_, name, color in [("tmin_c_anom", "Tmin", cm.BLUE), ("tmax_c_anom", "Tmax", cm.ORANGE)]:
        fig.add_scatter(x=an.index, y=an[c_], name=name, mode="lines+markers", line=dict(color=color, width=2),
                        marker=dict(size=5), hovertemplate="%{x}: %{y:+.2f} °C")
    fig.add_hline(y=0, line_color=cm.MUTED, line_width=1)
    st.plotly_chart(cm.layout(fig, 280, title="Annual anomaly vs 2000–2019", yaxis_title="°C"), width="stretch")

cl = clim[clim["district_id"] == did]
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
g1, g2 = st.columns(2)
with g1:
    fig = go.Figure()
    fig.add_scatter(x=months, y=cl["tmax_clim"], name="Tmax", line=dict(color=cm.ORANGE, width=2), hovertemplate="%{y:.1f} °C")
    fig.add_scatter(x=months, y=cl["tmin_clim"], name="Tmin", line=dict(color=cm.BLUE, width=2), hovertemplate="%{y:.1f} °C")
    st.plotly_chart(cm.layout(fig, 280, title=f"Seasonal cycle, {r['district']} (2000–2019)", yaxis_title="°C"), width="stretch")
with g2:
    fig = go.Figure(go.Bar(x=months, y=cl["rain_clim"], marker_color=cm.BLUE, hovertemplate="%{y:.0f} mm"))
    st.plotly_chart(cm.layout(fig, 280, title="Mean monthly rainfall (CHIRPS)", yaxis_title="mm"), width="stretch")

st.subheader("District ranking")
rank_col = col if qcol else "Tmin_sen_slope_per_decade"
tab = d[["district", "province", "elevation_m", rank_col] + ([qcol] if qcol else ["Tmin_q_value"])].sort_values(rank_col, ascending=False)
tab.columns = ["District", "Province", "Elevation (m)", f"{layer if qcol else 'Tmin trend'} ({unit if qcol else '°C / decade'})", "q-value"]
st.dataframe(tab.round(3), hide_index=True, width="stretch", height=300)
