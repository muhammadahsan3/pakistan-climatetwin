import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import common as cm

meta = cm.meta()
fc = cm.forecast()
pred = cm.predictions()
m = cm.anomalies_with_live()
ver = cm.verification()
fmonth = pd.Timestamp(meta["forecast_month"])
dend = cm.latest_data_month()
lo, hi = meta["interval_10_90"]

st.title("Forecast & model comparison")
st.markdown(f"One-month-ahead forecasts of the **Tmax anomaly** (departure from the district's 2000–2019 average for that month). "
            f"Best configuration: **{meta['best_model']}**, trained on 2000–2025. Latest data: **{dend:%B %Y}**"
            + (f" · updated {meta['updated_at']}" if "updated_at" in meta else "") + ".")

tabs = [f"Forecast for {fmonth:%B %Y}"] + (["Live check 2026 (unseen months)"] if ver is not None else []) + \
       ["How good are the models?", "Test period, district by district"]
tab1, *rest = st.tabs(tabs)
tab_live = rest.pop(0) if ver is not None else None
tab2, tab3 = rest

# ---------------------------------------------------------------------------------------------
with tab1:
    show = fc.copy()
    show["Predicted anomaly (°C)"] = show["pred_anom"].round(2)
    show["Predicted Tmax (°C)"] = show["pred_tmax"].round(1)
    show["80% range (°C)"] = show.apply(lambda r: f"{r.pred_tmax_lo:.1f} to {r.pred_tmax_hi:.1f}", axis=1)
    k = st.columns(4)
    k[0].metric("National mean forecast anomaly", f"{fc['pred_anom'].mean():+.2f} °C")
    k[1].metric(f"Observed anomaly, {dend:%b %Y}", f"{fc['tmax_anom_l0'].mean():+.2f} °C",
                help="The latest month the forecast is based on")
    k[2].metric("Districts forecast warmer than normal", f"{int((fc['pred_anom'] > 0).sum())} / {len(fc)}")
    k[3].metric("Typical error (test RMSE)", f"± {meta['best_test_rmse']:.2f} °C")
    left, right = st.columns([3, 2], gap="large")
    with left:
        fig = cm.choropleth(show, "Predicted anomaly (°C)", cm.DIVERGING, "°C", hover=["province", "Predicted Tmax (°C)", "80% range (°C)"],
                            midpoint=0, vmax=1.5, height=520)
        ev = st.plotly_chart(fig, width="stretch", on_select="rerun", selection_mode="points", key="fcmap")
    lookup = cm.district_lookup()
    ids = list(lookup.values())
    if cm.clicked_district(ev):
        st.session_state["district_id"] = cm.clicked_district(ev)
    did = st.session_state.get("district_id", ids[0])
    with right:
        label = st.selectbox("District", list(lookup), index=ids.index(did), key="fc_district")
        did = lookup[label]
        st.session_state["district_id"] = did
        r = fc.set_index("district_id").loc[did]
        st.markdown(f"**{r['district']}**: forecast mean daily maximum for {fmonth:%B %Y} is **{r['pred_tmax']:.1f} °C** "
                    f"(normal {r['target_tmax_clim']:.1f} °C; anomaly {r['pred_anom']:+.2f} °C). "
                    f"80% of test-period errors would put it between {r['pred_tmax_lo']:.1f} and {r['pred_tmax_hi']:.1f} °C.")
        h = m[m["district_id"] == did].tail(12)
        fig = go.Figure()
        fig.add_bar(x=h["date"], y=h["tmax_c_anom"], name="Observed", marker_color=[cm.RED if v > 0 else cm.BLUE for v in h["tmax_c_anom"]],
                    hovertemplate="%{x|%b %Y}: %{y:+.2f} °C")
        fig.add_scatter(x=[fmonth], y=[r["pred_anom"]], name="Forecast", mode="markers",
                        marker=dict(size=13, color=cm.INK_2, symbol="diamond"),
                        error_y=dict(type="data", array=[hi], arrayminus=[-lo], color=cm.INK_2, thickness=1.5),
                        hovertemplate="Forecast %{x|%b %Y}: %{y:+.2f} °C")
        fig.add_scatter(x=[fmonth], y=[r["damped_anom"]], name="Damped persistence", mode="markers",
                        marker=dict(size=9, color=cm.MUTED, symbol="circle-open", line=dict(width=2)),
                        hovertemplate="Baseline: %{y:+.2f} °C")
        fig.add_hline(y=0, line_color=cm.MUTED, line_width=1)
        st.plotly_chart(cm.layout(fig, 330, title="Last 12 months and next-month forecast (anomaly)", yaxis_title="°C"),
                        width="stretch")
    cm.caveat("The forecast is cautious by design. With weak month-to-month predictability, the best estimate stays close to normal, "
              "and the error band is wide. It is a statistical one-month outlook, not a weather forecast or a climate projection."
              + (f" Rainfall after {pd.Timestamp(meta['chirps_data_end']):%B %Y} is estimated from bias-corrected ERA5-Land until CHIRPS publishes it."
                 if "chirps_data_end" in meta else ""))

# ---------------------------------------------------------------------------------------------
if tab_live is not None:
    with tab_live:
        s = meta.get("live_verification", {})
        st.markdown(f"Every month since January 2026 the twin re-issues the forecast it would have made with the data available at the time, "
                    f"then compares it with what ERA5-Land observed. **None of these {s.get('months', 0)} months were used to train or tune the model**, "
                    "so this is a genuine out-of-sample test that grows by one month with every update.")
        k = st.columns(4)
        k[0].metric("Months verified", f"{s.get('months', 0)}")
        k[1].metric("Model RMSE", f"{s.get('rmse_model', float('nan')):.2f} °C")
        k[2].metric("Skill vs climatology", f"{s.get('skill_vs_clim', float('nan')):+.1%}",
                    help=f"Climatology RMSE {s.get('rmse_clim', float('nan')):.2f} °C")
        k[3].metric("Skill vs damped persistence", f"{s.get('skill_vs_damped', float('nan')):+.1%}",
                    help=f"Damped-persistence RMSE {s.get('rmse_damped', float('nan')):.2f} °C")
        nat = ver.groupby("target_date")[["target_anom", "pred_anom", "damped_anom"]].mean()
        fig = go.Figure()
        fig.add_bar(x=nat.index, y=nat["target_anom"], name="Observed", marker_color=[cm.RED if v > 0 else cm.BLUE for v in nat["target_anom"]],
                    hovertemplate="%{x|%b %Y}: %{y:+.2f} °C")
        fig.add_scatter(x=nat.index, y=nat["pred_anom"], name=meta["best_model"], mode="lines+markers",
                        line=dict(color=cm.INK_2, width=2.5), marker=dict(size=8, symbol="diamond"), hovertemplate="%{y:+.2f} °C")
        fig.add_scatter(x=nat.index, y=nat["damped_anom"], name="Damped persistence", mode="lines+markers",
                        line=dict(color=cm.MUTED, width=1.5, dash="dot"), hovertemplate="%{y:+.2f} °C")
        fig.add_hline(y=0, line_color=cm.MUTED, line_width=1)
        st.plotly_chart(cm.layout(fig, 380, title="National mean Tmax anomaly: forecast one month ahead vs observed", yaxis_title="°C"),
                        width="stretch")
        per = ver.assign(err_m=(ver["pred_anom"] - ver["target_anom"]) ** 2, err_c=ver["target_anom"] ** 2) \
                 .groupby("target_date")[["err_m", "err_c"]].mean() ** 0.5
        per.index = per.index.strftime("%b %Y")
        per.columns = ["Model RMSE (°C)", "Climatology RMSE (°C)"]
        per["Model better?"] = np.where(per["Model RMSE (°C)"] < per["Climatology RMSE (°C)"], "yes", "no")
        per.index.name = "Month"
        st.dataframe(per.round(2), width="stretch")
        cm.caveat("Months with sudden heat waves or cold spells (e.g. a very warm February 2026) are missed by every method: "
                  "one-month-ahead skill comes from persistence and slow drivers, not from weather events.")

# ---------------------------------------------------------------------------------------------
with tab2:
    reg = cm.metrics("test_metrics_regression").set_index("model")
    oc = cm.metrics("ocean_test_comparison").set_index("model")
    rows = []
    for n in ["Climatology", "Persistence", "Damped persistence"]:
        rows.append({"Model": n, "Features": "baseline", "RMSE (°C)": reg.loc[n, "RMSE"], "Skill vs climatology": reg.loc[n, "Skill vs climatology"], "Bias (°C)": reg.loc[n, "Bias"]})
    for n in ["Ridge", "Random Forest", "XGBoost", "SVR"]:
        rows.append({"Model": n, "Features": "local", "RMSE (°C)": oc.loc[n, "RMSE local"], "Skill vs climatology": oc.loc[n, "skill vs clim, local"], "Bias (°C)": oc.loc[n, "bias local"]})
        rows.append({"Model": n, "Features": "local + ocean", "RMSE (°C)": oc.loc[n, "RMSE + ocean"], "Skill vs climatology": oc.loc[n, "skill vs clim, + ocean"], "Bias (°C)": oc.loc[n, "bias + ocean"]})
    t = pd.DataFrame(rows)
    left, right = st.columns([3, 2], gap="large")
    with left:
        t2 = t[t["Model"] != "Persistence"].copy()
        t2["label"] = t2["Model"] + np.where(t2["Features"] == "local + ocean", " + ocean", "")
        t2 = t2.sort_values("Skill vs climatology")
        colors = [cm.MODEL_COLORS.get(mn, cm.INK_2) for mn in t2["Model"]]
        fig = go.Figure(go.Bar(x=t2["Skill vs climatology"] * 100, y=t2["label"], orientation="h",
                               marker=dict(color=colors, opacity=[1 if f != "local" else .55 for f in t2["Features"]]),
                               text=[f"{v:+.1f}%" for v in t2["Skill vs climatology"] * 100], textposition="outside",
                               hovertemplate="%{y}: %{x:+.1f}%<extra></extra>"))
        fig.add_vline(x=0, line_color=cm.INK_2, line_width=1)
        st.plotly_chart(cm.layout(fig, 420, title="Skill over climatology, test 2023–2025 (higher is better)",
                                  xaxis_title="MSE skill score (%)", hovermode="closest"), width="stretch")
    with right:
        st.markdown("""
**How to read this**
- *Climatology* always predicts "normal" (anomaly 0): the zero line.
- *Damped persistence* predicts 31% of this month's anomaly carries over. It is a one-parameter statistical baseline that is hard to beat.
- Lighter bars use local ERA5-Land/CHIRPS features only; full-colour bars add **ENSO, IOD and tropical-ocean** indices.
- With 36 test months, the 95% confidence intervals of all ML-vs-baseline differences include zero. Only Ridge's gain from ocean drivers is (just) significant.
""")
    st.dataframe(t.style.format({"RMSE (°C)": "{:.3f}", "Skill vs climatology": "{:+.1%}", "Bias (°C)": "{:+.2f}"}),
                 hide_index=True, width="stretch")
    a, b = st.columns(2)
    with a:
        st.markdown("**Extreme-heat months** (anomaly above the training-period 90th percentile)")
        ex = cm.metrics("test_metrics_extreme_heat").rename(columns={"model": "Model"})
        st.dataframe(ex[["Model", "Precision", "Recall", "F1", "PR-AUC"]].round(3), hide_index=True, width="stretch")
    with b:
        st.markdown("**Spatial generalisation** (province left out of training)")
        lp = cm.metrics("lopo_results").groupby("model")[["skill (seen)", "skill (held out)"]].mean().reindex(["Ridge", "Random Forest", "XGBoost", "SVR"])
        st.dataframe(lp.round(3), width="stretch")

# ---------------------------------------------------------------------------------------------
with tab3:
    lookup = cm.district_lookup()
    ids = list(lookup.values())
    c1, c2 = st.columns([2, 3])
    label = c1.selectbox("District", list(lookup), index=ids.index(st.session_state.get("district_id", ids[0])), key="test_district")
    did = lookup[label]
    choices = ["Ridge + ocean", "XGBoost + ocean", "Random Forest + ocean", "SVR + ocean", "Ridge", "Damped persistence"]
    sel = c2.multiselect("Models", choices, default=["Ridge + ocean", "Damped persistence"])
    p = pred[pred["district_id"] == did].sort_values("target_date")
    fig = go.Figure()
    fig.add_scatter(x=p["target_date"], y=p["target_anom"], name="Observed", line=dict(color="#0b0b0b", width=2.5),
                    hovertemplate="%{x|%b %Y}: %{y:+.2f} °C")
    for n in sel:
        base = n.replace(" + ocean", "")
        fig.add_scatter(x=p["target_date"], y=p[n], name=n, line=dict(color=cm.MODEL_COLORS.get(base, cm.INK_2), width=2,
                        dash="dot" if base == "Damped persistence" else "solid"), hovertemplate="%{y:+.2f} °C")
    fig.add_hline(y=0, line_color=cm.MUTED, line_width=1)
    st.plotly_chart(cm.layout(fig, 380, title=f"One-month-ahead Tmax anomaly, {label}", yaxis_title="°C"), width="stretch")
    if sel:
        rm = {n: float(np.sqrt(np.mean((p[n] - p["target_anom"]) ** 2))) for n in sel}
        rm["Climatology"] = float(np.sqrt(np.mean(p["target_anom"] ** 2)))
        st.dataframe(pd.DataFrame({"RMSE in this district (°C)": rm}).T.round(3), width="stretch")
