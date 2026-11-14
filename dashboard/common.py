"""Shared helpers for the dashboard: cached data loading, colours and the district map."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

DATA = Path(__file__).resolve().parent / "data"

# Same palette as the paper figures (src/plotting.py)
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
INK_2, MUTED = "#52514e", "#898781"
DIVERGING = [[0, "#104281"], [.17, "#2a78d6"], [.33, "#9ec5f4"], [.5, "#f0efec"],
             [.67, "#f2a3a2"], [.83, "#e34948"], [1, "#9c2121"]]
DIVERGING_R = [[1 - p, c] for p, c in DIVERGING][::-1]        # blue = more (rain), red = less
SEQ_ORANGE = [[0, "#fde8dd"], [.25, "#f6b393"], [.5, "#eb6834"], [.75, "#c24a1a"], [1, "#8a320f"]]
SEQ_BLUE = [[0, "#cde2fb"], [.25, "#86b6ef"], [.5, "#3987e5"], [.75, "#256abf"], [1, "#0d366b"]]
MODEL_COLORS = {"Ridge": BLUE, "Random Forest": ORANGE, "XGBoost": AQUA, "SVR": YELLOW,
                "Climatology": MUTED, "Persistence": "#c3c2b7", "Damped persistence": INK_2}
PROVINCE_ORDER = ["Punjab", "Sindh", "Khyber Pakhtunkhwa", "Balochistan",
                  "Gilgit-Baltistan", "Azad Kashmir", "Islamabad Capital Territory"]


@st.cache_data
def geojson() -> dict:
    return json.load(open(DATA / "districts.geojson"))


@st.cache_data
def districts() -> pd.DataFrame:
    return pd.read_csv(DATA / "districts.csv")


@st.cache_data
def monthly() -> pd.DataFrame:
    return pd.read_parquet(DATA / "monthly.parquet")


@st.cache_data
def clim_monthly() -> pd.DataFrame:
    return pd.read_csv(DATA / "clim_monthly.csv")


@st.cache_data
def predictions() -> pd.DataFrame:
    return pd.read_parquet(DATA / "test_predictions.parquet")


@st.cache_data
def forecast() -> pd.DataFrame:
    return pd.read_csv(DATA / "forecast.csv", parse_dates=["input_date", "target_date"])


@st.cache_data
def metrics(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / f"metrics_{name}.csv")


@st.cache_data
def meta() -> dict:
    return json.load(open(DATA / "meta.json"))


def district_lookup() -> dict[str, str]:
    """'District (Province)' label -> district_id, sorted by name."""
    d = districts().sort_values("district")
    return {f"{r.district} ({r.province})": r.district_id for r in d.itertuples()}


def choropleth(df: pd.DataFrame, value: str, colorscale, label: str, hover: list[str] | None = None,
               midpoint: float | None = None, vmax: float | None = None, height: int = 560) -> go.Figure:
    """District map. Diverging scales are made symmetric around `midpoint`."""
    rng = None
    if midpoint is not None:
        v = vmax or float(np.nanpercentile(np.abs(df[value] - midpoint), 98))
        rng = (midpoint - v, midpoint + v)
    fig = px.choropleth(df, geojson=geojson(), locations="district_id", featureidkey="properties.district_id",
                        color=value, color_continuous_scale=colorscale, range_color=rng,
                        hover_name="district", hover_data={"district_id": False, **{h: True for h in (hover or [])}},
                        labels={value: label})
    # Fixed Pakistan extent (fitbounds leaves a lot of empty space with this projection)
    fig.update_geos(visible=False, bgcolor="rgba(0,0,0,0)", projection_type="mercator",
                    lonaxis_range=[60.6, 77.9], lataxis_range=[23.6, 37.2])
    fig.update_traces(marker_line_color="white", marker_line_width=0.4)
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)",
                      coloraxis_colorbar=dict(title=label, thickness=12, len=0.6))
    return fig


def clicked_district(event) -> str | None:
    """district_id clicked on a map (st.plotly_chart on_select event), if any."""
    try:
        pts = event.selection.points
        return pts[0].get("location") if pts else None
    except AttributeError:
        return None


def layout(fig: go.Figure, height: int = 340, **kw) -> go.Figure:
    opts = dict(height=height, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.2),
                hovermode="x unified")
    opts.update(kw)                       # callers may override any default, e.g. hovermode="closest"
    fig.update_layout(**opts)
    return fig


def caveat(text: str) -> None:
    st.caption(f"ⓘ {text}")
