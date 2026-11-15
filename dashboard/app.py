"""
Pakistan ClimateTwin - interactive climate digital twin.

Run from the project folder (after `python dashboard/prepare_data.py`):
    streamlit run dashboard/app.py
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))   # so pages can `import common`

st.set_page_config(page_title="Pakistan ClimateTwin", page_icon="🌡️", layout="wide")

pages = st.navigation([
    st.Page("views/overview.py", title="Overview", icon=":material/home:", default=True),
    st.Page("views/hotspots.py", title="Climate map & hotspots", icon=":material/map:"),
    st.Page("views/forecast.py", title="Forecast & models", icon=":material/insights:"),
    st.Page("views/scenario.py", title="Scenario explorer", icon=":material/tune:"),
])
with st.sidebar:
    import common as cm
    st.caption("Pakistan ClimateTwin · research prototype\n\nData: ERA5-Land, CHIRPS v3, NOAA OISST via Google Earth Engine · "
               f"126 districts · 2000–2025 + live updates through {cm.latest_data_month():%b %Y}")
pages.run()
