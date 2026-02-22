"""
app.py — Streamlit Dashboard Entry Point
=========================================
Multi-page Electrathon telemetry dashboard.

Run
---
    streamlit run app.py
"""

import streamlit as st

st.set_page_config(
    page_title="Electrathon Telemetry",
    page_icon="🏁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar branding ─────────────────────────────────────────────────────────
st.sidebar.markdown(
    """
    # 🏁 Electrathon
    ### Telemetry Dashboard
    ---
    **IEEE Racing Team**

    Navigate using the pages below.
    """
)

# ── Landing page ─────────────────────────────────────────────────────────────
st.title("🏁 Electrathon Telemetry Dashboard")
st.markdown(
    """
    Welcome to the automated telemetry platform.  Use the sidebar to navigate:

    | Page | Description |
    |---|---|
    | **🧑‍✈ Driver Hub** | Select a driver, view lap history & metrics |
    | **📡 Live Telemetry** | Recent lap track map + speed/distance plots |
    | **👻 Ghost & Coaching** | Compare against ghost line + AI advice |

    ---

    ### System Status
    """
)

# Quick health check
import requests
try:
    r = requests.get("http://localhost:8000/health", timeout=2)
    data = r.json()
    col1, col2 = st.columns(2)
    col1.metric("API Status", "🟢 Online")
    col2.metric("Active Sessions", data.get("active_sessions", 0))
except Exception:
    st.warning("⚠️ Backend API is not running. Start it with: `uvicorn main:app --reload`")
