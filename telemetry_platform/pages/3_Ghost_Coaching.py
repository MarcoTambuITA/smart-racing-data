"""
Page 3 — AI Coaching & Ghost Comparison
========================================
Compare a driver's lap against the theoretical best ghost line.
Includes automated coaching advice via delta-time analysis.
"""

import sys
import os
import streamlit as st
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Allow import of processing module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

API = "http://localhost:8000"

st.title("👻 Ghost Comparison & AI Coaching")

# ═══════════════════════════════════════════════════════════════════════════════
# LAP SELECTION
# ═══════════════════════════════════════════════════════════════════════════════
driver = st.session_state.get("selected_driver")

if not driver:
    st.info("Please select a driver on the **Driver Hub** page first.")
    st.stop()

st.markdown(f"**Driver: {driver['name']}** (ID: {driver['driver_id']})")

try:
    laps = requests.get(f"{API}/api/drivers/{driver['id']}/laps", timeout=3).json()
except Exception:
    st.error("Cannot connect to backend API.")
    st.stop()

valid_laps = [l for l in laps if l.get("lap_time_s")]
if len(valid_laps) < 2:
    st.info("Need at least 2 completed laps for ghost comparison.")
    st.stop()

# Best lap = ghost reference
best_lap = min(valid_laps, key=lambda l: l["lap_time_s"])
other_laps = [l for l in valid_laps if l["id"] != best_lap["id"]]

st.markdown(f"🏆 **Ghost (Best Lap)**: Lap {best_lap['lap_number']} — "
            f"**{best_lap['lap_time_s']:.2f}s**")

selected = st.selectbox(
    "Compare lap:",
    other_laps,
    format_func=lambda l: f"Lap {l['lap_number']} — {l['lap_time_s']:.2f}s",
)

# ═══════════════════════════════════════════════════════════════════════════════
# FETCH TELEMETRY
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=60)
def fetch_telemetry(lap_id):
    resp = requests.get(f"{API}/api/laps/{lap_id}/telemetry", timeout=5)
    if resp.ok:
        return pd.DataFrame(resp.json())
    return pd.DataFrame()

df_ghost = fetch_telemetry(best_lap["id"])
df_driver = fetch_telemetry(selected["id"])

if df_ghost.empty or df_driver.empty:
    st.warning("Telemetry not yet processed for one or both laps.")
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# DELTA TIME COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════
# Resample both onto a common 1-meter distance grid
ds = 1.0
max_dist = min(df_ghost["distance_m"].max(), df_driver["distance_m"].max())
s_grid = np.arange(0, max_dist, ds)

v_ghost = np.interp(s_grid, df_ghost["distance_m"].values, df_ghost["speed_mps"].values)
v_driver = np.interp(s_grid, df_driver["distance_m"].values, df_driver["speed_mps"].values)

# Clamp speeds to avoid division by zero
v_ghost = np.maximum(v_ghost, 0.5)
v_driver = np.maximum(v_driver, 0.5)

# Cumulative delta time: ∫(1/v_driver - 1/v_ghost) ds
delta_t = np.cumsum((1.0 / v_driver - 1.0 / v_ghost) * ds)
delta_v = v_ghost - v_driver

# ═══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════════════════════════════════════
fig = make_subplots(
    rows=2, cols=1,
    row_heights=[0.5, 0.5],
    shared_xaxes=True,
    subplot_titles=(
        "Speed vs Distance",
        "Cumulative Time Delta (+ = slower than ghost)",
    ),
    vertical_spacing=0.08,
)

# ── Top: Speed comparison ────────────────────────────────────────────────────
fig.add_trace(go.Scatter(
    x=s_grid, y=v_ghost, mode="lines",
    line=dict(color="#00bfff", width=2),
    name=f"Ghost (Lap {best_lap['lap_number']})",
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=s_grid, y=v_driver, mode="lines",
    line=dict(color="#ff6b6b", width=2),
    name=f"Your Lap {selected['lap_number']}",
), row=1, col=1)

fig.update_yaxes(title_text="Speed (m/s)", row=1, col=1)

# ── Bottom: Delta time ──────────────────────────────────────────────────────
# Color: green when gaining (delta decreasing), red when losing
colors = np.where(
    np.gradient(delta_t) <= 0,
    "rgba(46, 204, 113, 0.8)",   # green = gaining
    "rgba(231, 76, 60, 0.8)",    # red   = losing
)

fig.add_trace(go.Bar(
    x=s_grid[::5], y=delta_t[::5],
    marker_color=colors[::5],
    name="Δt",
    showlegend=False,
    hovertemplate="Dist: %{x:.0f}m<br>Δt: %{y:+.3f}s",
), row=2, col=1)

fig.add_hline(y=0, line_dash="dash", line_color="white", row=2, col=1)
fig.update_yaxes(title_text="Δt (seconds)", row=2, col=1)
fig.update_xaxes(title_text="Distance (m)", row=2, col=1)

fig.update_layout(
    height=650,
    template="plotly_dark",
    paper_bgcolor="#1a1a2e",
    plot_bgcolor="#16213e",
    font=dict(color="white"),
    legend=dict(orientation="h", y=1.08),
    margin=dict(t=60, b=40),
)

st.plotly_chart(fig, use_container_width=True)

# ── Metrics ──────────────────────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
c1.metric("Ghost Time", f"{best_lap['lap_time_s']:.2f}s")
c2.metric("Your Time", f"{selected['lap_time_s']:.2f}s")

gap = selected["lap_time_s"] - best_lap["lap_time_s"]
c3.metric("Gap", f"{gap:+.2f}s", delta=f"{gap:+.2f}s",
          delta_color="inverse")

# ═══════════════════════════════════════════════════════════════════════════════
# AI COACHING ADVICE
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("🤖 AI Coaching Advice")

# Use the processing module's coaching generator
try:
    from processing import generate_coaching_advice
    advice = generate_coaching_advice(delta_t, s_grid, v_ghost, v_driver)
except ImportError:
    advice = _fallback_coaching(delta_t, s_grid)

st.markdown(advice)

# ── Track overlay comparison ─────────────────────────────────────────────────
with st.expander("🗺️ Track Overlay"):
    track_fig = go.Figure()
    track_fig.add_trace(go.Scatter(
        x=df_ghost["x_m"], y=df_ghost["y_m"],
        mode="lines", line=dict(color="#00bfff", width=3),
        name="Ghost",
    ))
    track_fig.add_trace(go.Scatter(
        x=df_driver["x_m"], y=df_driver["y_m"],
        mode="lines", line=dict(color="#ff6b6b", width=2, dash="dot"),
        name="Your Line",
    ))
    track_fig.update_layout(
        height=500,
        template="plotly_dark",
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#16213e",
        yaxis=dict(scaleanchor="x"),
        margin=dict(t=10),
    )
    st.plotly_chart(track_fig, use_container_width=True)
