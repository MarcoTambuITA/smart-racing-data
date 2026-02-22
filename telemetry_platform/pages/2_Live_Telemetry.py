"""
Page 2 — Live / Recent Telemetry
=================================
Shows the most recently completed lap with:
  - X/Y track map colored by speed (Plotly)
  - Synced Speed vs Distance subplot
"""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

API = "http://localhost:8000"

st.title("📡 Live Telemetry")

# ── Get latest lap ───────────────────────────────────────────────────────────
try:
    lap_resp = requests.get(f"{API}/api/laps/latest", timeout=3)
    if lap_resp.status_code == 404:
        st.info("No completed laps yet. Waiting for data...")
        if st.button("🔄 Refresh"):
            st.rerun()
        st.stop()
    lap = lap_resp.json()
except Exception:
    st.error("Cannot connect to backend API.")
    st.stop()

st.markdown(f"**Lap {lap['lap_number']}** — Time: **{lap.get('lap_time_s', 0):.2f}s**")

# ── Fetch telemetry ──────────────────────────────────────────────────────────
try:
    telem = requests.get(f"{API}/api/laps/{lap['id']}/telemetry", timeout=5).json()
except Exception:
    st.warning("Could not fetch telemetry data.")
    st.stop()

df = pd.DataFrame(telem)
if df.empty or "x_m" not in df.columns:
    st.warning("Telemetry data not yet processed.")
    st.stop()

# ── Clean up ─────────────────────────────────────────────────────────────────
df = df.dropna(subset=["x_m", "y_m"])
x = df["x_m"].values
y = df["y_m"].values
speed = df["speed_mps"].values
dist = df["distance_m"].values

# ═══════════════════════════════════════════════════════════════════════════════
# PLOTLY — Track Map + Speed vs Distance
# ═══════════════════════════════════════════════════════════════════════════════
fig = make_subplots(
    rows=1, cols=2,
    column_widths=[0.55, 0.45],
    subplot_titles=("Track Map (colored by speed)", "Speed vs Distance"),
    specs=[[{"type": "scatter"}, {"type": "scatter"}]],
)

# ── Left: Track map ──────────────────────────────────────────────────────────
fig.add_trace(
    go.Scatter(
        x=x, y=y,
        mode="markers+lines",
        marker=dict(
            color=speed,
            colorscale="RdYlGn",
            size=4,
            colorbar=dict(title="Speed<br>(m/s)", x=0.48, len=0.9),
        ),
        line=dict(color="rgba(100,100,100,0.3)", width=1),
        name="Track",
        hovertemplate="X: %{x:.1f}m<br>Y: %{y:.1f}m<br>Speed: %{marker.color:.1f} m/s",
    ),
    row=1, col=1,
)
fig.update_xaxes(title_text="X (m)", row=1, col=1)
fig.update_yaxes(title_text="Y (m)", scaleanchor="x", row=1, col=1)

# Mark start
fig.add_trace(
    go.Scatter(
        x=[x[0]], y=[y[0]],
        mode="markers",
        marker=dict(color="white", size=12, symbol="star"),
        name="Start/Finish",
        showlegend=True,
    ),
    row=1, col=1,
)

# ── Right: Speed vs Distance ────────────────────────────────────────────────
fig.add_trace(
    go.Scatter(
        x=dist, y=speed,
        mode="lines",
        line=dict(color="#2ca02c", width=2),
        name="Speed",
        hovertemplate="Dist: %{x:.0f}m<br>Speed: %{y:.1f} m/s",
    ),
    row=1, col=2,
)
fig.update_xaxes(title_text="Distance (m)", row=1, col=2)
fig.update_yaxes(title_text="Speed (m/s)", row=1, col=2)

# ── Layout ───────────────────────────────────────────────────────────────────
fig.update_layout(
    height=550,
    template="plotly_dark",
    paper_bgcolor="#1a1a2e",
    plot_bgcolor="#16213e",
    font=dict(color="white"),
    showlegend=True,
    legend=dict(orientation="h", y=-0.15),
    margin=dict(t=40, b=60),
)

st.plotly_chart(fig, use_container_width=True)

# ── Metrics row ──────────────────────────────────────────────────────────────
st.markdown("---")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Lap Time", f"{lap.get('lap_time_s', 0):.2f}s")
c2.metric("Max Speed", f"{speed.max():.1f} m/s")
c3.metric("Avg Speed", f"{speed.mean():.1f} m/s")
c4.metric("Track Length", f"{dist.max():.0f} m")

# ── Battery plot ─────────────────────────────────────────────────────────────
if "battery_v" in df.columns and df["battery_v"].notna().any():
    with st.expander("🔋 Battery Voltage"):
        batt_fig = go.Figure()
        batt_fig.add_trace(go.Scatter(
            x=dist, y=df["battery_v"].values,
            mode="lines", line=dict(color="#ff6b6b", width=2),
        ))
        batt_fig.update_layout(
            height=250,
            template="plotly_dark",
            paper_bgcolor="#1a1a2e",
            xaxis_title="Distance (m)",
            yaxis_title="Battery (V)",
            margin=dict(t=10),
        )
        st.plotly_chart(batt_fig, use_container_width=True)

# ── Auto-refresh ─────────────────────────────────────────────────────────────
if st.checkbox("🔄 Auto-refresh (5s)"):
    import time
    time.sleep(5)
    st.rerun()
