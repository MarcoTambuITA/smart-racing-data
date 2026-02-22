"""
Page 1 — Driver Selection & Hub
================================
Select a driver, view their lap history and profiling metrics.
"""

import streamlit as st
import requests
import pandas as pd

API = "http://localhost:8000"

st.title("🧑‍✈️ Driver Hub")

# ── Fetch drivers ────────────────────────────────────────────────────────────
try:
    drivers = requests.get(f"{API}/api/drivers", timeout=3).json()
except Exception:
    st.error("Cannot connect to backend API.")
    st.stop()

if not drivers:
    st.info("No drivers registered yet. Add drivers via the API.")

    with st.expander("➕ Register a new driver"):
        with st.form("new_driver"):
            name = st.text_input("Name")
            did = st.number_input("LoRa Driver ID (0-15)", 0, 15, 1)
            weight = st.number_input("Weight (kg)", 50.0, 150.0, 70.0)
            if st.form_submit_button("Register"):
                resp = requests.post(f"{API}/api/drivers", json={
                    "driver_id": did, "name": name, "weight_kg": weight,
                })
                if resp.ok:
                    st.success(f"✓ Registered {name}")
                    st.rerun()
                else:
                    st.error(resp.json().get("detail", "Error"))
    st.stop()

# ── Driver selector ──────────────────────────────────────────────────────────
driver_map = {f"{d['name']} (ID: {d['driver_id']})": d for d in drivers}
selected_key = st.selectbox("Select Driver", list(driver_map.keys()))
driver = driver_map[selected_key]

st.session_state["selected_driver"] = driver

# ── Driver card ──────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
col1.metric("Driver ID", driver["driver_id"])
col2.metric("Name", driver["name"])
col3.metric("Weight", f"{driver.get('weight_kg', '—')} kg")

# ── Lap history ──────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📋 Lap History")

try:
    laps = requests.get(f"{API}/api/drivers/{driver['id']}/laps", timeout=3).json()
except Exception:
    laps = []

if not laps:
    st.info(f"No laps recorded for {driver['name']} yet.")
    st.stop()

df = pd.DataFrame(laps)

# Format for display
display_cols = ["lap_number", "lap_time_s", "coast_efficiency",
                "throttle_smoothness", "trail_braking_pct", "started_at"]
available = [c for c in display_cols if c in df.columns]
st.dataframe(
    df[available].style.format({
        "lap_time_s": "{:.2f}s",
        "coast_efficiency": "{:.3f}",
        "throttle_smoothness": "{:.3f}",
        "trail_braking_pct": "{:.1%}",
    }, na_rep="—"),
    use_container_width=True,
)

# ── Summary metrics ──────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📊 Summary")

valid_laps = df[df["lap_time_s"].notna()]
if not valid_laps.empty:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Laps", len(valid_laps))
    c2.metric("Best Lap", f"{valid_laps['lap_time_s'].min():.2f}s")
    c3.metric("Avg Lap", f"{valid_laps['lap_time_s'].mean():.2f}s")
    if "coast_efficiency" in valid_laps.columns:
        avg_coast = valid_laps["coast_efficiency"].dropna().mean()
        c4.metric("Avg Coast Eff", f"{avg_coast:.3f}" if pd.notna(avg_coast) else "—")
