"""
processing.py — Background task wrappers for offline script integration.

These functions are called by FastAPI background tasks when a lap completes.
They wrap the existing standalone scripts without modifying them.
"""

from __future__ import annotations

import json
import math
import sys
import os
from datetime import datetime
from typing import List, Optional

import numpy as np

# Add project root to path so we can import existing scripts
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auto_lap import GPSPoint, haversine
from models import SessionLocal, Lap, TelemetryData, TrackTemplate


# ═══════════════════════════════════════════════════════════════════════════════
# COORDINATE CONVERSION
# ═══════════════════════════════════════════════════════════════════════════════
_R_EARTH = 6_371_000.0


def latlon_to_xy(
    lat: np.ndarray, lon: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert lat/lon arrays to local X/Y (metres) using equirectangular
    projection.  Origin = first point.

    Mirrors load_telemetry._latlon_to_xy() but is importable.
    """
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    x_m = (lon_rad - lon_rad[0]) * _R_EARTH * np.cos(lat_rad[0])
    y_m = (lat_rad - lat_rad[0]) * _R_EARTH
    return x_m, y_m


def compute_cumulative_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Cumulative 2D distance from first point."""
    dx = np.diff(x, prepend=x[0])
    dy = np.diff(y, prepend=y[0])
    return np.cumsum(np.sqrt(dx**2 + dy**2))


# ═══════════════════════════════════════════════════════════════════════════════
# PROCESS COMPLETED LAP
# ═══════════════════════════════════════════════════════════════════════════════
def process_completed_lap(
    lap_id: int,
    buffer: List[GPSPoint],
) -> None:
    """
    Called as a background task when auto-lap detects a lap completion.

    1. Convert lat/lon → local X/Y
    2. Compute cumulative distance
    3. Store derived fields in telemetry_data rows
    4. Compute lap time
    """
    if not buffer or len(buffer) < 2:
        return

    db = SessionLocal()
    try:
        lap = db.query(Lap).filter(Lap.id == lap_id).first()
        if not lap:
            return

        # Arrays for bulk computation
        lats = np.array([p.lat for p in buffer])
        lons = np.array([p.lon for p in buffer])
        times = np.array([p.timestamp_s for p in buffer])
        speeds = np.array([p.speed_mps for p in buffer])

        x_m, y_m = latlon_to_xy(lats, lons)
        dist_m = compute_cumulative_distance(x_m, y_m)

        # Update lap metadata
        lap.lap_time_s = times[-1] - times[0]
        lap.started_at = datetime.utcfromtimestamp(times[0])
        lap.finished_at = datetime.utcfromtimestamp(times[-1])

        # Bulk-insert telemetry rows with derived fields
        telemetry_rows = []
        for i, pt in enumerate(buffer):
            telemetry_rows.append(TelemetryData(
                lap_id=lap_id,
                timestamp=datetime.utcfromtimestamp(pt.timestamp_s),
                lat=pt.lat,
                lon=pt.lon,
                x_m=float(x_m[i]),
                y_m=float(y_m[i]),
                distance_m=float(dist_m[i]),
                speed_mps=pt.speed_mps,
                battery_v=pt.battery_v,
            ))
        db.bulk_save_objects(telemetry_rows)

        # Compute basic profiling metrics
        _compute_driver_metrics(lap, speeds, times)

        db.commit()
        print(f"✓ Processed lap {lap.lap_number}: {lap.lap_time_s:.2f}s, "
              f"{len(buffer)} points, {dist_m[-1]:.0f}m")

    except Exception as e:
        db.rollback()
        print(f"✗ Error processing lap {lap_id}: {e}")
    finally:
        db.close()


def _compute_driver_metrics(
    lap: Lap,
    speeds: np.ndarray,
    times: np.ndarray,
) -> None:
    """Fill in the driver-profiling metric columns on the Lap record."""
    dt = np.diff(times)
    dt[dt == 0] = 0.001  # avoid division by zero
    accel = np.diff(speeds) / dt

    # Coast efficiency: mean deceleration when coasting (no accel, no braking)
    # Approximation: segments where |accel| < 0.3 m/s² and speed > 1 m/s
    coast_mask = (np.abs(accel) < 0.3) & (speeds[:-1] > 1.0)
    if coast_mask.any():
        lap.coast_efficiency = float(np.mean(np.abs(accel[coast_mask])))

    # Throttle smoothness: std of acceleration derivative (jerk)
    if len(accel) > 2:
        jerk = np.diff(accel)
        lap.throttle_smoothness = float(np.std(jerk))


# ═══════════════════════════════════════════════════════════════════════════════
# STORE TRACK TEMPLATE
# ═══════════════════════════════════════════════════════════════════════════════
def store_track_template(
    session_id: int,
    sf_lat: float,
    sf_lon: float,
    centroid_lat: float,
    centroid_lon: float,
    track_length_m: float,
    trace: List[GPSPoint],
) -> None:
    """Persist the auto-learned track to the database."""
    db = SessionLocal()
    try:
        template = TrackTemplate(
            session_id=session_id,
            sf_lat=sf_lat,
            sf_lon=sf_lon,
            centroid_lat=centroid_lat,
            centroid_lon=centroid_lon,
            track_length_m=track_length_m,
            num_points=len(trace),
            trace_lats_json=json.dumps([p.lat for p in trace]),
            trace_lons_json=json.dumps([p.lon for p in trace]),
        )
        db.add(template)
        db.commit()
        print(f"✓ Track template stored: {track_length_m:.0f}m, "
              f"S/F=({sf_lat:.6f}, {sf_lon:.6f})")
    except Exception as e:
        db.rollback()
        print(f"✗ Error storing track template: {e}")
    finally:
        db.close()


# ═══════════════════════════════════════════════════════════════════════════════
# AI COACHING ADVICE (placeholder)
# ═══════════════════════════════════════════════════════════════════════════════
def generate_coaching_advice(
    delta_t: np.ndarray,
    s_grid: np.ndarray,
    v_ref: np.ndarray,
    v_driver: np.ndarray,
) -> str:
    """
    Analyze the delta-time array and produce automated coaching advice.

    This is a rule-based placeholder. Replace with an LLM call or more
    sophisticated analysis when ready.
    """
    advice_lines = []

    # Find the worst time-loss zones (where delta_t increases most steeply)
    if len(delta_t) < 20:
        return "Not enough data for coaching analysis."

    d_delta = np.diff(delta_t)
    window = max(10, len(d_delta) // 20)

    # Sliding window to find worst segments
    worst_loss = 0.0
    worst_idx = 0
    for i in range(len(d_delta) - window):
        loss = d_delta[i:i + window].sum()
        if loss > worst_loss:
            worst_loss = loss
            worst_idx = i

    if worst_loss > 0.05:
        sector_start = s_grid[worst_idx]
        sector_end = s_grid[min(worst_idx + window, len(s_grid) - 1)]
        speed_diff = float(np.mean(v_ref[worst_idx:worst_idx + window]
                                   - v_driver[worst_idx:worst_idx + window]))

        advice_lines.append(
            f"🔴 **Biggest time loss**: {worst_loss:.2f}s between "
            f"{sector_start:.0f}m–{sector_end:.0f}m."
        )
        if speed_diff > 0:
            advice_lines.append(
                f"   You're {speed_diff:.1f} m/s slower than the ghost here. "
                f"Try carrying more speed through this section."
            )

    # Check braking points: where driver decelerates earlier than ghost
    decel_ref = np.diff(v_ref)
    decel_drv = np.diff(v_driver)

    for i in range(5, len(decel_ref) - 5):
        # ref still accelerating but driver already braking
        if decel_ref[i] > 0.1 and decel_drv[i] < -0.3:
            brake_early_m = 0
            for j in range(i, min(i + 50, len(decel_ref))):
                if decel_ref[j] < -0.1:
                    brake_early_m = int(s_grid[j] - s_grid[i])
                    break
            if brake_early_m > 5:
                advice_lines.append(
                    f"⚠️ **Braking {brake_early_m}m too early** at "
                    f"{s_grid[i]:.0f}m. The ghost brakes later — "
                    f"trust your entry speed!"
                )
                break  # only report the first instance

    # Overall summary
    total_delta = delta_t[-1] - delta_t[0]
    if total_delta > 0:
        advice_lines.insert(0,
            f"📊 Overall: **{total_delta:.2f}s slower** than the ghost line."
        )
    elif total_delta < 0:
        advice_lines.insert(0,
            f"🟢 Overall: **{abs(total_delta):.2f}s faster** than the ghost! "
            f"Great lap."
        )

    return "\n\n".join(advice_lines) if advice_lines else "✅ Solid lap — no major issues detected."
