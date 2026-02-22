#!/usr/bin/env python3
"""
generate_synthetic_log.py
=========================
Generate synthetic GPS telemetry CSVs that mirror the real ``log.csv`` schema.
Supports **multiple track shapes** for comprehensive testing of sectorization,
ghost-lap, and driver-profiling algorithms.

Available tracks
----------------
* ``lemniscate`` — Figure-8 (Lemniscate of Bernoulli)
* ``oval``       — Simple elliptical oval
* ``kidney``     — D-shaped / kidney circuit with distinct slow & fast sections
* ``circuit``    — Procedurally generated circuit with random corners & straights

Usage
-----
    python3 generate_synthetic_log.py                              # defaults
    python3 generate_synthetic_log.py --track oval --laps 5
    python3 generate_synthetic_log.py --track circuit --size 200
    python3 generate_synthetic_log.py --track all                  # one CSV per shape
"""

from __future__ import annotations

import argparse
import math
import os
from typing import Optional

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_CENTRE_LAT = 28.0587           # degrees N  (Tampa, FL)
_CENTRE_LON = -82.4139          # degrees W
_METRES_PER_DEG_LAT = 110_540.0
_METRES_PER_DEG_LON_EQ = 111_320.0

# Speed limits (m/s)  —  5 mph ≈ 2.24 m/s, 40 mph ≈ 17.88 m/s
_V_MIN = 2.24
_V_MAX = 17.88

# Electrical model
_V_BATT_OPEN = 54.0   # volts (no load)
_R_INTERNAL = 0.05     # ohms  (sag coefficient)

# Registry of available tracks
TRACK_NAMES = ["lemniscate", "oval", "kidney", "circuit"]


# ═══════════════════════════════════════════════════════════════════════════
# Track geometry generators
# ═══════════════════════════════════════════════════════════════════════════

def _track_lemniscate(t: np.ndarray, a: float):
    """Figure-8: Lemniscate of Bernoulli.  S/F at rightmost cusp."""
    sin_t, cos_t = np.sin(t), np.cos(t)
    denom = 1.0 + sin_t ** 2
    return a * cos_t / denom, a * sin_t * cos_t / denom


def _track_oval(t: np.ndarray, a: float):
    """Elliptical oval — 2:1 aspect ratio.  S/F at 3-o'clock."""
    return a * np.cos(t), 0.5 * a * np.sin(t)


def _track_kidney(t: np.ndarray, a: float):
    """
    D-shaped / kidney curve.  One long sweeping bend + one tight hairpin.
    Built from a radius that varies sinusoidally with angle.
    """
    r = a * (1.0 + 0.5 * np.cos(t) + 0.15 * np.cos(2 * t))
    return r * np.cos(t), r * np.sin(t)


def _track_circuit(t: np.ndarray, a: float, seed: int = 7):
    """
    Procedurally generated circuit using cubic-spline interpolation
    through random control points arranged around a circle.

    Each run with the same *seed* produces the same track, so results
    are reproducible.
    """
    from scipy.interpolate import CubicSpline

    rng = np.random.default_rng(seed)
    n_ctrl = rng.integers(6, 12)             # 6–11 corners
    angles = np.sort(rng.uniform(0, 2 * np.pi, n_ctrl))

    # Random radii for each control point
    radii = a * (0.6 + 0.4 * rng.random(n_ctrl))

    # Control points in X/Y
    cx = radii * np.cos(angles)
    cy = radii * np.sin(angles)

    # Close the loop
    angles_ext = np.concatenate([angles, [angles[0] + 2 * np.pi]])
    cx_ext = np.concatenate([cx, [cx[0]]])
    cy_ext = np.concatenate([cy, [cy[0]]])

    # Cubic-spline interpolation (periodic boundary)
    cs_x = CubicSpline(angles_ext, cx_ext, bc_type='periodic')
    cs_y = CubicSpline(angles_ext, cy_ext, bc_type='periodic')

    # Map t → [0, 2π]  for one lap
    t_mod = t % (2 * np.pi)
    return cs_x(t_mod), cs_y(t_mod)


_TRACK_FN = {
    "lemniscate": _track_lemniscate,
    "oval":       _track_oval,
    "kidney":     _track_kidney,
    "circuit":    _track_circuit,
}


# ═══════════════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════════════

def _curvature(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Unsigned curvature κ = |x'y'' − y'x''| / (x'²+y'²)^(3/2)."""
    dx, dy = np.gradient(x), np.gradient(y)
    ddx, ddy = np.gradient(dx), np.gradient(dy)
    num = np.abs(dx * ddy - dy * ddx)
    den = np.maximum((dx ** 2 + dy ** 2) ** 1.5, 1e-12)
    return num / den


def _xy_to_latlon(x_m: np.ndarray, y_m: np.ndarray):
    """Local X/Y metres → lat/lon (inverse equirectangular)."""
    lat_ref_rad = math.radians(_CENTRE_LAT)
    lon = _CENTRE_LON + x_m / (math.cos(lat_ref_rad) * _METRES_PER_DEG_LON_EQ)
    lat = _CENTRE_LAT + y_m / _METRES_PER_DEG_LAT
    return lat, lon


def _speed_from_curvature(kappa: np.ndarray) -> np.ndarray:
    """Map curvature → speed.  High κ → slow, low κ → fast."""
    kappa_norm = (kappa - kappa.min()) / (kappa.max() - kappa.min() + 1e-12)
    speed = _V_MAX - (_V_MAX - _V_MIN) * kappa_norm
    return uniform_filter1d(speed, size=15, mode='wrap')


def _electrical(speed: np.ndarray, dt: float):
    """Battery-sag model: amps ∝ acceleration, voltage = 54 − 0.05·|I|."""
    accel = np.gradient(speed, dt)
    amps = np.clip(accel * 8.0, -5.0, 40.0)
    voltage = _V_BATT_OPEN - _R_INTERNAL * np.abs(amps)
    return amps, voltage


def _esptime_series(n: int, dt: float = 0.2) -> np.ndarray:
    return 1.0 + np.arange(n) * dt


def _datetime_columns(n: int, dt: float = 0.2):
    """Return (date_ddmmyy, time_hhmmsscs) integer arrays."""
    base = pd.Timestamp("2025-07-09 12:00:00")
    ts = base + pd.to_timedelta(np.arange(n) * dt, unit="s")
    date_int = ts.day * 10000 + ts.month * 100 + (ts.year - 2000)
    csec = (ts.microsecond // 10000).astype(int)
    time_int = (ts.hour * 100_000_000 + ts.minute * 1_000_000
                + ts.second * 10_000 + csec * 100)
    return date_int.values, time_int.values


# ═══════════════════════════════════════════════════════════════════════════
# Main generator
# ═══════════════════════════════════════════════════════════════════════════

def generate_synthetic_log(
    track: str = "lemniscate",
    n_laps: int = 3,
    pts_per_lap: int = 1000,
    track_size_m: float = 150.0,
    dt: float = 0.2,
    output_dir: Optional[str] = None,
) -> str:
    """
    Generate a synthetic telemetry CSV and return the file path.

    Parameters
    ----------
    track : str
        Track shape — one of ``"lemniscate"``, ``"oval"``, ``"kidney"``,
        ``"circuit"``.
    n_laps : int
        Number of complete laps.
    pts_per_lap : int
        Data points per lap (default 1000 ≈ 200 s at 5 Hz).
    track_size_m : float
        Characteristic size of the track in metres.
    dt : float
        Time step in seconds (0.2 = 5 Hz).
    output_dir : str or None
        Base directory for the CSV.  ``None`` → script directory.

    Returns
    -------
    str
        Absolute path to the generated CSV.
    """
    track = track.lower()
    if track not in _TRACK_FN:
        raise ValueError(
            f"Unknown track '{track}'. Choose from: {TRACK_NAMES}"
        )

    n_total = n_laps * pts_per_lap
    rng = np.random.default_rng(42)

    # ---- Parametric curve --------------------------------------------------
    t = np.linspace(0, n_laps * 2 * np.pi, n_total, endpoint=False)
    x_m, y_m = _TRACK_FN[track](t, a=track_size_m)

    # ---- Speed profile -----------------------------------------------------
    kappa = _curvature(x_m, y_m)
    speed = _speed_from_curvature(kappa)

    # Per-lap variation ±4 % so laps aren't identical
    for lap_i in range(n_laps):
        s, e = lap_i * pts_per_lap, (lap_i + 1) * pts_per_lap
        speed[s:e] *= 1.0 + rng.uniform(-0.04, 0.04)
    speed = np.clip(speed, _V_MIN, _V_MAX)

    # ---- GPS ---------------------------------------------------------------
    lat, lon = _xy_to_latlon(x_m, y_m)

    # ---- Electrical --------------------------------------------------------
    amps, voltage = _electrical(speed, dt)

    # ---- Timestamps --------------------------------------------------------
    esptime = _esptime_series(n_total, dt)
    date_col, time_col = _datetime_columns(n_total, dt)

    # ---- Assemble DataFrame ------------------------------------------------
    df = pd.DataFrame({
        "esptime":           esptime,
        "date_ddmmyy":       date_col,
        "time_hhmmsscs":     time_col,
        "latitude":          np.round(lat, 7),
        "longitude":         np.round(lon, 7),
        "speed":             np.round(speed, 6),
        "ca_amp_hour":       np.round(np.cumsum(np.abs(amps) * dt / 3600), 6),
        "ca_voltage":        np.round(voltage, 4),
        "ca_amperes":        np.round(amps, 4),
        "ca_speed":          np.round(speed * 2.23694, 4),      # m/s → mph
        "ca_distance":       np.round(np.cumsum(speed * dt), 4),
        "ca_temp_degC":      np.round(25.0 + rng.normal(0, 0.5, n_total), 2),
        "ca_PAS_RPM":        0.0,
        "ca_human_watts":    0.0,
        "ca_PAS_torque_Nm":  0.0,
        "ca_throttle_in":    np.round(np.clip(amps / 40.0, 0, 1), 4),
        "ca_throttle_out":   np.round(np.clip(amps / 40.0, 0, 1) * 0.95, 4),
    })

    # ---- Write CSV ---------------------------------------------------------
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(output_dir, "telemetry-main", "Data")
    os.makedirs(out_dir, exist_ok=True)

    # Auto-increment file name
    idx = 1
    while True:
        fname = f"synthetic_{track}_{idx}.csv"
        fpath = os.path.join(out_dir, fname)
        if not os.path.exists(fpath):
            break
        idx += 1

    df.to_csv(fpath, index=False)
    return fpath


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic GPS telemetry for testing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Track shapes:\n"
            "  lemniscate   Figure-8 (Lemniscate of Bernoulli)\n"
            "  oval         Simple elliptical oval (2:1 aspect)\n"
            "  kidney       D-shaped / kidney with tight hairpin\n"
            "  circuit      Procedural random corners & straights\n"
            "  all          Generate one CSV for every shape\n"
        ),
    )
    parser.add_argument("--track", type=str, default="lemniscate",
                        choices=TRACK_NAMES + ["all"],
                        help="Track shape (default: lemniscate)")
    parser.add_argument("--laps", type=int, default=3,
                        help="Number of laps (default 3)")
    parser.add_argument("--pts", type=int, default=1000,
                        help="Data points per lap (default 1000)")
    parser.add_argument("--size", type=float, default=150.0,
                        help="Track size in metres (default 150)")
    args = parser.parse_args()

    tracks = TRACK_NAMES if args.track == "all" else [args.track]

    for trk in tracks:
        path = generate_synthetic_log(
            track=trk,
            n_laps=args.laps,
            pts_per_lap=args.pts,
            track_size_m=args.size,
        )
        df = pd.read_csv(path)
        print(f"[{trk:12s}]  {path}")
        print(f"  Rows: {len(df):>5}   "
              f"Duration: {df['esptime'].iloc[-1] - df['esptime'].iloc[0]:.0f}s   "
              f"Speed: [{df['speed'].min():.1f}, {df['speed'].max():.1f}] m/s   "
              f"Voltage: [{df['ca_voltage'].min():.1f}, {df['ca_voltage'].max():.1f}] V")
