#!/usr/bin/env python3
"""
generate_track_csv.py
=====================
Convert raw race telemetry (lat/lon) into a TUMFTM-compatible track map CSV.

Pipeline
--------
1. Load & clean telemetry via ``load_telemetry()``
2. Extract the **longest complete lap** (most data points)
3. Smooth X/Y with a Savitzky-Golay filter to remove GPS jitter
4. Enforce loop closure (snap last point → first point if gap < 5 m)
5. Add constant track-width columns (w_tr_right, w_tr_left)
6. Write ``# x_m,y_m,w_tr_right_m,w_tr_left_m`` CSV
7. Show a verification scatter plot

Usage
-----
    python3 generate_track_csv.py
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from load_telemetry import load_telemetry

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_CSV = os.path.join(_SCRIPT_DIR, "telemetry-main", "Data", "log.csv")
OUTPUT_CSV = os.path.join(
    _SCRIPT_DIR,
    "global_racetrajectory_optimization-master",
    "inputs",
    "tracks",
    "electrathon_track.csv",
)

TRACK_HALF_WIDTH = 2.0          # metres  (total width = 4.0 m)
LOOP_CLOSURE_THRESH = 5.0       # metres — max gap to auto-close
SAVGOL_WINDOW = 15              # must be odd; larger → smoother
SAVGOL_POLYORDER = 3            # polynomial order for Savitzky-Golay


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------
def generate_track(
    input_csv: str = INPUT_CSV,
    output_csv: str = OUTPUT_CSV,
    half_width: float = TRACK_HALF_WIDTH,
    closure_thresh: float = LOOP_CLOSURE_THRESH,
    savgol_window: int = SAVGOL_WINDOW,
    savgol_poly: int = SAVGOL_POLYORDER,
    show_plot: bool = True,
) -> pd.DataFrame:
    """
    End-to-end pipeline: telemetry CSV → TUMFTM track CSV.

    Returns the final DataFrame that was written to disk.
    """
    # ---- 1. Load telemetry ------------------------------------------------
    print(f"Loading telemetry from: {input_csv}")
    df = load_telemetry(input_csv)
    print(f"  Total points : {len(df)}")
    print(f"  Laps detected: {df['LapNumber'].nunique()}")

    # ---- 2. Pick the longest complete lap ---------------------------------
    lap_counts = df.groupby("LapNumber").size()
    best_lap = int(lap_counts.idxmax())
    lap_df = df[df["LapNumber"] == best_lap].reset_index(drop=True)
    print(f"  Using lap {best_lap}  ({len(lap_df)} points)")

    x_raw = lap_df["X"].values.copy()
    y_raw = lap_df["Y"].values.copy()

    # ---- 3. Savitzky-Golay smoothing --------------------------------------
    #  Ensure window ≤ number of points and is odd
    win = min(savgol_window, len(x_raw))
    if win % 2 == 0:
        win -= 1
    win = max(win, savgol_poly + 2)              # must be > polyorder

    x_smooth = savgol_filter(x_raw, window_length=win, polyorder=savgol_poly)
    y_smooth = savgol_filter(y_raw, window_length=win, polyorder=savgol_poly)
    print(f"  Savitzky-Golay filter applied  (window={win}, poly={savgol_poly})")

    # ---- 4. Loop closure --------------------------------------------------
    gap = np.hypot(x_smooth[-1] - x_smooth[0],
                   y_smooth[-1] - y_smooth[0])
    print(f"  First→Last gap: {gap:.2f} m", end="")

    if gap < closure_thresh:
        x_smooth[-1] = x_smooth[0]
        y_smooth[-1] = y_smooth[0]
        print(f"  → closed (snapped last point to origin)")
    else:
        print(f"  → NOT closed (gap > {closure_thresh} m threshold)")
        print("    ⚠  The track may not form a valid loop for the solver.")

    # ---- 5. Build output DataFrame ----------------------------------------
    track = pd.DataFrame({
        "# x_m": np.round(x_smooth, 4),
        "y_m":   np.round(y_smooth, 4),
        "w_tr_right_m": half_width,
        "w_tr_left_m":  half_width,
    })

    # ---- 6. Write CSV -----------------------------------------------------
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    track.to_csv(output_csv, index=False)
    print(f"\n✅  Track CSV written → {output_csv}")
    print(f"   {len(track)} points, width = ±{half_width} m")

    # ---- 7. Verification plot ---------------------------------------------
    if show_plot:
        _plot_track(x_raw, y_raw, x_smooth, y_smooth, half_width, output_csv)

    return track


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
def _plot_track(
    x_raw: np.ndarray,
    y_raw: np.ndarray,
    x_smooth: np.ndarray,
    y_smooth: np.ndarray,
    half_width: float,
    title_path: str,
) -> None:
    """Dark-themed verification plot: raw GPS vs. smoothed centre-line."""
    fig, ax = plt.subplots(figsize=(12, 9), facecolor="#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    # Raw GPS scatter (faint)
    ax.scatter(x_raw, y_raw, s=4, color="#555577", alpha=0.4, label="Raw GPS")

    # Smoothed centre-line
    ax.plot(x_smooth, y_smooth, color="#00d4ff", linewidth=2.0,
            label="Smoothed centre-line", zorder=3)

    # Track edges (offset by ±half_width along local normals)
    dx = np.gradient(x_smooth)
    dy = np.gradient(y_smooth)
    norm = np.hypot(dx, dy)
    norm[norm == 0] = 1.0
    nx = -dy / norm           # outward-left normal
    ny =  dx / norm

    x_left  = x_smooth + nx * half_width
    y_left  = y_smooth + ny * half_width
    x_right = x_smooth - nx * half_width
    y_right = y_smooth - ny * half_width

    ax.plot(x_left, y_left, color="#ff6b6b", linewidth=0.8, alpha=0.6,
            label=f"Track edge (±{half_width} m)")
    ax.plot(x_right, y_right, color="#ff6b6b", linewidth=0.8, alpha=0.6)

    # Start / Finish marker
    ax.plot(x_smooth[0], y_smooth[0], "o", color="white", markersize=10,
            zorder=5, label="Start / Finish")

    ax.set_aspect("equal")
    ax.set_xlabel("X  (m)", fontsize=12, color="white")
    ax.set_ylabel("Y  (m)", fontsize=12, color="white")
    ax.set_title(
        f"Track Map — {os.path.basename(title_path)}",
        fontsize=15, fontweight="bold", color="white", pad=14,
    )
    ax.tick_params(colors="white")
    ax.grid(True, color="#333355", linewidth=0.4, alpha=0.5)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.6,
              facecolor="#2d2d44", labelcolor="white")

    fig.tight_layout()

    # Save alongside the CSV
    png_path = os.path.splitext(title_path)[0] + "_preview.png"
    fig.savefig(png_path, dpi=150, facecolor=fig.get_facecolor(),
                bbox_inches="tight")
    print(f"   Preview plot saved → {png_path}")

    plt.show()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    generate_track()
