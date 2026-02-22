#!/usr/bin/env python3
"""
delta_time_analysis.py
======================
Compare any lap against the **fastest lap** (Reference) using the
TUMFTM ``trajectory_planning_helpers`` (tph) library for path matching
and spline interpolation.

Produces a professional dual-panel plot:
    Top    — Speed vs. Distance  (Reference blue, comparison red)
    Bottom — Cumulative Time Delta  (green = gaining, red = losing)

Usage
-----
    python3 delta_time_analysis.py                                         # defaults
    python3 delta_time_analysis.py --csv telemetry-main/Data/log.csv
    python3 delta_time_analysis.py --compare-lap 2 --save delta_time.png
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.interpolate import interp1d

# Ensure the local project is importable
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

import trajectory_planning_helpers as tph
from load_telemetry import load_telemetry


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _cumulative_s(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return cumulative arc-length starting at 0."""
    ds = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    return np.concatenate([[0.0], np.cumsum(ds)])


def _build_reference_path(
    x: np.ndarray, y: np.ndarray, stepsize: float = 1.0
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Build a dense, closed reference path from X/Y coordinates using tph
    cubic splines.

    Returns
    -------
    path_cl : np.ndarray, shape (N+1, 3)  — [s, x, y] (closed)
    path_interp : np.ndarray, shape (M, 2) — dense (x, y)
    s_tot : float — total track length in metres
    """
    # Close the path by appending the first point
    path_xy = np.column_stack([x, y])
    if not np.allclose(path_xy[0], path_xy[-1], atol=1.0):
        path_xy = np.vstack([path_xy, path_xy[0]])

    # Fit cubic splines (closed path detected automatically)
    coeffs_x, coeffs_y, _, _ = tph.calc_splines.calc_splines(path=path_xy)

    # Compute spline segment lengths
    spline_lengths = tph.calc_spline_lengths.calc_spline_lengths(
        coeffs_x=coeffs_x, coeffs_y=coeffs_y
    )
    s_tot = float(np.sum(spline_lengths))

    # Interpolate at ~1 m resolution
    path_interp, _, _, dists_interp = tph.interp_splines.interp_splines(
        coeffs_x=coeffs_x,
        coeffs_y=coeffs_y,
        spline_lengths=spline_lengths,
        incl_last_point=False,
        stepsize_approx=stepsize,
    )

    # Build path_cl: [s, x, y] — closed (last row copies first with s=s_tot)
    s_vals = dists_interp
    path_cl = np.column_stack([s_vals, path_interp])
    # Close it by appending a row with s_tot and the first XY
    closing_row = np.array([[s_tot, path_interp[0, 0], path_interp[0, 1]]])
    path_cl = np.vstack([path_cl, closing_row])

    return path_cl, path_interp, s_tot


def _project_lap_onto_path(
    path_cl: np.ndarray,
    lap_x: np.ndarray,
    lap_y: np.ndarray,
) -> np.ndarray:
    """
    Project every GPS point of a lap onto the reference path.

    Returns array of s-coordinates (one per GPS point).
    """
    s_tot = path_cl[-1, 0]
    # Scale the search range to ~10% of track length (minimum 50 m)
    s_range = max(s_tot * 0.1, 50.0)

    s_coords = np.empty(len(lap_x))
    s_expected = None

    for i in range(len(lap_x)):
        ego = np.array([lap_x[i], lap_y[i]])
        s_interp, _ = tph.path_matching_global.path_matching_global(
            path_cl=path_cl,
            ego_position=ego,
            s_expected=s_expected,
            s_range=s_range,
        )
        s_coords[i] = s_interp
        s_expected = s_interp   # narrow search for next point

    return s_coords


def _resample_speed(
    s_raw: np.ndarray,
    speed_raw: np.ndarray,
    s_grid: np.ndarray,
    s_tot: float,
) -> np.ndarray:
    """
    Unwrap the s-coordinate across the start/finish boundary and
    interpolate Speed onto the uniform ``s_grid``.

    Returns
    -------
    speed_interp : np.ndarray
    """
    # --- Unwrap s so it's monotonically increasing -------------------------
    s_unwrapped = s_raw.copy()
    ds = np.diff(s_unwrapped)
    for i in np.where(ds < -s_tot * 0.3)[0]:
        s_unwrapped[i + 1:] += s_tot

    # Sort by unwrapped s
    order = np.argsort(s_unwrapped)
    s_sorted = s_unwrapped[order]
    speed_sorted = speed_raw[order]

    # Remove any duplicate s values
    unique_mask = np.concatenate([[True], np.diff(s_sorted) > 1e-6])
    s_sorted = s_sorted[unique_mask]
    speed_sorted = speed_sorted[unique_mask]

    if len(s_sorted) < 2:
        return np.full_like(s_grid, np.nan)

    # --- Map s_grid into the unwrapped domain (monotonically) ---------------
    s0 = s_sorted[0] % s_tot
    dist_from_start = (s_grid - s0) % s_tot
    s_query = dist_from_start + s_sorted[0]

    # Sort for interpolation, then unsort
    sort_idx = np.argsort(dist_from_start)
    s_query_sorted = s_query[sort_idx]

    f_speed = interp1d(s_sorted, speed_sorted, kind='linear',
                       bounds_error=False,
                       fill_value=(speed_sorted[0], speed_sorted[-1]))

    speed_sorted_result = f_speed(s_query_sorted)

    # Unsort back to s_grid order
    speed_result = np.empty_like(s_grid)
    speed_result[sort_idx] = speed_sorted_result

    return speed_result


def _compute_delta_time(
    v_ref: np.ndarray,
    v_cmp: np.ndarray,
    ds: float = 1.0,
) -> np.ndarray:
    """
    Compute the cumulative time delta using the standard F1 method:

        Δt(s) = ∫₀ˢ (1/v_cmp - 1/v_ref) ds'

    Positive Δt means the comparison lap is *slower* (losing time).

    Parameters
    ----------
    v_ref, v_cmp : np.ndarray  — speeds on the uniform grid
    ds : float — grid spacing in metres

    Returns
    -------
    delta_t : np.ndarray  — cumulative time delta at each grid point
    """
    # Slowness difference (s/m) at each grid point
    # Guard against zero speed
    v_ref_safe = np.maximum(v_ref, 0.1)
    v_cmp_safe = np.maximum(v_cmp, 0.1)

    slowness_diff = (1.0 / v_cmp_safe) - (1.0 / v_ref_safe)

    # Cumulative integral via trapezoidal-like approach (cumsum × ds)
    delta_t = np.cumsum(slowness_diff) * ds

    return delta_t


# ═══════════════════════════════════════════════════════════════════════════
# Visualisation
# ═══════════════════════════════════════════════════════════════════════════

def plot_delta_time(
    s_grid: np.ndarray,
    v_ref: np.ndarray,
    v_lap: np.ndarray,
    delta_t: np.ndarray,
    ref_lap_num: int,
    cmp_lap_num: int,
    ref_lap_time: float,
    cmp_lap_time: float,
    save: Optional[str] = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    Dual-panel Delta-Time plot.

    Top: Speed vs Distance (reference blue, comparison red).
    Bottom: Cumulative Time Delta (green = gaining, red = losing).
    """
    fig, (ax_speed, ax_delta) = plt.subplots(
        2, 1, figsize=(14, 8), sharex=True,
        gridspec_kw={'height_ratios': [2, 1.2], 'hspace': 0.08},
        facecolor='#1a1a2e',
    )

    s_km = s_grid  # keep in metres; label in m

    # --- Top panel: Speed vs Distance -------------------------------------
    ax_speed.set_facecolor('#1a1a2e')
    ax_speed.plot(
        s_km, v_ref, color='#4aa3ff', linewidth=1.2, alpha=0.9,
        label=f'Lap {ref_lap_num}  (REF)  {ref_lap_time:.1f}s',
    )
    ax_speed.plot(
        s_km, v_lap, color='#ff5555', linewidth=1.2, alpha=0.9,
        label=f'Lap {cmp_lap_num}  {cmp_lap_time:.1f}s',
    )
    # Subtle fill between curves
    ax_speed.fill_between(
        s_km, v_ref, v_lap, alpha=0.10,
        where=v_ref >= v_lap, facecolor='#4aa3ff',
    )
    ax_speed.fill_between(
        s_km, v_ref, v_lap, alpha=0.10,
        where=v_ref < v_lap, facecolor='#ff5555',
    )
    ax_speed.set_ylabel('Speed  (m/s)', fontsize=12, color='white')
    ax_speed.legend(
        loc='upper right', fontsize=10, framealpha=0.6,
        facecolor='#2d2d44', labelcolor='white',
    )
    ax_speed.tick_params(colors='white')
    ax_speed.grid(True, color='#333355', linewidth=0.4, alpha=0.5)
    ax_speed.set_title(
        'Delta Time Analysis — Fastest Lap vs. Comparison',
        fontsize=15, fontweight='bold', color='white', pad=12,
    )

    # --- Bottom panel: Time Delta -----------------------------------------
    ax_delta.set_facecolor('#1a1a2e')
    ax_delta.axhline(0, color='white', linewidth=0.5, alpha=0.4)

    # Green where delta_t < 0 (gaining), red where > 0 (losing)
    ax_delta.fill_between(
        s_km, delta_t, 0,
        where=delta_t <= 0,
        facecolor='#2ecc71', alpha=0.65,
        label='Gaining time',
    )
    ax_delta.fill_between(
        s_km, delta_t, 0,
        where=delta_t > 0,
        facecolor='#e74c3c', alpha=0.65,
        label='Losing time',
    )
    ax_delta.plot(s_km, delta_t, color='white', linewidth=0.8, alpha=0.7)

    ax_delta.set_xlabel('Distance  (m)', fontsize=12, color='white')
    ax_delta.set_ylabel('Δt  (s)', fontsize=12, color='white')
    ax_delta.tick_params(colors='white')
    ax_delta.grid(True, color='#333355', linewidth=0.4, alpha=0.5)
    ax_delta.legend(
        loc='upper right', fontsize=10, framealpha=0.6,
        facecolor='#2d2d44', labelcolor='white',
    )

    # Annotate final delta
    final_dt = delta_t[-1]
    sign = '+' if final_dt >= 0 else ''
    ax_delta.annotate(
        f'{sign}{final_dt:.2f} s',
        xy=(s_km[-1], final_dt), xycoords='data',
        fontsize=12, fontweight='bold',
        color='#e74c3c' if final_dt > 0 else '#2ecc71',
        ha='right', va='bottom' if final_dt < 0 else 'top',
        xytext=(-10, -10 if final_dt >= 0 else 10),
        textcoords='offset points',
    )

    fig.tight_layout()

    if save:
        fig.savefig(save, dpi=dpi, facecolor=fig.get_facecolor(),
                    bbox_inches='tight')
        print(f'Figure saved → {save}')

    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════════════

def run_delta_analysis(
    csv_path: str,
    compare_lap: Optional[int] = None,
    save: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    End-to-end Delta Time analysis.

    1. Load telemetry, detect laps, identify fastest.
    2. Build reference path from fastest lap via tph splines.
    3. Project fastest + comparison lap onto reference path.
    4. Resample onto 1-m grid.
    5. Compute speed delta & cumulative time delta.
    6. Plot.
    """

    # ---- 1. Load telemetry ------------------------------------------------
    print(f'Loading telemetry: {csv_path}')
    df = load_telemetry(csv_path)

    # We also need raw esptime for time calculations
    df_raw = pd.read_csv(csv_path)
    # After load_telemetry filters zero-GPS rows, align esptime
    mask = (df_raw['latitude'] != 0.0) | (df_raw['longitude'] != 0.0)
    df['EspTime'] = df_raw.loc[mask, 'esptime'].values

    # ---- Identify laps and lap times --------------------------------------
    all_lap_nums = sorted(df['LapNumber'].unique())
    print(f'Detected {len(all_lap_nums)} raw laps: {all_lap_nums}')

    # Filter out incomplete laps (fragments at start/end of data)
    lap_counts = df.groupby('LapNumber').size()
    max_pts = lap_counts.max()
    # Keep only laps with at least 50% of the longest lap's points
    full_laps = lap_counts[lap_counts >= max_pts * 0.5].index.tolist()
    lap_nums = sorted(full_laps)
    if len(all_lap_nums) != len(lap_nums):
        dropped = set(all_lap_nums) - set(lap_nums)
        print(f'  Dropped incomplete laps: {sorted(dropped)}')
    print(f'Using {len(lap_nums)} complete laps: {lap_nums}')

    if len(lap_nums) < 2:
        raise ValueError('Need at least 2 complete laps for delta-time analysis.')

    lap_times = {}
    for ln in lap_nums:
        lap_df = df[df['LapNumber'] == ln]
        t_arr = lap_df['EspTime'].values
        lap_times[ln] = t_arr[-1] - t_arr[0]
        print(f'  Lap {ln}: {lap_times[ln]:.1f} s  '
              f'({len(lap_df)} pts, '
              f'avg speed {lap_df["Speed"].mean():.1f} m/s)')

    # Fastest lap = reference
    ref_lap = min(lap_times, key=lap_times.get)
    print(f'\n→ Reference lap (fastest): Lap {ref_lap}  '
          f'({lap_times[ref_lap]:.1f} s)')

    # Comparison lap: user-specified or median-time
    if compare_lap is not None:
        if compare_lap not in lap_nums:
            raise ValueError(f'Lap {compare_lap} not found. '
                             f'Available: {lap_nums}')
        cmp_lap = compare_lap
    else:
        # Choose the lap closest to median time (excluding ref)
        other_laps = [ln for ln in lap_nums if ln != ref_lap]
        median_t = np.median([lap_times[ln] for ln in other_laps])
        cmp_lap = min(other_laps,
                      key=lambda ln: abs(lap_times[ln] - median_t))
    print(f'→ Comparison lap:          Lap {cmp_lap}  '
          f'({lap_times[cmp_lap]:.1f} s)\n')

    # ---- 2. Build reference path from fastest lap --------------------------
    ref_df = df[df['LapNumber'] == ref_lap].reset_index(drop=True)
    print('Building reference path via tph splines ...')
    path_cl, path_interp, s_tot = _build_reference_path(
        ref_df['X'].values, ref_df['Y'].values, stepsize=1.0,
    )
    print(f'  Track length: {s_tot:.1f} m   '
          f'({len(path_interp)} dense points)')

    # ---- 3. Project both laps onto reference path -------------------------
    cmp_df = df[df['LapNumber'] == cmp_lap].reset_index(drop=True)

    print(f'Projecting Lap {ref_lap} (ref) onto path ...')
    s_ref = _project_lap_onto_path(
        path_cl, ref_df['X'].values, ref_df['Y'].values,
    )

    print(f'Projecting Lap {cmp_lap} (cmp) onto path ...')
    s_cmp = _project_lap_onto_path(
        path_cl, cmp_df['X'].values, cmp_df['Y'].values,
    )

    # ---- 4. Resample onto 1 m grid ----------------------------------------
    s_grid = np.arange(0, s_tot, 1.0)
    print(f'Resampling onto {len(s_grid)}-point grid (1 m steps) ...')

    v_ref = _resample_speed(s_ref, ref_df['Speed'].values, s_grid, s_tot)
    v_cmp = _resample_speed(s_cmp, cmp_df['Speed'].values, s_grid, s_tot)

    # ---- 5. Calculate deltas -----------------------------------------------
    speed_delta = v_ref - v_cmp          # positive = ref faster
    time_delta = _compute_delta_time(v_ref, v_cmp, ds=1.0)  # positive = cmp slower

    print(f'\n=== Results ===')
    print(f'Speed Δ range: [{speed_delta.min():+.2f}, '
          f'{speed_delta.max():+.2f}] m/s')
    print(f'Time  Δ final: {time_delta[-1]:+.2f} s')

    # ---- 6. Visualise -----------------------------------------------------
    fig = plot_delta_time(
        s_grid=s_grid,
        v_ref=v_ref,
        v_lap=v_cmp,
        delta_t=time_delta,
        ref_lap_num=ref_lap,
        cmp_lap_num=cmp_lap,
        ref_lap_time=lap_times[ref_lap],
        cmp_lap_time=lap_times[cmp_lap],
        save=save,
    )

    if show:
        plt.show()

    return fig


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Delta Time analysis — fastest lap vs. comparison.',
    )
    parser.add_argument(
        '--csv', type=str, default=None,
        help='Path to telemetry CSV. Default: telemetry-main/Data/synthetic_circuit_1.csv',
    )
    parser.add_argument(
        '--compare-lap', type=int, default=None,
        help='Lap number to compare (default: median-time lap).',
    )
    parser.add_argument(
        '--save', type=str, default=None,
        help='Save the figure to this path (e.g. delta_time.png).',
    )
    parser.add_argument(
        '--no-show', action='store_true',
        help='Suppress interactive plot window.',
    )
    args = parser.parse_args()

    csv = args.csv
    if csv is None:
        csv = os.path.join(_SCRIPT_DIR, 'telemetry-main', 'Data',
                           'synthetic_circuit_1.csv')

    run_delta_analysis(
        csv_path=csv,
        compare_lap=args.compare_lap,
        save=args.save,
        show=not args.no_show,
    )
