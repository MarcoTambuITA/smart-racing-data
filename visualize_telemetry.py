#!/usr/bin/env python3
"""
visualize_telemetry.py
======================
Visualize cleaned telemetry as a speed-colored racing-line plot.

Colour scheme
-------------
* **Red**    → aggressive braking (deceleration)
* **Yellow** → constant speed
* **Green**  → acceleration / top speed

Usage
-----
    # As a module
    from visualize_telemetry import plot_racing_line
    plot_racing_line(df)                     # df from load_telemetry()
    plot_racing_line(df, lap=2)              # single lap
    plot_racing_line(df, save="track.png")   # save to file

    # Standalone (uses telemetry-main/Data/log.csv)
    python3 visualize_telemetry.py
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core visualisation
# ---------------------------------------------------------------------------
def plot_racing_line(
    df: pd.DataFrame,
    lap: Optional[int] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 9),
    linewidth: float = 2.5,
    save: Optional[str] = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    Plot the X/Y trajectory coloured by speed.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns ``X``, ``Y``, ``Speed``, ``LapNumber``.
    lap : int or None
        If given, plot only that lap number.  ``None`` → all laps.
    title : str or None
        Custom plot title.  ``None`` → auto-generated.
    figsize : tuple
        Figure size in inches.
    linewidth : float
        Track line width.
    save : str or None
        If given, save figure to this path (e.g. ``"track.png"``).
    dpi : int
        Resolution for saved figures.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # ---- Subset to requested lap(s) --------------------------------------
    data = df.copy()
    if lap is not None:
        data = data[data["LapNumber"] == lap].reset_index(drop=True)
        if data.empty:
            raise ValueError(f"Lap {lap} not found in DataFrame.")

    x = data["X"].values
    y = data["Y"].values
    speed = data["Speed"].values

    # ---- Build custom colourmap: Red → Yellow → Green --------------------
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "braking_accel",
        [
            (0.0, "#d62728"),    # red   – braking / low speed
            (0.5, "#ffdd44"),    # yellow – constant / mid speed
            (1.0, "#2ca02c"),    # green  – acceleration / top speed
        ],
    )

    # ---- Create line segments for LineCollection -------------------------
    points = np.column_stack([x, y]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Colour by the average speed of each segment
    seg_speed = 0.5 * (speed[:-1] + speed[1:])

    norm = mcolors.Normalize(vmin=seg_speed.min(), vmax=seg_speed.max())
    lc = mcoll.LineCollection(segments, cmap=cmap, norm=norm,
                              linewidths=linewidth, capstyle="round")
    lc.set_array(seg_speed)

    # ---- Plot ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize, facecolor="#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    ax.add_collection(lc)
    ax.autoscale_view()
    ax.set_aspect("equal")

    # Mark Start/Finish
    ax.plot(x[0], y[0], marker="o", markersize=10, color="white",
            zorder=5, label="Start / Finish")
    ax.plot(x[-1], y[-1], marker="s", markersize=8, color="#ff6b6b",
            zorder=5, label="End")

    # Colourbar
    cbar = fig.colorbar(lc, ax=ax, pad=0.02, shrink=0.85)
    cbar.set_label("Speed  (m/s)", fontsize=12, color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    # Labels
    ax.set_xlabel("X  (m)", fontsize=12, color="white")
    ax.set_ylabel("Y  (m)", fontsize=12, color="white")
    ax.tick_params(colors="white")

    if title is None:
        n_laps = data["LapNumber"].nunique()
        if lap is not None:
            title = f"Racing Line — Lap {lap}"
        elif n_laps > 1:
            title = f"Racing Line — {n_laps} Laps"
        else:
            title = "Racing Line"
    ax.set_title(title, fontsize=16, fontweight="bold", color="white",
                 pad=14)

    ax.legend(loc="upper right", fontsize=10, framealpha=0.6,
              facecolor="#2d2d44", labelcolor="white")

    # Grid
    ax.grid(True, color="#333355", linewidth=0.4, alpha=0.5)

    fig.tight_layout()

    if save:
        fig.savefig(save, dpi=dpi, facecolor=fig.get_facecolor(),
                    bbox_inches="tight")
        print(f"Figure saved → {save}")

    return fig


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from load_telemetry import load_telemetry

    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _csv = os.path.join(_script_dir, "telemetry-main", "Data", "log.csv")

    print(f"Loading  : {_csv}")
    df = load_telemetry(_csv)
    print(f"Rows     : {len(df)}")
    print(f"Laps     : {df['LapNumber'].nunique()}")
    print(f"Speed    : {df['Speed'].min():.2f} … {df['Speed'].max():.2f} m/s\n")

    out_path = os.path.join(_script_dir, "racing_line.png")
    plot_racing_line(df, save=out_path)
    plt.show()
