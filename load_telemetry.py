#!/usr/bin/env python3
"""
load_telemetry.py
=================
Load a telemetry CSV (e.g. ``telemetry-main/Data/log.csv``), clean it, convert
GPS coordinates to local X/Y (metres), detect lap boundaries, and return a tidy
DataFrame ready for downstream analysis with TUMFTM trajectory_planning_helpers.

Usage
-----
    # As a module
    from load_telemetry import load_telemetry
    df = load_telemetry("telemetry-main/Data/log.csv")

    # Standalone
    python3 load_telemetry.py
"""

import math
import os
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_METRES_PER_DEG_LAT = 110_540.0          # approximate m per degree latitude
_METRES_PER_DEG_LON_EQ = 111_320.0       # m per degree longitude at equator


# ---------------------------------------------------------------------------
# Helper: timestamp parsing
# ---------------------------------------------------------------------------
def _parse_timestamp(date_series: pd.Series, time_series: pd.Series) -> pd.Series:
    """
    Combine ``date_ddmmyy`` and ``time_hhmmsscs`` columns into proper
    ``datetime`` objects.

    Parameters
    ----------
    date_series : pd.Series
        Integer column in DDMMYY format (e.g. 90725 → 09/07/2025).
        Rows with value ``0`` are treated as invalid and produce ``NaT``.
    time_series : pd.Series
        Integer column in HHMMSScc format (e.g. 12200100 → 12:20:01.00).

    Returns
    -------
    pd.Series[datetime64]
    """
    # Identify rows where both date and time are zero (invalid)
    valid = (date_series != 0) | (time_series != 0)

    # Zero-pad to 6 and 10 digits respectively
    date_str = date_series.astype(int).astype(str).str.zfill(6)
    time_str = time_series.astype(int).astype(str).str.zfill(10)

    # Extract components
    day   = date_str.str[:2].astype(int)
    month = date_str.str[2:4].astype(int)
    year  = 2000 + date_str.str[4:6].astype(int)

    hour   = time_str.str[:2].astype(int)
    minute = time_str.str[2:4].astype(int)
    second = time_str.str[4:6].astype(int)
    csec   = time_str.str[6:8].astype(int)          # centiseconds

    # Clamp invalid-date components so pd.to_datetime doesn't raise
    month = month.clip(lower=1)
    day   = day.clip(lower=1)

    timestamps = pd.to_datetime(
        {
            "year": year,
            "month": month,
            "day": day,
            "hour": hour,
            "minute": minute,
            "second": second,
        }
    ) + pd.to_timedelta(csec * 10, unit="ms")        # add centiseconds

    # Mark invalid rows as NaT
    timestamps = timestamps.where(valid, pd.NaT)

    return timestamps


# ---------------------------------------------------------------------------
# Helper: lat/lon → local X/Y (equirectangular projection)
# ---------------------------------------------------------------------------
def _latlon_to_xy(
    lat: np.ndarray, lon: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert latitude / longitude arrays to local X/Y coordinates (metres)
    using the equirectangular (flat-Earth) approximation.

    The **first point** in the arrays is taken as the origin ``(0, 0)``.

    Parameters
    ----------
    lat, lon : np.ndarray
        GPS coordinates in decimal degrees.

    Returns
    -------
    x_m, y_m : np.ndarray
        Easting (X) and Northing (Y) in metres, relative to the first point.
    """
    lat_ref = lat.iloc[0] if hasattr(lat, "iloc") else lat[0]
    lon_ref = lon.iloc[0] if hasattr(lon, "iloc") else lon[0]
    lat_ref_rad = math.radians(lat_ref)

    x_m = (lon - lon_ref) * math.cos(lat_ref_rad) * _METRES_PER_DEG_LON_EQ
    y_m = (lat - lat_ref) * _METRES_PER_DEG_LAT

    return np.asarray(x_m, dtype=float), np.asarray(y_m, dtype=float)


# ---------------------------------------------------------------------------
# Helper: lap detection at start/finish line
# ---------------------------------------------------------------------------
def _detect_laps(
    x: np.ndarray,
    y: np.ndarray,
    sf_x: Optional[float] = None,
    sf_y: Optional[float] = None,
    threshold_m: float = 5.0,
    min_lap_points: int = 200,
) -> np.ndarray:
    """
    Detect laps by checking when the car passes within *threshold_m* metres
    of an explicit Start/Finish point ``(sf_x, sf_y)``.

    If ``sf_x`` / ``sf_y`` are not given the first data point is used by
    default.

    A new lap is only triggered after at least *min_lap_points* samples since
    the last lap start, to avoid false triggering while the car is near the
    start/finish area.

    Parameters
    ----------
    x, y : np.ndarray
        Local coordinates in metres.
    sf_x, sf_y : float or None
        Start/Finish point in the local X/Y frame.  ``None`` → first point.
    threshold_m : float
        Radius (metres) around the Start/Finish gate.  Default 5 m.
    min_lap_points : int
        Minimum number of samples between two consecutive lap starts.

    Returns
    -------
    lap_numbers : np.ndarray[int]
        Lap number for each sample (0-indexed).
    """
    n = len(x)
    lap_numbers = np.zeros(n, dtype=int)
    current_lap = 0
    last_lap_start = 0

    # Default S/F gate = first recorded point
    x0 = sf_x if sf_x is not None else x[0]
    y0 = sf_y if sf_y is not None else y[0]

    for i in range(1, n):
        dist = math.sqrt((x[i] - x0) ** 2 + (y[i] - y0) ** 2)
        if dist < threshold_m and (i - last_lap_start) >= min_lap_points:
            current_lap += 1
            last_lap_start = i
        lap_numbers[i] = current_lap

    return lap_numbers


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------
def load_telemetry(
    csv_path: str,
    sf_x: Optional[float] = None,
    sf_y: Optional[float] = None,
    sf_radius: float = 5.0,
) -> pd.DataFrame:
    """
    Load and clean a telemetry CSV, returning a DataFrame with columns::

        Timestamp  X  Y  Speed  LapNumber

    Steps
    -----
    1. Read CSV with Pandas.
    2. Parse ``date_ddmmyy`` + ``time_hhmmsscs`` → ``Timestamp`` (datetime).
    3. Filter rows where latitude **or** longitude is exactly ``0.0``.
    4. Convert lat/lon → local X/Y (metres) via equirectangular projection
       (origin = first valid GPS point).
    5. Detect lap boundaries and assign ``LapNumber``.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file (e.g. ``telemetry-main/Data/log.csv``).
    sf_x, sf_y : float or None
        Start/Finish gate in local X/Y metres.  ``None`` → first GPS point.
    sf_radius : float
        Radius (metres) around the S/F gate.  Default 5 m.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with columns ``Timestamp, X, Y, Speed, LapNumber``.
    """
    # ---- 1. Load ----------------------------------------------------------
    df = pd.read_csv(csv_path)

    # ---- 2. Parse timestamps ----------------------------------------------
    df["Timestamp"] = _parse_timestamp(df["date_ddmmyy"], df["time_hhmmsscs"])

    # ---- 3. Filter zero GPS -----------------------------------------------
    mask = (df["latitude"] != 0.0) | (df["longitude"] != 0.0)
    df = df.loc[mask].reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid GPS data found after filtering zero rows.")

    # ---- 4. GPS → X/Y (metres) -------------------------------------------
    x_m, y_m = _latlon_to_xy(df["latitude"], df["longitude"])
    df["X"] = x_m
    df["Y"] = y_m

    # ---- 5. Detect laps ---------------------------------------------------
    df["LapNumber"] = _detect_laps(x_m, y_m, sf_x=sf_x, sf_y=sf_y,
                                   threshold_m=sf_radius)

    # ---- 6. Return clean DataFrame ----------------------------------------
    return df[["Timestamp", "X", "Y", "speed", "LapNumber"]].rename(
        columns={"speed": "Speed"}
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _default_csv = os.path.join(
        _script_dir, "telemetry-main", "Data", "log.csv"
    )

    print(f"Loading telemetry from: {_default_csv}\n")
    result = load_telemetry(_default_csv)

    print("=== DataFrame Info ===")
    print(f"Shape : {result.shape}")
    print(f"Dtypes:\n{result.dtypes}\n")

    print("=== First 10 rows ===")
    print(result.head(10).to_string(index=False))

    print(f"\n=== Lap summary ===")
    print(result.groupby("LapNumber").agg(
        count=("X", "size"),
        mean_speed=("Speed", "mean"),
    ).to_string())
