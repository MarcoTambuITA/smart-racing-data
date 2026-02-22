"""
auto_lap.py — Auto-Lap & Auto-Track Detection Engine
=====================================================
Finite-state machine that detects lap completions and learns the track
shape from incoming GPS data.  No hardcoded S/F coordinates needed.

States
------
WAITING         No GPS data yet
RECORDING       Accumulating first lap buffer
TRACKING        Steady-state: detecting subsequent laps

Usage
-----
    from auto_lap import AutoLapDetector

    detector = AutoLapDetector()                    # per-session instance
    result = detector.feed(lat, lon, timestamp_s)   # call for each GPS fix
    if result is not None:
        lap_number, lap_buffer = result             # lap just completed!
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple

from config import CLOSURE_RADIUS_M, MIN_LAP_TIME_S, MIN_LAP_POINTS, SF_REFINEMENT


# ═══════════════════════════════════════════════════════════════════════════════
# Haversine
# ═══════════════════════════════════════════════════════════════════════════════
_R_EARTH = 6_371_000.0  # metres


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres between two GPS points."""
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    Δφ = math.radians(lat2 - lat1)
    Δλ = math.radians(lon2 - lon1)
    a = math.sin(Δφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(Δλ / 2) ** 2
    return _R_EARTH * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ═══════════════════════════════════════════════════════════════════════════════
# GPS point data class
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class GPSPoint:
    lat: float
    lon: float
    timestamp_s: float          # epoch seconds (or monotonic counter)
    speed_mps: float = 0.0
    battery_v: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# FSM states
# ═══════════════════════════════════════════════════════════════════════════════
class _State(Enum):
    WAITING   = auto()
    RECORDING = auto()
    TRACKING  = auto()


# ═══════════════════════════════════════════════════════════════════════════════
# Track template (learned from first lap)
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class LearnedTrack:
    sf_lat: float
    sf_lon: float
    centroid_lat: float
    centroid_lon: float
    track_length_m: float
    trace: List[GPSPoint]


# ═══════════════════════════════════════════════════════════════════════════════
# Auto-Lap Detector
# ═══════════════════════════════════════════════════════════════════════════════
class AutoLapDetector:
    """
    Per-session lap detector.  Call ``feed()`` for each incoming GPS fix.

    Returns ``None`` while accumulating, or ``(lap_number, buffer)`` when
    a lap is completed.
    """

    def __init__(
        self,
        closure_radius_m: float = CLOSURE_RADIUS_M,
        min_lap_time_s: float = MIN_LAP_TIME_S,
        min_lap_points: int = MIN_LAP_POINTS,
        refine_sf: bool = SF_REFINEMENT,
    ):
        self._radius = closure_radius_m
        self._min_time = min_lap_time_s
        self._min_pts = min_lap_points
        self._refine = refine_sf

        self._state: _State = _State.WAITING
        self._sf_lat: float = 0.0
        self._sf_lon: float = 0.0
        self._lap_start_t: float = 0.0
        self._current_buffer: List[GPSPoint] = []
        self._lap_count: int = 0
        self._track: Optional[LearnedTrack] = None

    # ── Public API ────────────────────────────────────────────────────────
    def feed(
        self,
        lat: float,
        lon: float,
        timestamp_s: float,
        speed_mps: float = 0.0,
        battery_v: float = 0.0,
    ) -> Optional[Tuple[int, List[GPSPoint]]]:
        """
        Ingest one GPS fix.

        Returns
        -------
        None              — still accumulating
        (lap_num, buffer) — a lap just completed
        """
        pt = GPSPoint(lat, lon, timestamp_s, speed_mps, battery_v)

        # ── WAITING: first ever GPS fix → bootstrap ──────────────────────
        if self._state is _State.WAITING:
            self._sf_lat = lat
            self._sf_lon = lon
            self._lap_start_t = timestamp_s
            self._current_buffer = [pt]
            self._state = _State.RECORDING
            return None

        # ── Append to current buffer ─────────────────────────────────────
        self._current_buffer.append(pt)

        # ── Check loop closure ───────────────────────────────────────────
        d = haversine(lat, lon, self._sf_lat, self._sf_lon)
        elapsed = timestamp_s - self._lap_start_t

        if (
            d < self._radius
            and elapsed >= self._min_time
            and len(self._current_buffer) >= self._min_pts
        ):
            return self._complete_lap(timestamp_s)

        return None

    @property
    def learned_track(self) -> Optional[LearnedTrack]:
        """Access the auto-learned track (available after first lap)."""
        return self._track

    @property
    def current_lap_number(self) -> int:
        return self._lap_count + 1

    # ── Private ───────────────────────────────────────────────────────────
    def _complete_lap(
        self, timestamp_s: float
    ) -> Tuple[int, List[GPSPoint]]:
        self._lap_count += 1
        completed_buffer = list(self._current_buffer)

        # Phase 2: learn track from first lap
        if self._lap_count == 1:
            self._learn_track(completed_buffer)

        # Reset for next lap
        self._lap_start_t = timestamp_s
        self._current_buffer = [self._current_buffer[-1]]  # carry-over last point

        return (self._lap_count, completed_buffer)

    def _learn_track(self, buffer: List[GPSPoint]) -> None:
        """Phase 2: compute centroid, track length, and optionally refine S/F."""
        lats = [p.lat for p in buffer]
        lons = [p.lon for p in buffer]

        centroid_lat = sum(lats) / len(lats)
        centroid_lon = sum(lons) / len(lons)

        # Total track length
        total_len = 0.0
        for i in range(1, len(buffer)):
            total_len += haversine(
                buffer[i - 1].lat, buffer[i - 1].lon,
                buffer[i].lat, buffer[i].lon,
            )

        # Refine S/F: pick the lowest-curvature point near the original S/F
        if self._refine and len(buffer) > 10:
            sf_lat, sf_lon = self._refine_sf(buffer)
            self._sf_lat = sf_lat
            self._sf_lon = sf_lon

        self._track = LearnedTrack(
            sf_lat=self._sf_lat,
            sf_lon=self._sf_lon,
            centroid_lat=centroid_lat,
            centroid_lon=centroid_lon,
            track_length_m=total_len,
            trace=buffer,
        )
        self._state = _State.TRACKING

    @staticmethod
    def _refine_sf(buffer: List[GPSPoint]) -> Tuple[float, float]:
        """
        Find the point in the first 15% / last 15% of the buffer with the
        lowest curvature (straightest segment). Gives more repeatable S/F
        crossing detection than the raw first GPS fix.
        """
        n = len(buffer)
        window = max(5, n // 7)  # search in first/last ~15%
        candidates = list(range(window)) + list(range(n - window, n))

        best_idx = candidates[0]
        best_curv = float("inf")

        for i in candidates:
            if i < 2 or i >= n - 2:
                continue
            # Simple discrete curvature: angle change between 3 points
            dx1 = buffer[i].lon - buffer[i - 2].lon
            dy1 = buffer[i].lat - buffer[i - 2].lat
            dx2 = buffer[i + 2].lon - buffer[i].lon
            dy2 = buffer[i + 2].lat - buffer[i].lat
            cross = abs(dx1 * dy2 - dy1 * dx2)
            dot = dx1 * dx2 + dy1 * dy2
            curv = cross / (dot + 1e-12)
            if curv < best_curv:
                best_curv = curv
                best_idx = i

        return buffer[best_idx].lat, buffer[best_idx].lon
