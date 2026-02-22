"""
config.py — Central configuration for the Electrathon Telemetry Platform.

Override any value via environment variables.
"""

import os

# ─── Database ────────────────────────────────────────────────────────────────
# Default: SQLite (zero setup).  For production, set DATABASE_URL to a
# PostgreSQL/TimescaleDB connection string:
#   export DATABASE_URL="postgresql://user:pass@localhost:5432/electrathon"
DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "sqlite:///./telemetry.db"
)

# ─── Auto-Lap Detection ─────────────────────────────────────────────────────
CLOSURE_RADIUS_M: float = float(os.getenv("CLOSURE_RADIUS_M", "8.0"))
MIN_LAP_TIME_S: float = float(os.getenv("MIN_LAP_TIME_S", "30.0"))
MIN_LAP_POINTS: int = int(os.getenv("MIN_LAP_POINTS", "50"))
SF_REFINEMENT: bool = os.getenv("SF_REFINEMENT", "true").lower() == "true"

# ─── API ─────────────────────────────────────────────────────────────────────
API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("API_PORT", "8000"))

# ─── Paths to existing offline scripts (relative to project root) ────────────
PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GHOST_LINE_SCRIPT: str = os.path.join(PROJECT_ROOT, "visualize_ghost_line.py")
DELTA_TIME_SCRIPT: str = os.path.join(PROJECT_ROOT, "delta_time_analysis.py")
LOAD_TELEMETRY_SCRIPT: str = os.path.join(PROJECT_ROOT, "load_telemetry.py")
OPTIMIZER_OUTPUTS: str = os.path.join(
    PROJECT_ROOT, "global_racetrajectory_optimization-master", "outputs"
)

# ─── LoRa Binary Packet ─────────────────────────────────────────────────────
# Layout (15 bytes packed, as per Driving_profile_algorithms.md):
#   Byte  0:      DriverID (4 bits hi) | PacketID (4 bits lo)
#   Bytes 1-4:    Timestamp (uint32, big-endian)
#   Bytes 5-8:    Lat       (int32,  big-endian, degrees × 1e7)
#   Bytes 9-12:   Lon       (int32,  big-endian, degrees × 1e7)
#   Byte  13:     Speed     (uint8,  km/h)
#   Byte  14:     Batt      (uint8,  mapped 0-255 → 3.0-4.2V)
PACKET_SIZE: int = 15
BATT_V_MIN: float = 3.0
BATT_V_MAX: float = 4.2
