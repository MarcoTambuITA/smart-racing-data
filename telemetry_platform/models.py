"""
models.py — SQLAlchemy ORM schema for the Electrathon Telemetry Platform.

Tables
------
drivers           Team roster (6-7 members)
sessions          Group laps into practice / race sessions
laps              Per-lap aggregates + driver-profiling metrics
telemetry_data    High-frequency GPS stream (TimescaleDB hypertable candidate)
track_templates   Auto-learned track shapes (per session)

Usage
-----
    python models.py          # creates all tables in the configured DB
"""

from datetime import datetime

from sqlalchemy import (
    Column, Integer, Float, String, DateTime, Boolean,
    ForeignKey, Text, create_engine, UniqueConstraint,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

from config import DATABASE_URL

Base = declarative_base()


# ═══════════════════════════════════════════════════════════════════════════════
# DRIVERS
# ═══════════════════════════════════════════════════════════════════════════════
class Driver(Base):
    __tablename__ = "drivers"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    driver_id   = Column(Integer, unique=True, nullable=False, comment="4-bit LoRa ID (0-15)")
    name        = Column(String(100), nullable=False)
    car_number  = Column(Integer, nullable=True)
    weight_kg   = Column(Float, nullable=True)
    created_at  = Column(DateTime, default=datetime.utcnow)

    sessions = relationship("Session", back_populates="driver", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Driver {self.driver_id}: {self.name}>"


# ═══════════════════════════════════════════════════════════════════════════════
# SESSIONS
# ═══════════════════════════════════════════════════════════════════════════════
class Session(Base):
    __tablename__ = "sessions"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    driver_id   = Column(Integer, ForeignKey("drivers.id"), nullable=False)
    date        = Column(DateTime, default=datetime.utcnow)
    track_name  = Column(String(200), nullable=True, comment="Auto-detected or user-labeled")
    notes       = Column(Text, nullable=True)

    driver = relationship("Driver", back_populates="sessions")
    laps   = relationship("Lap", back_populates="session", cascade="all, delete-orphan")
    track_template = relationship("TrackTemplate", back_populates="session", uselist=False,
                                  cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Session {self.id} driver={self.driver_id} date={self.date}>"


# ═══════════════════════════════════════════════════════════════════════════════
# LAPS
# ═══════════════════════════════════════════════════════════════════════════════
class Lap(Base):
    __tablename__ = "laps"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    session_id    = Column(Integer, ForeignKey("sessions.id"), nullable=False)
    lap_number    = Column(Integer, nullable=False)
    lap_time_s    = Column(Float, nullable=True, comment="Seconds")
    is_valid      = Column(Boolean, default=True)

    # ── Driver-profiling metrics (filled by background processing) ────────
    coast_efficiency       = Column(Float, nullable=True,
        comment="Decel rate when throttle=0, brake=0 (lower = more efficient)")
    throttle_smoothness    = Column(Float, nullable=True,
        comment="Std-dev of throttle input derivative (lower = smoother)")
    trail_braking_pct      = Column(Float, nullable=True,
        comment="% of corner-entry time spent trail-braking")
    apex_speed_consistency = Column(Float, nullable=True,
        comment="Std-dev of apex speeds across same-corner repeats")

    # ── Timestamps ────────────────────────────────────────────────────────
    started_at  = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)

    session   = relationship("Session", back_populates="laps")
    telemetry = relationship("TelemetryData", back_populates="lap",
                             cascade="all, delete-orphan",
                             order_by="TelemetryData.timestamp")

    __table_args__ = (
        UniqueConstraint("session_id", "lap_number", name="uq_session_lap"),
    )

    def __repr__(self):
        return f"<Lap {self.lap_number} session={self.session_id} time={self.lap_time_s}>"


# ═══════════════════════════════════════════════════════════════════════════════
# TELEMETRY DATA (high-frequency GPS stream)
# ═══════════════════════════════════════════════════════════════════════════════
class TelemetryData(Base):
    __tablename__ = "telemetry_data"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    lap_id      = Column(Integer, ForeignKey("laps.id"), nullable=True)
    timestamp   = Column(DateTime, nullable=False, index=True)

    # ── Raw GPS ───────────────────────────────────────────────────────────
    lat         = Column(Float, nullable=False, comment="Decimal degrees")
    lon         = Column(Float, nullable=False, comment="Decimal degrees")

    # ── Derived local coordinates ─────────────────────────────────────────
    x_m         = Column(Float, nullable=True, comment="Easting (metres)")
    y_m         = Column(Float, nullable=True, comment="Northing (metres)")
    distance_m  = Column(Float, nullable=True, comment="Cumulative distance from lap start")

    # ── Sensor data ───────────────────────────────────────────────────────
    speed_mps   = Column(Float, nullable=True, comment="m/s")
    battery_v   = Column(Float, nullable=True, comment="Battery voltage")

    lap = relationship("Lap", back_populates="telemetry")

    def __repr__(self):
        return f"<Telemetry t={self.timestamp} lat={self.lat:.5f} lon={self.lon:.5f}>"


# ═══════════════════════════════════════════════════════════════════════════════
# TRACK TEMPLATES (auto-learned track shapes)
# ═══════════════════════════════════════════════════════════════════════════════
class TrackTemplate(Base):
    __tablename__ = "track_templates"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    session_id      = Column(Integer, ForeignKey("sessions.id"), nullable=False, unique=True)

    sf_lat          = Column(Float, nullable=False, comment="Refined Start/Finish latitude")
    sf_lon          = Column(Float, nullable=False, comment="Refined Start/Finish longitude")
    centroid_lat    = Column(Float, nullable=True)
    centroid_lon    = Column(Float, nullable=True)
    track_length_m  = Column(Float, nullable=True, comment="Total track perimeter in metres")
    num_points      = Column(Integer, nullable=True, comment="GPS points in first lap trace")

    # JSON-serialised arrays of lat/lon for the learned track boundary
    trace_lats_json = Column(Text, nullable=True)
    trace_lons_json = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("Session", back_populates="track_template")


# ═══════════════════════════════════════════════════════════════════════════════
# ENGINE & SESSION FACTORY
# ═══════════════════════════════════════════════════════════════════════════════
engine = create_engine(
    DATABASE_URL,
    echo=False,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """FastAPI dependency — yields a DB session, auto-closes on exit."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Create all tables (safe to call repeatedly)."""
    Base.metadata.create_all(bind=engine)
    print(f"✓ Database tables created ({DATABASE_URL})")


# ─── CLI: create tables ──────────────────────────────────────────────────────
if __name__ == "__main__":
    init_db()
