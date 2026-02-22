"""
main.py — FastAPI Backend for the Electrathon Telemetry Platform
================================================================
Endpoints
---------
POST /api/telemetry/ingest      Receive binary LoRa packets
GET  /api/drivers               List all drivers
POST /api/drivers               Create a driver
GET  /api/drivers/{id}/laps     Lap history for a driver
GET  /api/laps/{id}/telemetry   Telemetry data for a specific lap
GET  /api/laps/latest           Most recent completed lap (any driver)

Run
---
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import struct
import time
from datetime import datetime
from typing import Dict, List, Optional

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session as DBSession

import config
from auto_lap import AutoLapDetector
from models import (
    Base, Driver, Lap, Session as RaceSession, TelemetryData, TrackTemplate,
    engine, get_db, init_db,
)
from processing import process_completed_lap, store_track_template


# ═══════════════════════════════════════════════════════════════════════════════
# APP SETUP
# ═══════════════════════════════════════════════════════════════════════════════
app = FastAPI(
    title="Electrathon Telemetry API",
    description="Automated telemetry ingestion and analysis for IEEE Electrathon",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    init_db()
    print("🏁 Electrathon Telemetry API is live")


# ═══════════════════════════════════════════════════════════════════════════════
# IN-MEMORY STATE (per-session lap detectors)
# ═══════════════════════════════════════════════════════════════════════════════
# Key: driver_id (int), Value: (AutoLapDetector, session_db_id)
_active_sessions: Dict[int, tuple[AutoLapDetector, int]] = {}


def _get_or_create_session(
    driver_id: int, db: DBSession
) -> tuple[AutoLapDetector, int]:
    """Get an existing session/detector for this driver, or create a new one."""
    if driver_id in _active_sessions:
        return _active_sessions[driver_id]

    # Find or create the driver record
    driver = db.query(Driver).filter(Driver.driver_id == driver_id).first()
    if not driver:
        driver = Driver(driver_id=driver_id, name=f"Driver {driver_id}")
        db.add(driver)
        db.flush()

    # Create a new session
    session = RaceSession(driver_id=driver.id)
    db.add(session)
    db.flush()

    detector = AutoLapDetector()
    _active_sessions[driver_id] = (detector, session.id)
    db.commit()
    return detector, session.id


# ═══════════════════════════════════════════════════════════════════════════════
# PYDANTIC SCHEMAS
# ═══════════════════════════════════════════════════════════════════════════════
class DriverCreate(BaseModel):
    driver_id: int
    name: str
    car_number: Optional[int] = None
    weight_kg: Optional[float] = None

class DriverOut(BaseModel):
    id: int
    driver_id: int
    name: str
    car_number: Optional[int]
    weight_kg: Optional[float]
    class Config:
        from_attributes = True

class LapOut(BaseModel):
    id: int
    session_id: int
    lap_number: int
    lap_time_s: Optional[float]
    coast_efficiency: Optional[float]
    throttle_smoothness: Optional[float]
    trail_braking_pct: Optional[float]
    started_at: Optional[datetime]
    finished_at: Optional[datetime]
    class Config:
        from_attributes = True

class TelemetryOut(BaseModel):
    timestamp: datetime
    lat: float
    lon: float
    x_m: Optional[float]
    y_m: Optional[float]
    distance_m: Optional[float]
    speed_mps: Optional[float]
    battery_v: Optional[float]
    class Config:
        from_attributes = True


# ═══════════════════════════════════════════════════════════════════════════════
# BINARY PACKET INGESTION
# ═══════════════════════════════════════════════════════════════════════════════
@app.post("/api/telemetry/ingest")
async def ingest_telemetry(
    request: Request,
    background_tasks: BackgroundTasks,
    db: DBSession = Depends(get_db),
):
    """
    Receive an 18-byte binary LoRa packet and process it.

    Packet layout (15 bytes):
        Byte 0:       DriverID (hi 4 bits) | PacketID (lo 4 bits)
        Bytes 1-4:    Timestamp (uint32 BE)
        Bytes 5-8:    Latitude  (int32 BE, degrees × 1e7)
        Bytes 9-12:   Longitude (int32 BE, degrees × 1e7)
        Byte 13:      Speed (uint8, km/h)
        Byte 14:      Battery (uint8, 0-255 → 3.0-4.2V)
    """
    body = await request.body()

    if len(body) < config.PACKET_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Packet too short: got {len(body)} bytes, need {config.PACKET_SIZE}",
        )

    # ── Unpack binary ────────────────────────────────────────────────────
    header = body[0]
    driver_id = (header >> 4) & 0x0F
    packet_id = header & 0x0F

    timestamp_raw, lat_raw, lon_raw = struct.unpack(">Iii", body[1:13])
    speed_kmh = body[13]
    batt_raw = body[14]

    lat = lat_raw / 1e7
    lon = lon_raw / 1e7
    speed_mps = speed_kmh / 3.6
    battery_v = config.BATT_V_MIN + (batt_raw / 255.0) * (config.BATT_V_MAX - config.BATT_V_MIN)
    timestamp_s = float(timestamp_raw)

    # ── Auto-lap detection ───────────────────────────────────────────────
    detector, session_id = _get_or_create_session(driver_id, db)
    result = detector.feed(lat, lon, timestamp_s, speed_mps, battery_v)

    response = {
        "status": "ok",
        "driver_id": driver_id,
        "packet_id": packet_id,
        "lat": lat,
        "lon": lon,
        "speed_mps": round(speed_mps, 2),
        "battery_v": round(battery_v, 2),
    }

    if result is not None:
        lap_number, lap_buffer = result

        # Create lap record
        lap = Lap(session_id=session_id, lap_number=lap_number)
        db.add(lap)
        db.commit()
        db.refresh(lap)

        # Store track template on first lap
        if lap_number == 1 and detector.learned_track:
            t = detector.learned_track
            background_tasks.add_task(
                store_track_template,
                session_id, t.sf_lat, t.sf_lon,
                t.centroid_lat, t.centroid_lon,
                t.track_length_m, t.trace,
            )

        # Process lap in background
        background_tasks.add_task(process_completed_lap, lap.id, lap_buffer)

        response["event"] = "lap_completed"
        response["lap_number"] = lap_number
        response["lap_id"] = lap.id
    else:
        response["event"] = "point_recorded"
        response["current_lap"] = detector.current_lap_number

    return response


# ═══════════════════════════════════════════════════════════════════════════════
# REST ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════
@app.get("/api/drivers", response_model=List[DriverOut])
def list_drivers(db: DBSession = Depends(get_db)):
    """List all registered drivers."""
    return db.query(Driver).all()


@app.post("/api/drivers", response_model=DriverOut)
def create_driver(data: DriverCreate, db: DBSession = Depends(get_db)):
    """Register a new driver."""
    existing = db.query(Driver).filter(Driver.driver_id == data.driver_id).first()
    if existing:
        raise HTTPException(400, f"Driver ID {data.driver_id} already exists")
    driver = Driver(**data.model_dump())
    db.add(driver)
    db.commit()
    db.refresh(driver)
    return driver


@app.get("/api/drivers/{driver_db_id}/laps", response_model=List[LapOut])
def get_driver_laps(driver_db_id: int, db: DBSession = Depends(get_db)):
    """Get all laps for a driver, newest first."""
    return (
        db.query(Lap)
        .join(RaceSession)
        .filter(RaceSession.driver_id == driver_db_id)
        .order_by(Lap.finished_at.desc())
        .all()
    )


@app.get("/api/laps/{lap_id}/telemetry", response_model=List[TelemetryOut])
def get_lap_telemetry(lap_id: int, db: DBSession = Depends(get_db)):
    """Get all telemetry data points for a specific lap."""
    rows = (
        db.query(TelemetryData)
        .filter(TelemetryData.lap_id == lap_id)
        .order_by(TelemetryData.timestamp)
        .all()
    )
    if not rows:
        raise HTTPException(404, f"No telemetry for lap {lap_id}")
    return rows


@app.get("/api/laps/latest", response_model=LapOut)
def get_latest_lap(db: DBSession = Depends(get_db)):
    """Get the most recently completed lap across all drivers."""
    lap = (
        db.query(Lap)
        .filter(Lap.lap_time_s.isnot(None))
        .order_by(Lap.finished_at.desc())
        .first()
    )
    if not lap:
        raise HTTPException(404, "No completed laps yet")
    return lap


@app.post("/api/sessions/{driver_id}/reset")
def reset_session(driver_id: int):
    """Reset the active session for a driver (start fresh auto-lap detection)."""
    if driver_id in _active_sessions:
        del _active_sessions[driver_id]
    return {"status": "ok", "message": f"Session reset for driver {driver_id}"}


# ═══════════════════════════════════════════════════════════════════════════════
# HEALTH CHECK
# ═══════════════════════════════════════════════════════════════════════════════
@app.get("/health")
def health_check():
    return {"status": "ok", "active_sessions": len(_active_sessions)}


# ─── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=config.API_HOST, port=config.API_PORT, reload=True)
