# Walkthrough: Electrathon Telemetry Web Platform

## What Was Built

Full-stack telemetry platform in [telemetry_platform/](file:///Users/marcotamburini/Desktop/Driver's%20Profiles%20IEEE/telemetry_platform):

```mermaid
flowchart LR
    subgraph Hardware
        H[Heltec LoRa V1.1] -->|15-byte binary| GW[Gateway]
    end
    subgraph Backend
        GW -->|POST| API[FastAPI :8000]
        API --> ALD[Auto-Lap Detector]
        ALD -->|lap complete| BG[Background Tasks]
        BG --> DB[(SQLite/TimescaleDB)]
    end
    subgraph Frontend
        ST[Streamlit :8501] -->|REST| API
        ST --> P1["Driver Hub"]
        ST --> P2["Live Telemetry"]
        ST --> P3["Ghost + Coaching"]
    end
```

## Files Created (10)

| File | Purpose |
|---|---|
| [config.py](file:///Users/marcotamburini/Desktop/Driver's%20Profiles%20IEEE/telemetry_platform/config.py) | Central config with env-var overrides |
| [models.py](file:///Users/marcotamburini/Desktop/Driver's%20Profiles%20IEEE/telemetry_platform/models.py) | SQLAlchemy ORM: 5 tables (drivers, sessions, laps, telemetry_data, track_templates) |
| [auto_lap.py](file:///Users/marcotamburini/Desktop/Driver's%20Profiles%20IEEE/telemetry_platform/auto_lap.py) | 3-phase Auto-Lap/Track detection FSM |
| [processing.py](file:///Users/marcotamburini/Desktop/Driver's%20Profiles%20IEEE/telemetry_platform/processing.py) | Background tasks: coord conversion, metrics, AI coaching |
| [main.py](file:///Users/marcotamburini/Desktop/Driver's%20Profiles%20IEEE/telemetry_platform/main.py) | FastAPI backend: binary ingest + REST API |
| [app.py](file:///Users/marcotamburini/Desktop/Driver's%20Profiles%20IEEE/telemetry_platform/app.py) | Streamlit entry point |
| [1_Driver_Hub.py](file:///Users/marcotamburini/Desktop/Driver's%20Profiles%20IEEE/telemetry_platform/pages/1_Driver_Hub.py) | Driver selector, lap history, profiling metrics |
| [2_Live_Telemetry.py](file:///Users/marcotamburini/Desktop/Driver's%20Profiles%20IEEE/telemetry_platform/pages/2_Live_Telemetry.py) | Plotly track map + speed vs distance |
| [3_Ghost_Coaching.py](file:///Users/marcotamburini/Desktop/Driver's%20Profiles%20IEEE/telemetry_platform/pages/3_Ghost_Coaching.py) | Ghost comparison, delta-time, AI advice |
| [README.md](file:///Users/marcotamburini/Desktop/Driver's%20Profiles%20IEEE/telemetry_platform/README.md) | Setup instructions + test commands |

## Verification Results

| Test | Result |
|---|---|
| All 9 Python files syntax | ✅ `ast.parse` clean |
| Dependencies installed | ✅ fastapi, uvicorn, sqlalchemy, streamlit, plotly |
| `python models.py` (DB init) | ✅ `telemetry.db` created, all tables present |
| `uvicorn main:app` startup | ✅ `🏁 Electrathon Telemetry API is live` |
| Test binary packet (15 bytes) | ✅ Unpacked: lat=28.0587, lon=-82.4139, speed=8.33m/s, batt=3.6V |
| Driver auto-created | ✅ "Driver 1" in DB |
| Auto-lap FSM started | ✅ Recording first lap, session active |
| `/health` endpoint | ✅ `{"status": "ok", "active_sessions": 1}` |

## How to Use

```bash
# Terminal 1: Backend
cd telemetry_platform
source ../.venv/bin/activate
uvicorn main:app --reload --port 8000

# Terminal 2: Frontend
cd telemetry_platform
source ../.venv/bin/activate
streamlit run app.py
```

API docs at http://localhost:8000/docs, dashboard at http://localhost:8501
