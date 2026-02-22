# 🏁 Electrathon Telemetry Platform

Automated telemetry ingestion + Streamlit dashboard for the IEEE Electrathon racing team.

## Architecture

```
telemetry_platform/
├── main.py          # FastAPI backend (binary ingest + REST API)
├── models.py        # SQLAlchemy ORM (5 tables)
├── auto_lap.py      # Auto-Lap/Track detection FSM
├── processing.py    # Background tasks (coord conversion, metrics, coaching)
├── config.py        # Configuration (env-var overrides)
├── app.py           # Streamlit entry point
├── pages/
│   ├── 1_Driver_Hub.py        # Driver selection, lap history
│   ├── 2_Live_Telemetry.py    # Track map + speed vs distance
│   └── 3_Ghost_Coaching.py    # Ghost comparison + AI coaching
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Install dependencies

```bash
cd telemetry_platform
pip install -r requirements.txt
```

### 2. Initialize database

```bash
python models.py    # creates SQLite DB (telemetry.db)
```

### 3. Start the backend

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
# or: python main.py
```

API docs available at: http://localhost:8000/docs

### 4. Start the dashboard

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

## Sending Test Data

### Register a driver

```bash
curl -X POST http://localhost:8000/api/drivers \
  -H "Content-Type: application/json" \
  -d '{"driver_id": 1, "name": "Marco", "weight_kg": 70.0}'
```

### Send a binary LoRa packet

```python
import struct, requests

# Pack: DriverID=1, PacketID=0, Timestamp, Lat*1e7, Lon*1e7, Speed(km/h), Batt
header = (1 << 4) | 0  # driver 1, packet 0
payload = struct.pack(">B I i i B B",
    header,
    1708000000,             # timestamp
    int(28.0587 * 1e7),     # lat
    int(-82.4139 * 1e7),    # lon
    30,                     # 30 km/h
    128,                    # ~3.6V
)
requests.post("http://localhost:8000/api/telemetry/ingest", data=payload)
```

## Configuration

Override defaults via environment variables:

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `sqlite:///./telemetry.db` | PostgreSQL: `postgresql://user:pass@host/db` |
| `CLOSURE_RADIUS_M` | `8.0` | Auto-lap detection radius (metres) |
| `MIN_LAP_TIME_S` | `30.0` | Min seconds between lap triggers |
| `API_PORT` | `8000` | Backend port |
