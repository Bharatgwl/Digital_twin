# CCTV-Based Digital Twin System

This project now follows the `SYS.tex` architecture and functional requirements with an end-to-end digital twin pipeline:

1. Frame capture (webcam / RTSP / video file)  
2. Human detection + tracking (YOLOv8 + persistent track IDs)  
3. Feature extraction (face embedding, clothing color, spatial position, height-in-pixels)  
4. Identity matching (embedding similarity)  
5. Digital twin profile/history storage in **MongoDB**  
6. Live monitoring dashboard + retrieval APIs

## Implemented Modules

- **Frame Capture Module**: continuous acquisition, resizing, frame buffering
- **Person Detection + Tracking Module**: near real-time multi-person handling with consistent IDs
- **Feature Extraction Module**: embeddings and visual behavior descriptors
- **Identity Matching Module**: matches returning individuals and creates new profiles for unknown people
- **Digital Twin Database Module (MongoDB)**:
  - `profiles` collection
  - `observations` collection
  - `visits` collection
- **Monitoring Dashboard**:
  - Live stream
  - Active tracks
  - Profile counters
  - Recent profiles table

## API Endpoints

- `GET /health`
- `GET /video`
- `GET /digital-twins`
- `GET /digital-twins/{person_id}`
- `GET /digital-twins/{person_id}/timeline`
- `GET /digital-twins/{person_id}/trajectory`
- `GET /digital-twins/{person_id}/visits`
- `GET /stats/live`

## Configuration

Environment variables:

- `MONGO_URI` (default: `mongodb://localhost:27017`)
- `MONGO_DB_NAME` (default: `digital_twin`)
- `VIDEO_SOURCE` (default: `0`)
- `CAMERA_ID` (default: `CAM-1`)
- `FRAME_WIDTH` (default: `1280`)
- `FRAME_HEIGHT` (default: `720`)
- `SNAPSHOT_INTERVAL` (default: `20`)

## Run

```bash
# ensure MongoDB is running locally (or set MONGO_URI)
uv pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open:

- Dashboard: `http://localhost:8000`
- API docs: `http://localhost:8000/docs`

## Notes

- This implementation covers the required initial prototype scope from `SYS.tex`: live capture, human detection/tracking, snapshot storage, and digital profile generation.
- Advanced items such as behavior prediction and multi-camera synchronization remain out of scope for the initial prototype.
