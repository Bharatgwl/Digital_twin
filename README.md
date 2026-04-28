# 🏙️ Digital Twin — CCTV & Aerial Imagery Intelligence System

A real-time **Digital Twin** platform that transforms raw CCTV and aerial camera feeds into an intelligent, queryable virtual representation of physical spaces. The system detects, tracks, recognizes, and profiles every person in the scene — building a living digital mirror of the real world.

---

## 🎯 Project Vision

Traditional CCTV systems are passive — they record video that humans must manually review. This project reimagines surveillance as an **active intelligence platform** where:

- Every person detected becomes a **digital entity** with a persistent profile
- The system **learns and remembers** identities across sessions
- A **virtual twin** of the monitored space updates in real-time
- Operators can **query** the system semantically (e.g., *"Who was near the entrance at 3 PM?"*)
- **Anomalies** are detected and flagged automatically

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    CCTV Digital Twin System                  │
├─────────────┬─────────────┬─────────────┬───────────────────┤
│  Level 1    │  Level 2    │  Level 3    │  Level 4          │
│  Detection  │  Tracking   │ Recognition │  Profiling        │
│  (YOLOv8)   │ (BoT-SORT)  │ (DeepFace)  │  (Gemini + DB)   │
├─────────────┴─────────────┴─────────────┴───────────────────┤
│                     FastAPI Backend                          │
├──────────────────┬──────────────────────────────────────────┤
│  SQLite Database │  ChromaDB (Vector Store) [Planned]       │
├──────────────────┴──────────────────────────────────────────┤
│              Live Dashboard & Virtual Twin UI [Planned]      │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔬 Multi-Level Processing Pipeline

### ✅ Level 1 — Human Detection (Completed)
- **Model:** YOLOv8 Nano (`yolov8n.pt`) for real-time performance
- **Function:** Detects all humans in each frame using COCO class ID `0`
- **Output:** Bounding boxes with "Human Detected" labels drawn on frame

### ✅ Level 2 — Person Tracking (Completed)
- **Model:** YOLOv8 with built-in BoT-SORT/ByteTrack
- **Function:** Assigns persistent unique IDs to each detected person across frames
- **Output:** Tracked bounding boxes with `Person ID: X` labels
- **Key Feature:** `persist=True` enables ID consistency across frames

### ✅ Level 3 — Face Recognition (Completed)
- **Model:** FaceNet via DeepFace library
- **Function:** Extracts 128-dimensional face embeddings and matches against registered identities in the database
- **Output:** Identity label overlaid on video (e.g., `Identity: Bharat (ID: 1)`)
- **Matching:** Cosine distance between embeddings with configurable threshold

### ✅ Identity Database (Completed)
- **Storage:** SQLite via SQLAlchemy ORM
- **Schema:** Person profiles with name, face embedding (JSON), face image path, age, gender, timestamps, sighting count
- **API:** Full REST API for person registration, listing, retrieval, and deletion
- **Registration:** Upload a face photo via `/persons/register` endpoint — embedding is auto-extracted and stored

---

## 📁 Project Structure

```
cctv_system/
├── app/
│   ├── main.py                              # FastAPI application entry point
│   ├── api/
│   │   └── persons.py                       # REST API routes for person management
│   ├── core/
│   │   └── database.py                      # SQLAlchemy engine & session setup
│   ├── models/
│   │   └── person.py                        # Person ORM model
│   ├── services/
│   │   ├── level_1_detection/
│   │   │   └── detector.py                  # YOLOv8 human detection
│   │   ├── level_2_tracking/
│   │   │   └── tracker.py                   # BoT-SORT person tracking
│   │   └── level_3_recognition/
│   │       └── recognizer.py                # DeepFace recognition + DB matching
│   └── utils/                               # Utility functions
├── data/
│   ├── known_faces/                         # Stored face images (organized by name)
│   ├── logs/                                # System logs
│   ├── test_videos/                         # Test video files
│   └── cctv_system.db                       # SQLite database (auto-created)
├── static/                                  # Static assets for dashboard
├── yolov8n.pt                               # YOLOv8 Nano model weights
├── requirements.txt                         # Python dependencies
├── setup.py                                 # Project scaffolding script
└── .gitignore
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Webcam or IP camera
- [uv](https://docs.astral.sh/uv/) package manager (recommended)

### Installation

```bash
# Clone the repository
git clone https://github.com/Bharatgwl/Digital_twin.git
cd Digital_twin

# Create and activate virtual environment
uv venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

### Running the System

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Access Points

| URL | Description |
|-----|-------------|
| `http://localhost:8000` | API health check |
| `http://localhost:8000/video` | Live camera feed with detection, tracking & recognition |
| `http://localhost:8000/docs` | Swagger UI — interactive API documentation |
| `http://localhost:8000/persons/` | List all registered persons |

### Register a Person
1. Open `http://localhost:8000/docs`
2. Navigate to **POST** `/persons/register`
3. Upload a clear face photo, enter name, age, gender
4. The system will now recognize this person in the live feed

---

## 🔮 Future Roadmap

### Phase 1 — Semantic Person Profiling
- [ ] Integrate **Google Gemini API** for generating rich text descriptions of detected persons (clothing, appearance, behavior)
- [ ] Store semantic embeddings in **ChromaDB** (vector database) for similarity search
- [ ] Enable natural language queries: *"Find the person wearing a red jacket"*
- [ ] Auto-generate person profiles combining visual features + recognition data

### Phase 2 — Virtual Twin Visualization
- [ ] Build a **2D floor-plan view** that mirrors the camera's physical space
- [ ] Show real-time **person positions as avatars/dots** on the map
- [ ] Display **movement trajectories** and path history
- [ ] Implement **zone mapping** (camera coordinates → floor-plan coordinates)

### Phase 3 — Analytics Dashboard
- [ ] **Occupancy counter** — real-time people count with historical graphs
- [ ] **Heatmaps** — visualize high-traffic areas over time
- [ ] **Dwell time analysis** — detect how long people stay in specific zones
- [ ] **Peak hour detection** — identify busiest times automatically
- [ ] **Person timeline** — full activity history for each registered individual

### Phase 4 — Smart Alerts & Event System
- [ ] **Unknown person alerts** — flag unregistered faces
- [ ] **Loitering detection** — alert when someone stays in a zone too long
- [ ] **Crowd detection** — alert when occupancy exceeds thresholds
- [ ] **Restricted zone breach** — alert when someone enters a no-go area
- [ ] **WebSocket real-time notifications** to the dashboard

### Phase 5 — Multi-Camera & Aerial Support
- [ ] Support **multiple CCTV feeds** simultaneously
- [ ] **Cross-camera tracking** — maintain person IDs across different cameras
- [ ] **Aerial/drone imagery** analysis for large-area monitoring
- [ ] **Camera network topology** — define spatial relationships between cameras

### Phase 6 — Advanced Intelligence
- [ ] **Behavior analysis** — detect running, fighting, falling
- [ ] **Object interaction** — detect when persons interact with specific objects
- [ ] **Re-identification** — match persons across non-overlapping camera views
- [ ] **Predictive analytics** — forecast occupancy and movement patterns

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend Framework | FastAPI |
| Object Detection | YOLOv8 (Ultralytics) |
| Person Tracking | BoT-SORT / ByteTrack |
| Face Recognition | DeepFace (FaceNet) |
| Database | SQLite + SQLAlchemy |
| Video Processing | OpenCV |
| Vector Store (Planned) | ChromaDB |
| AI Profiling (Planned) | Google Gemini API |
| Dashboard (Planned) | HTML/CSS/JS with WebSocket |
| Package Manager | uv |

---

## 👥 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "feat: add your feature"`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

This project is part of an academic/research initiative for building intelligent Digital Twin systems from CCTV and aerial imagery.
