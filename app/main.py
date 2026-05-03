import os

import cv2
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.api.digital_twins import router as digital_twins_router
from app.core.mongo import get_db, initialize_indexes
from app.services.frame_capture.capture import FrameCapture
from app.services.pipeline.runtime import DigitalTwinRuntime


VIDEO_SOURCE = os.getenv("VIDEO_SOURCE", "0")
CAMERA_ID = os.getenv("CAMERA_ID", "CAM-1")
FRAME_WIDTH = int(os.getenv("FRAME_WIDTH", "1280"))
FRAME_HEIGHT = int(os.getenv("FRAME_HEIGHT", "720"))
SNAPSHOT_INTERVAL = int(os.getenv("SNAPSHOT_INTERVAL", "20"))

os.makedirs("static", exist_ok=True)
os.makedirs("data", exist_ok=True)

db = get_db()
initialize_indexes(db)

capture = FrameCapture(
    source=VIDEO_SOURCE,
    width=FRAME_WIDTH,
    height=FRAME_HEIGHT,
    buffer_size=30,
)
runtime = DigitalTwinRuntime(
    db=db,
    camera_id=CAMERA_ID,
    snapshot_interval=SNAPSHOT_INTERVAL,
    snapshot_root="data/snapshots",
)

app = FastAPI(title="CCTV Digital Twin System")
app.include_router(digital_twins_router)
app.mount("/static", StaticFiles(directory="static"), name="static")
app.state.runtime = runtime


@app.on_event("startup")
def ensure_database_connection():
    db.command("ping")


def stream_camera():
    try:
        while True:
            frame = capture.read()
            if frame is None:
                continue

            processed_frame, active_count = runtime.process_frame(frame)
            cv2.putText(
                processed_frame,
                f"Active persons: {active_count}",
                (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )

            encoded, buffer = cv2.imencode(".jpg", processed_frame)
            if not encoded:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )
    finally:
        capture.release()


@app.get("/", include_in_schema=False)
def dashboard():
    return FileResponse(os.path.join("static", "index.html"))


@app.get("/health")
def health():
    return {"status": "ok", "database": db.name}


@app.get("/video")
def video_feed():
    return StreamingResponse(
        stream_camera(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
