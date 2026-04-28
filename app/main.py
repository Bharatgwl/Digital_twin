import cv2
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from app.core.database import engine, SessionLocal
from app.models.person import Person  # noqa: F401 — needed for table creation
from app.core.database import Base
from app.api.persons import router as persons_router
from app.services.level_1_detection.detector import HumanDetector
from app.services.level_2_tracking.tracker import HumanTracker
from app.services.level_3_recognition.recognizer import FaceRecognizer

# Create all database tables on startup
Base.metadata.create_all(bind=engine)

app = FastAPI(title="CCTV Digital Twin System")

# Register API routes
app.include_router(persons_router)

detector = HumanDetector()
tracker = HumanTracker()
recognizer = FaceRecognizer()


def stream_camera():
    # Step 1: Capture from Webcam (0) or IP Camera [cite: 20, 21]
    # video_path = "data/test_videos/TownCentreXVID.mp4"
    # camera = cv2.VideoCapture(0)
    
    # use tracker  by live camera input
    camera = cv2.VideoCapture(0)
    
    # Create a DB session for the stream
    db = SessionLocal()
    
    try:
        while True:
            success, frame = camera.read()
            
            if not success:
                camera.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # Level 1: Detect humans [cite: 17, 18]
            # processed_frame, _ = detector.detect(frame)
            processed_frame, current_count = tracker.track(frame)
            
            # Level 3: Recognize faces against the database
            name, person_id = recognizer.recognize_from_db(frame, db)
            
            label = f"Identity: {name}"
            if person_id:
                label += f" (ID: {person_id})"
            
            cv2.putText(
                processed_frame,
                label,
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
            # Encode frame as JPEG
            _, buffer = cv2.imencode(".jpg", processed_frame)
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )
    finally:
        db.close()
        camera.release()


@app.get("/")
def root():
    return {"message": "CCTV System Level 1 Active"}


@app.get("/video")
def video_feed():
    # Serves the live feed to the browser [cite: 23]
    return StreamingResponse(
        stream_camera(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    import uvicorn

    # Start the system manager [cite: 121]
    uvicorn.run(app, host="0.0.0.0", port=8000)
