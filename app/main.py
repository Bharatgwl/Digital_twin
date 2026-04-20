import cv2
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from app.services.level_1_detection.detector import HumanDetector
from app.services.level_2_tracking.tracker import HumanTracker
from app.services.level_3_recognition.recognizer import FaceRecognizer

app = FastAPI(title="CCTV Digital Twin System")

detector = HumanDetector()
tracker = HumanTracker()
recognizer = FaceRecognizer()


def stream_camera():
    # Step 1: Capture from Webcam (0) or IP Camera [cite: 20, 21]
    video_path = "data/test_videos/TownCentreXVID.mp4"
    camera = cv2.VideoCapture(video_path)
    while True:
        success, frame = camera.read()
        if not success:
            camera.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Level 1: Detect humans [cite: 17, 18]
        # processed_frame, _ = detector.detect(frame)
        processed_frame, current_count = tracker.track(frame)
        name = recognizer.recognize(frame)
        cv2.putText(
            processed_frame,
            f"Identity: {name}",
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
