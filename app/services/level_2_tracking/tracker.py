from dataclasses import dataclass
from typing import List, Tuple

import cv2
from ultralytics import YOLO


@dataclass
class TrackedPerson:
    track_id: int
    bbox: Tuple[int, int, int, int]
    confidence: float


class HumanTracker:
    def __init__(self, model_path: str = "yolov8n.pt"):
        self.model = YOLO(model_path)

    def track(self, frame) -> Tuple[object, List[TrackedPerson]]:
        results = self.model.track(frame, persist=True, classes=[0], verbose=False)
        tracked_people: List[TrackedPerson] = []

        if not results or results[0].boxes is None or results[0].boxes.id is None:
            return frame, tracked_people

        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        confidences = (
            results[0].boxes.conf.cpu().numpy().tolist()
            if results[0].boxes.conf is not None
            else [0.0] * len(ids)
        )

        for box, person_track_id, confidence in zip(boxes, ids, confidences):
            x1, y1, x2, y2 = box
            tracked_people.append(
                TrackedPerson(
                    track_id=int(person_track_id),
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    confidence=float(confidence),
                )
            )

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(
                frame,
                f"Track ID: {person_track_id}",
                (x1, max(y1 - 10, 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 0, 0),
                2,
            )

        return frame, tracked_people
