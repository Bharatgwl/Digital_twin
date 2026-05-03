import os
from datetime import datetime
from typing import Optional

import cv2
from pymongo.database import Database

from app.services.digital_twin.repository import DigitalTwinRepository
from app.services.feature_extraction.extractor import FeatureExtractor
from app.services.level_2_tracking.tracker import HumanTracker


class DigitalTwinRuntime:
    def __init__(
        self,
        db: Database,
        camera_id: str = "CAM-1",
        snapshot_interval: int = 15,
        snapshot_root: str = "data/snapshots",
    ):
        self.camera_id = camera_id
        self.snapshot_interval = max(snapshot_interval, 1)
        self.snapshot_root = snapshot_root
        self.frame_count = 0
        self.active_tracks: dict[str, dict] = {}

        self.tracker = HumanTracker()
        self.extractor = FeatureExtractor()
        self.repository = DigitalTwinRepository(db=db)

        os.makedirs(snapshot_root, exist_ok=True)

    def _track_key(self, camera_id: str, track_id: int) -> str:
        return f"{camera_id}:{track_id}"

    def _save_snapshot(self, frame, bbox: tuple[int, int, int, int], person_id: str) -> Optional[str]:
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, w), min(y2, h)
        person_crop = frame[y1:y2, x1:x2]
        if person_crop.size == 0:
            return None

        person_dir = os.path.join(self.snapshot_root, person_id)
        os.makedirs(person_dir, exist_ok=True)

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{timestamp}.jpg"
        filepath = os.path.join(person_dir, filename)
        cv2.imwrite(filepath, person_crop)
        return filepath

    def _close_missing_tracks(self, camera_id: str, visible_keys: set[str]) -> None:
        stale_keys = [
            key
            for key in self.active_tracks.keys()
            if key.startswith(f"{camera_id}:") and key not in visible_keys
        ]
        for key in stale_keys:
            visit_id = self.active_tracks[key]["visit_id"]
            self.repository.close_visit(visit_id)
            del self.active_tracks[key]

    def process_frame(self, frame):
        self.frame_count += 1
        annotated_frame, tracked_people = self.tracker.track(frame.copy())
        visible_keys: set[str] = set()

        for tracked in tracked_people:
            key = self._track_key(self.camera_id, tracked.track_id)
            visible_keys.add(key)

            state = self.active_tracks.get(key)
            should_extract_embedding = state is None or self.frame_count % 60 == 0
            features = self.extractor.extract(
                frame,
                tracked.bbox,
                include_face_embedding=should_extract_embedding,
            )
            if state is None:
                matched = self.repository.find_match(features.face_embedding)
                if matched is None:
                    matched = self.repository.create_profile(features, self.camera_id)
                person_id = matched["person_id"]
                visit_id = self.repository.open_visit(person_id, self.camera_id)
                state = {"person_id": person_id, "visit_id": visit_id}
                self.active_tracks[key] = state

            person_id = state["person_id"]
            snapshot_path = None
            if self.frame_count % self.snapshot_interval == 0:
                snapshot_path = self._save_snapshot(frame, tracked.bbox, person_id)

            self.repository.update_profile(
                person_id=person_id,
                features=features,
                camera_id=self.camera_id,
                snapshot_path=snapshot_path,
            )
            self.repository.increment_visit_frames(state["visit_id"])
            self.repository.add_observation(
                person_id=person_id,
                camera_id=self.camera_id,
                track_id=tracked.track_id,
                bbox=tracked.bbox,
                features=features,
                confidence=tracked.confidence,
                snapshot_path=snapshot_path,
            )

            x1, y1, _, _ = tracked.bbox
            cv2.putText(
                annotated_frame,
                f"Twin: {person_id}",
                (x1, max(y1 - 28, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2,
            )
            cv2.putText(
                annotated_frame,
                f"Color: {features.clothing_color}",
                (x1, max(y1 - 8, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2,
            )

        self._close_missing_tracks(self.camera_id, visible_keys)
        return annotated_frame, len(visible_keys)

    def get_live_stats(self) -> dict:
        return {
            "active_tracks": len(self.active_tracks),
            "processed_frames": self.frame_count,
        }
