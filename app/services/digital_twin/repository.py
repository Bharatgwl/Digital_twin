from datetime import datetime, timezone
from typing import Optional

import numpy as np
from bson import ObjectId
from pymongo import ReturnDocument
from pymongo.database import Database

from app.services.feature_extraction.extractor import ExtractedFeatures


class DigitalTwinRepository:
    def __init__(self, db: Database, match_threshold: float = 0.35):
        self.db = db
        self.match_threshold = match_threshold
        self.profiles = db.profiles
        self.observations = db.observations
        self.visits = db.visits
        self.counters = db.counters

    @staticmethod
    def _cosine_distance(embedding_a: list[float], embedding_b: list[float]) -> float:
        a = np.array(embedding_a, dtype=np.float32)
        b = np.array(embedding_b, dtype=np.float32)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return float("inf")
        return float(1 - np.dot(a, b) / (norm_a * norm_b))

    def _next_person_id(self) -> str:
        counter = self.counters.find_one_and_update(
            {"_id": "person_id_counter"},
            {"$inc": {"seq": 1}},
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )
        return f"DT-{counter['seq']:06d}"

    def find_match(self, face_embedding: Optional[list[float]]) -> Optional[dict]:
        if face_embedding is None:
            return None

        best_profile = None
        best_distance = float("inf")
        for profile in self.profiles.find(
            {"face_embedding": {"$type": "array"}},
            {"person_id": 1, "face_embedding": 1},
        ):
            distance = self._cosine_distance(face_embedding, profile["face_embedding"])
            if distance < best_distance:
                best_distance = distance
                best_profile = profile

        if best_profile and best_distance <= self.match_threshold:
            return best_profile
        return None

    def create_profile(self, features: ExtractedFeatures, camera_id: str) -> dict:
        now = datetime.now(timezone.utc)
        person_id = self._next_person_id()
        profile = {
            "person_id": person_id,
            "display_name": person_id,
            "face_embedding": features.face_embedding,
            "first_seen_at": now,
            "last_seen_at": now,
            "detection_count": 0,
            "visit_count": 0,
            "last_camera_id": camera_id,
            "dominant_clothing_color": features.clothing_color,
            "avg_height_px": float(features.estimated_height_px),
            "last_snapshot_path": None,
            "created_at": now,
            "updated_at": now,
        }
        self.profiles.insert_one(profile)
        return profile

    def update_profile(
        self,
        person_id: str,
        features: ExtractedFeatures,
        camera_id: str,
        snapshot_path: Optional[str],
    ) -> None:
        profile = self.profiles.find_one({"person_id": person_id})
        if profile is None:
            raise RuntimeError(f"Digital twin profile not found for person_id={person_id}")

        old_count = int(profile.get("detection_count", 0))
        new_count = old_count + 1
        old_avg = float(profile.get("avg_height_px") or features.estimated_height_px or 0)
        new_avg = ((old_avg * old_count) + float(features.estimated_height_px)) / max(new_count, 1)
        now = datetime.now(timezone.utc)

        update_fields = {
            "detection_count": new_count,
            "last_seen_at": now,
            "last_camera_id": camera_id,
            "dominant_clothing_color": features.clothing_color,
            "avg_height_px": new_avg,
            "updated_at": now,
        }
        if profile.get("face_embedding") is None and features.face_embedding is not None:
            update_fields["face_embedding"] = features.face_embedding
        if snapshot_path:
            update_fields["last_snapshot_path"] = snapshot_path

        self.profiles.update_one({"person_id": person_id}, {"$set": update_fields})

    def open_visit(self, person_id: str, camera_id: str) -> ObjectId:
        now = datetime.now(timezone.utc)
        visit_doc = {
            "person_id": person_id,
            "camera_id": camera_id,
            "entry_time": now,
            "exit_time": None,
            "frames_seen": 0,
        }
        result = self.visits.insert_one(visit_doc)
        self.profiles.update_one(
            {"person_id": person_id},
            {"$inc": {"visit_count": 1}, "$set": {"updated_at": now}},
        )
        return result.inserted_id

    def increment_visit_frames(self, visit_id: ObjectId) -> None:
        self.visits.update_one({"_id": visit_id}, {"$inc": {"frames_seen": 1}})

    def close_visit(self, visit_id: ObjectId) -> None:
        self.visits.update_one(
            {"_id": visit_id, "exit_time": None},
            {"$set": {"exit_time": datetime.now(timezone.utc)}},
        )

    def add_observation(
        self,
        person_id: str,
        camera_id: str,
        track_id: int,
        bbox: tuple[int, int, int, int],
        features: ExtractedFeatures,
        confidence: float,
        snapshot_path: Optional[str],
    ) -> None:
        x1, y1, x2, y2 = bbox
        self.observations.insert_one(
            {
                "person_id": person_id,
                "camera_id": camera_id,
                "track_id": track_id,
                "captured_at": datetime.now(timezone.utc),
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "center": {"x": features.center[0], "y": features.center[1]},
                "clothing_color": features.clothing_color,
                "estimated_height_px": features.estimated_height_px,
                "confidence": confidence,
                "snapshot_path": snapshot_path,
                "face_detected": features.face_embedding is not None,
            }
        )
