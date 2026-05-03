from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from deepface import DeepFace


@dataclass
class ExtractedFeatures:
    face_embedding: Optional[list[float]]
    clothing_color: str
    estimated_height_px: int
    center: Tuple[int, int]


class FeatureExtractor:
    def __init__(self, model_name: str = "Facenet"):
        self.model_name = model_name
        self._color_palette_bgr = {
            "black": np.array([10, 10, 10], dtype=np.float32),
            "white": np.array([245, 245, 245], dtype=np.float32),
            "gray": np.array([128, 128, 128], dtype=np.float32),
            "red": np.array([40, 40, 210], dtype=np.float32),
            "green": np.array([40, 170, 40], dtype=np.float32),
            "blue": np.array([210, 50, 40], dtype=np.float32),
            "yellow": np.array([40, 210, 210], dtype=np.float32),
            "orange": np.array([30, 140, 220], dtype=np.float32),
            "brown": np.array([40, 75, 130], dtype=np.float32),
            "purple": np.array([145, 75, 160], dtype=np.float32),
        }

    def _clip_bbox(self, frame, bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        return max(x1, 0), max(y1, 0), min(x2, w), min(y2, h)

    def _extract_face_embedding(self, person_crop) -> Optional[list[float]]:
        embeddings = DeepFace.represent(
            img_path=person_crop,
            model_name=self.model_name,
            enforce_detection=False,
        )
        if not embeddings:
            return None

        first_embedding = embeddings[0].get("embedding")
        if first_embedding is None:
            return None
        return [float(value) for value in first_embedding]

    def _dominant_clothing_color(self, person_crop) -> str:
        if person_crop.size == 0:
            return "unknown"

        h = person_crop.shape[0]
        clothing_region = person_crop[h // 2 :, :]
        if clothing_region.size == 0:
            clothing_region = person_crop

        avg_bgr = clothing_region.reshape(-1, 3).mean(axis=0).astype(np.float32)
        best_color_name = "unknown"
        best_distance = float("inf")
        for color_name, color_bgr in self._color_palette_bgr.items():
            distance = float(np.linalg.norm(avg_bgr - color_bgr))
            if distance < best_distance:
                best_distance = distance
                best_color_name = color_name
        return best_color_name

    def extract(
        self,
        frame,
        bbox: Tuple[int, int, int, int],
        include_face_embedding: bool = True,
    ) -> ExtractedFeatures:
        x1, y1, x2, y2 = self._clip_bbox(frame, bbox)
        person_crop = frame[y1:y2, x1:x2]
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        estimated_height_px = max(y2 - y1, 0)

        if person_crop.size == 0:
            return ExtractedFeatures(
                face_embedding=None,
                clothing_color="unknown",
                estimated_height_px=estimated_height_px,
                center=(center_x, center_y),
            )

        face_embedding = None
        if include_face_embedding:
            face_embedding = self._extract_face_embedding(person_crop)
        clothing_color = self._dominant_clothing_color(person_crop)

        return ExtractedFeatures(
            face_embedding=face_embedding,
            clothing_color=clothing_color,
            estimated_height_px=estimated_height_px,
            center=(center_x, center_y),
        )
