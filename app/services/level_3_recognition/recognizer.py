import json
import os

import cv2
import numpy as np
from deepface import DeepFace
from sqlalchemy.orm import Session

from app.models.person import Person


class FaceRecognizer:
    def __init__(self, db_path="data/known_faces", match_threshold=10.0):
        self.db_path = db_path
        self.match_threshold = match_threshold
        # Ensure the directory exists
        if not os.path.exists(self.db_path):
            os.makedirs(self.db_path)

    def _get_embedding(self, frame):
        """Extract face embedding from a frame using DeepFace/Facenet."""
        try:
            embeddings = DeepFace.represent(
                img_path=frame,
                model_name="Facenet",
                enforce_detection=False,
            )
            if embeddings and len(embeddings) > 0:
                return embeddings[0]["embedding"]
        except Exception:
            pass
        return None

    def _cosine_distance(self, emb1, emb2):
        """Compute cosine distance between two embedding vectors."""
        a = np.array(emb1)
        b = np.array(emb2)
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return float("inf")
        return 1 - (dot / (norm_a * norm_b))

    def recognize(self, frame):
        """Recognize a face from the frame (backward-compatible, no DB)."""
        try:
            results = DeepFace.find(
                img_path=frame,
                db_path=self.db_path,
                model_name="Facenet",
                enforce_detection=False,
                silent=True,
            )

            if len(results) > 0 and not results[0].empty:
                full_path = results[0]["identity"][0]
                name = os.path.basename(full_path).split(".")[0]
                return name
            return "Unknown"
        except Exception:
            return "Searching..."

    def recognize_from_db(self, frame, db: Session):
        """
        Recognize a face by comparing its embedding against all persons in the DB.
        Returns (name, person_id) or ("Unknown", None).
        """
        embedding = self._get_embedding(frame)
        if embedding is None:
            return "No Face", None

        # Fetch all stored persons
        persons = db.query(Person).all()
        if not persons:
            return "Unknown", None

        best_match = None
        best_distance = float("inf")

        for person in persons:
            stored_emb = json.loads(person.face_embedding)
            distance = self._cosine_distance(embedding, stored_emb)
            if distance < best_distance:
                best_distance = distance
                best_match = person

        if best_match and best_distance < self.match_threshold:
            # Update sighting count and last_seen
            best_match.sighting_count += 1
            db.commit()
            return best_match.name, best_match.id

        return "Unknown", None

    def register_person(self, name: str, frame, db: Session, age=None, gender=None):
        """
        Register a new person: extract embedding from frame, save face image, store in DB.
        Returns the created Person object.
        """
        embedding = self._get_embedding(frame)
        if embedding is None:
            raise ValueError("No face detected in the provided image.")

        # Save face image to disk
        face_dir = os.path.join(self.db_path, name)
        os.makedirs(face_dir, exist_ok=True)

        existing = len(os.listdir(face_dir))
        filename = f"{name}_{existing + 1}.jpg"
        filepath = os.path.join(face_dir, filename)
        cv2.imwrite(filepath, frame)

        # Store in database
        person = Person(
            name=name,
            face_embedding=json.dumps(embedding),
            face_image_path=filepath,
            age=age,
            gender=gender,
        )
        db.add(person)
        db.commit()
        db.refresh(person)
        return person