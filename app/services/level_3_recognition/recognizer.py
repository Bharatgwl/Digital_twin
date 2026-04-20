from deepface import DeepFace
import os

class FaceRecognizer:
    def __init__(self, db_path="data/known_faces"):
        self.db_path = db_path
        # Ensure the directory exists
        if not os.path.exists(self.db_path):
            os.makedirs(self.db_path)

    def recognize(self, frame):
        try:
            # Step 4: Recognition (Compare new face with stored embeddings)
            # This handles Detection + Embedding + Matching in one go
            results = DeepFace.find(
                img_path=frame, 
                db_path=self.db_path, 
                model_name="Facenet", 
                enforce_detection=False,
                silent=True
            )
            
            if len(results) > 0 and not results[0].empty:
                # Get the name from the file path
                full_path = results[0]['identity'][0]
                name = os.path.basename(full_path).split('.')[0]
                return name
            return "Unknown" 
        except Exception as e:
            return "Searching..."