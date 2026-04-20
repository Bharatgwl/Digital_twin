import numpy as np
from ultralytics import YOLO

class HumanTracker:
    def __init__(self):
        # We use the same model but leverage the built-in BoT-SORT/ByteTrack 
        # features in Ultralytics for better stability [cite: 57, 60]
        self.model = YOLO('yolov8n.pt')

    def track(self, frame):
        # Step 3: Combine Detection + Tracking [cite: 58]
        # persist=True tells YOLO to remember IDs across frames
        results = self.model.track(frame, persist=True, classes=[0], verbose=False)
        
        counts = 0
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            counts = len(ids)

            for box, id in zip(boxes, ids):
                x1, y1, x2, y2 = box
                # Step 4: Draw Tracking Info (Unique ID) [cite: 61, 63]
                cv2_color = (255, 0, 0) # Blue for Tracking
                import cv2
                cv2.rectangle(frame, (x1, y1), (x2, y2), cv2_color, 2)
                cv2.putText(frame, f"Person ID: {id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, cv2_color, 2)
        
        return frame, counts