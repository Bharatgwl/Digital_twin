import cv2
from ultralytics import YOLO

class HumanDetector:
    def __init__(self):
        # Using YOLOv8 Nano for real-time performance [cite: 25, 26]
        self.model = YOLO('yolov8n.pt') 
        # Class ID 0 is 'person' in the COCO dataset [cite: 31]
        self.target_class = 0 

    def detect(self, frame):
        results = self.model(frame, verbose=False)
        detections = []

        for result in results:
            for box in result.boxes:
                if int(box.cls[0]) == self.target_class:
                    # x1, y1, x2, y2 coordinates [cite: 32]
                    coords = box.xyxy[0].tolist()
                    detections.append(coords)
                    
                    # Draw visual results [cite: 33, 34]
                    x1, y1, x2, y2 = map(int, coords)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, "Human Detected", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame, detections