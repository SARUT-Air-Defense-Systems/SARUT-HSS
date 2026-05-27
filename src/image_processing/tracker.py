import cv2
import numpy as np
from object_detection import Detector

class TrackedObject:
    def __init__(self,frame,bbox):
        self.tracker = cv2.TrackerCSRT_create()
        self.tracker.init(frame, bbox)
        self.bbox = bbox
        self.velocity_list = []
        self.status = "tracking"

class Tracker:
    def __init__(self):
        self.mode = "searching"
        self.tracked_object = None
        self.detector = Detector(profile="Red")

    def update(self, frame):
        if self.mode == "searching":
            best, mask = self.detector.detect(frame)
            self.stable = self.detector.is_stable(best)
            if self.stable:
                self.tracked_object = TrackedObject(frame, best[2:6])
                self.mode = "tracking"
        elif self.mode == "tracking":
            success, bbox = self.tracked_object.tracker.update(frame)
            if success:
                self.tracked_object.bbox = bbox
            else:
                self.mode = "searching"
                self.tracked_object = None
        
        bbox = self.tracked_object.bbox if self.tracked_object is not None else None
        return self.mode, bbox
    
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    tracker = Tracker()

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        mode, bbox = tracker.update(frame)

        if mode == "tracking" and bbox is not None:
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cx, cy = int(x + w//2), int(y + h//2)
            cv2.line(frame, (cx - 5, cy), (cx + 5, cy), (255, 255, 0), 2)
            cv2.line(frame, (cx, cy - 5), (cx, cy + 5), (255, 255, 0), 2)
            cv2.putText(frame, f"Mode: {mode}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:

            cv2.putText(frame, f"Mode: Searching", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
