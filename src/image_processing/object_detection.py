import cv2
import numpy as np

COLOR_PROFILES = {
    "Red": [
        ((0,   120, 200), (6,   230, 255)), 
        ((170, 120, 200), (180, 230, 255))
    ],
    "Blue": [
        ((100, 150, 100), (130, 255, 255))
    ],  
}

class Detector:
    def __init__(self,profile="Red"):
        self.profile = COLOR_PROFILES.get(profile, COLOR_PROFILES["Red"])
        self.kernel = np.ones((3, 3), np.uint8)
        self.candidate = None
        self.stable_count = 0
        self.stable_threshold = 5
        

    def detect(self, img):
        blurred = cv2.GaussianBlur(img, (3, 3), 0)

        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv,self.profile[0][0], self.profile[0][1])
        for lower, upper in self.profile[1:]:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))

        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 100:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h
            if aspect_ratio < 0.5 or aspect_ratio > 2:
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < 0.5:
                continue

            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            score = area * circularity

            detections.append((cx, cy, x, y, w, h, score))

        if not detections:
            return None, cleaned
        
        best = max(detections, key=lambda d: d[6])
        return best[:6], cleaned
    
    def is_stable(self,detection):
        if detection is None:
            self.candidate = None
            self.stable_count = 0
        else:
            if self.candidate is None:
                self.candidate = detection
                self.stable_count = 1
            else:
                dist = np.sqrt((detection[0] - self.candidate[0]) ** 2 + (detection[1] - self.candidate[1]) ** 2)
                if dist < 50:
                    self.stable_count += 1
                    self.candidate = detection
                    if self.stable_count >= self.stable_threshold:
                        return True
                else:
                    self.candidate = detection
                    self.stable_count = 1
            

if __name__ == "__main__":
    cap      = cv2.VideoCapture(0)
    detector = Detector(profile="Red")

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        best, mask = detector.detect(frame)
        stable = detector.is_stable(best)

        if stable:
            cx, cy, x, y, w, h = best
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.circle(frame, (cx, cy), 5, (255, 255, 0), -1)
            print(f"Detected object at ({cx}, {cy}) with size ({w}x{h})")

        cv2.imshow("Frame", frame)
        cv2.imshow("Mask", mask)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()