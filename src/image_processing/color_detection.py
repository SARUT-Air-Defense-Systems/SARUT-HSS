import cv2
import numpy as np

cv2.namedWindow("Frame", cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)

class ColorDetect:
    def __init__(self):
        self.red1_lower = (0,180,120)
        self.red1_upper = (10,255,255)
        self.red2_lower = (170,180,120)
        self.red2_upper = (179,255,255)
        self.blue_lower = (100,150,100)
        self.blue_upper = (130,255,255)
        self.mask_red = None
        self.mask_blue = None
        self.lower_skin = (0,30,60)
        self.upper_skin = (20,180,255)

    def detect(self,frame):
        hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        mask_red1 = cv2.inRange(hsv,self.red1_lower,self.red1_upper)
        mask_red2 = cv2.inRange(hsv,self.red2_lower,self.red2_upper)
        mask_skin = cv2.inRange(hsv,self.lower_skin,self.upper_skin)
        self.mask_red = cv2.bitwise_or(mask_red1,mask_red2)
        self.mask_red = cv2.bitwise_and(self.mask_red,cv2.bitwise_not(mask_skin))
        self.mask_blue = cv2.inRange(hsv,self.blue_lower,self.blue_upper)
        red_count = cv2.countNonZero(self.mask_red)
        blue_count = cv2.countNonZero(self.mask_blue)
        if red_count > blue_count and red_count > 50:
            return "Red"
        elif blue_count > red_count and blue_count > 50:
            return "Blue"
        else:
            return "None"
        
if __name__ == "__main__":

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    detector = ColorDetect()
    while True:
        ret,frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        color = detector.detect(frame)
        print(color)
        cv2.putText(frame, f"Detected Color: {color}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Frame",frame)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


 