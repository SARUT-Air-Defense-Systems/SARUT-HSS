import cv2
import numpy as np

def nothing(x):
    pass

def run_calibration(source=0):
    cap = cv2.VideoCapture(source)

    cv2.namedWindow("Calibration")
    cv2.createTrackbar("H_min", "Calibration", 0,   180, nothing)
    cv2.createTrackbar("H_max", "Calibration", 10,  180, nothing)
    cv2.createTrackbar("S_min", "Calibration", 120, 255, nothing)
    cv2.createTrackbar("S_max", "Calibration", 255, 255, nothing)
    cv2.createTrackbar("V_min", "Calibration", 80,  255, nothing)
    cv2.createTrackbar("V_max", "Calibration", 255, 255, nothing)

    # second range for red (wraps around 180)
    cv2.createTrackbar("H2_min", "Calibration", 170, 180, nothing)
    cv2.createTrackbar("H2_max", "Calibration", 180, 180, nothing)
    cv2.createTrackbar("Use_H2",  "Calibration", 1,   1,  nothing)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h_min = cv2.getTrackbarPos("H_min", "Calibration")
        h_max = cv2.getTrackbarPos("H_max", "Calibration")
        s_min = cv2.getTrackbarPos("S_min", "Calibration")
        s_max = cv2.getTrackbarPos("S_max", "Calibration")
        v_min = cv2.getTrackbarPos("V_min", "Calibration")
        v_max = cv2.getTrackbarPos("V_max", "Calibration")
        h2_min  = cv2.getTrackbarPos("H2_min",  "Calibration")
        h2_max  = cv2.getTrackbarPos("H2_max",  "Calibration")
        use_h2  = cv2.getTrackbarPos("Use_H2",  "Calibration")

        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv     = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv,
                           (h_min, s_min, v_min),
                           (h_max, s_max, v_max))

        if use_h2:
            mask2 = cv2.inRange(hsv,
                                (h2_min, s_min, v_min),
                                (h2_max, s_max, v_max))
            mask = cv2.bitwise_or(mask, mask2)

        # overlay mask on frame so you see whats detected
        result = cv2.bitwise_and(frame, frame, mask=mask)

        # print current values on screen
        values_text = (f"H:[{h_min}-{h_max}]  S:[{s_min}-{s_max}]  "
                       f"V:[{v_min}-{v_max}]  H2:[{h2_min}-{h2_max}] use_h2={use_h2}")
        cv2.putText(result, values_text, (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        cv2.imshow("Calibration", result)
        cv2.imshow("Mask", mask)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            print("\n--- Copy these into detect.py ---")
            print(f"LOWER1 = ({h_min}, {s_min}, {v_min})")
            print(f"UPPER1 = ({h_max}, {s_max}, {v_max})")
            if use_h2:
                print(f"LOWER2 = ({h2_min}, {s_min}, {v_min})")
                print(f"UPPER2 = ({h2_max}, {s_max}, {v_max})")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_calibration(source=0)