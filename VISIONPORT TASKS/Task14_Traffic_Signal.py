


import cv2
import numpy as np

video = cv2.VideoCapture("traffic.mp4")

while True:
    ret, frame = video.read()

    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    red1 = cv2.inRange(hsv, np.array([0, 100, 100]),
                       np.array([10, 255, 255]))

    red2 = cv2.inRange(hsv, np.array([160, 100, 100]),
                       np.array([180, 255, 255]))

    red = cv2.bitwise_or(red1, red2)

    yellow = cv2.inRange(hsv, np.array([15, 100, 100]),
                         np.array([35, 255, 255]))

    green = cv2.inRange(hsv, np.array([40, 50, 50]),
                        np.array([90, 255, 255]))

    red_count = cv2.countNonZero(red)
    yellow_count = cv2.countNonZero(yellow)
    green_count = cv2.countNonZero(green)

    if red_count > yellow_count and red_count > green_count:
        status = "RED SIGNAL"

    elif yellow_count > red_count and yellow_count > green_count:
        status = "YELLOW SIGNAL"

    elif green_count > red_count and green_count > yellow_count:
        status = "GREEN SIGNAL"

    else:
        status = "SIGNAL NOT DETECTED"

    cv2.putText(frame, status, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0), 2)

    cv2.imshow("Task 14 - Traffic Signal Detection", frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
