

import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

video = cv2.VideoCapture("traffic.mp4")

while True:
    ret, frame = video.read()

    if not ret:
        break

    results = model(frame, classes=[2, 3, 5, 7])

    count = 0

    for result in results:
        count = len(result.boxes)

        frame = result.plot()

    cv2.putText(frame, "Vehicle Count: " + str(count),
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 2)

    cv2.imshow("Task 9 - Vehicle Counting", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
