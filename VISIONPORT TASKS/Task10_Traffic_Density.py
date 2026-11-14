
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

    if count <= 5:
        density = "Low Traffic"
        color = (0, 255, 0)

    elif count <= 15:
        density = "Medium Traffic"
        color = (0, 255, 255)

    else:
        density = "High Traffic"
        color = (0, 0, 255)

    cv2.putText(frame, "Vehicle Count: " + str(count),
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1, color, 2)

    cv2.putText(frame, "Traffic Density: " + density,
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                1, color, 2)

    cv2.imshow("Task 10 - Traffic Density", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()