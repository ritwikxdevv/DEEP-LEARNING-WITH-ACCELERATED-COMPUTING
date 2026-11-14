
import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

video = cv2.VideoCapture("traffic.mp4")

while True:
    ret, frame = video.read()

    if not ret:
        break

    results = model(frame)

    for result in results:
        for box in result.boxes:
            if int(box.cls[0]) == 2:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cv2.rectangle(frame, (x1, y1), (x2, y2),
                              (0, 255, 0), 2)

                cv2.putText(frame, "Car",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2)

    cv2.imshow("Task 6 - Car Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
