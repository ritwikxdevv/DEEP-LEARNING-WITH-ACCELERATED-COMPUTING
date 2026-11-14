
import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

video = cv2.VideoCapture("traffic.mp4")

while True:
    ret, frame = video.read()

    if not ret:
        break

    results = model(frame, classes=[2, 3, 5, 7])

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            class_id = int(box.cls[0])
            confidence = float(box.conf[0]) * 100

            name = model.names[class_id]
            label = name + " " + str(round(confidence, 2)) + "%"

            cv2.rectangle(frame, (x1, y1), (x2, y2),
                          (0, 255, 0), 2)

            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0), 2)

    cv2.imshow("Task 8 - Vehicle Confidence", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
