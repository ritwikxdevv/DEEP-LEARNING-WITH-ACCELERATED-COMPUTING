
import cv2
import math
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

video = cv2.VideoCapture("traffic.mp4")

fps = video.get(cv2.CAP_PROP_FPS)

previous_positions = {}

meters_per_pixel = 0.05

while True:
    ret, frame = video.read()

    if not ret:
        break

    results = model.track(frame, persist=True,
                          classes=[2, 3, 5, 7],
                          tracker="bytetrack.yaml")

    for result in results:
        if result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            ids = result.boxes.id.cpu().numpy()

            for box, track_id in zip(boxes, ids):
                x1, y1, x2, y2 = map(int, box)

                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                track_id = int(track_id)

                speed = 0

                if track_id in previous_positions:
                    prev_x, prev_y = previous_positions[track_id]

                    distance = math.sqrt(
                        (center_x - prev_x) ** 2 +
                        (center_y - prev_y) ** 2
                    )

                    speed = distance * meters_per_pixel * fps * 3.6

                previous_positions[track_id] = (
                    center_x, center_y
                )

                label = "ID " + str(track_id) + " Speed: " + str(round(speed, 2)) + " km/h"

                cv2.rectangle(frame, (x1, y1), (x2, y2),
                              (0, 255, 0), 2)

                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), 2)

    cv2.imshow("Task 13 - Vehicle Speed Estimation", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
