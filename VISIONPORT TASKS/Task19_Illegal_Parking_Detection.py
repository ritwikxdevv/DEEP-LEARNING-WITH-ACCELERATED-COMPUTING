
import cv2
import math
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("traffic.mp4")

previous_positions = {}
stationary_frames = {}
illegal_vehicles = set()

# Stationary time threshold (90 frames = about 3 seconds at 30 FPS)
parking_threshold = 90

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    height, width = frame.shape[:2]

    # Restricted parking zone (adjust these coordinates)
    x1 = int(width * 0.10)
    y1 = int(height * 0.60)
    x2 = int(width * 0.40)
    y2 = int(height * 0.95)

    results = model.track(
        frame,
        persist=True,
        classes=[2, 3, 5, 7],
        tracker="bytetrack.yaml"
    )

    for result in results:
        if result.boxes.id is not None:

            boxes = result.boxes.xyxy.cpu().numpy()
            ids = result.boxes.id.cpu().numpy()

            for box, track_id in zip(boxes, ids):

                bx1, by1, bx2, by2 = box

                center_x = int((bx1 + bx2) / 2)
                center_y = int((by1 + by2) / 2)

                track_id = int(track_id)

                inside_zone = (
                    x1 < center_x < x2 and
                    y1 < center_y < y2
                )

                if track_id not in previous_positions:
                    previous_positions[track_id] = (
                        center_x, center_y
                    )
                    stationary_frames[track_id] = 0

                old_x, old_y = previous_positions[track_id]

                distance = math.sqrt(
                    (center_x - old_x) ** 2 +
                    (center_y - old_y) ** 2
                )

                if inside_zone:

                    if distance < 4:
                        stationary_frames[track_id] += 1
                    else:
                        stationary_frames[track_id] = 0

                    if stationary_frames[track_id] >= parking_threshold:
                        illegal_vehicles.add(track_id)

                        print(
                            "ILLEGAL PARKING DETECTED:",
                            track_id
                        )

                else:
                    stationary_frames[track_id] = 0

                previous_positions[track_id] = (
                    center_x, center_y
                )

    frame = results[0].plot()

    # Draw restricted zone
    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (0, 0, 255),
        3
    )

    cv2.putText(
        frame,
        "Restricted Parking Zone",
        (x1, max(y1 - 10, 30)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )

    cv2.putText(
        frame,
        "Illegal Parking: " + str(len(illegal_vehicles)),
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    if len(illegal_vehicles) > 0:
        cv2.putText(
            frame,
            "ILLEGAL PARKING ALERT!",
            (30, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

    cv2.imshow("Illegal Parking Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
