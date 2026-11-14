
import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("traffic.mp4")

vehicle_positions = {}
wrong_way_ids = set()

# Expected traffic direction: down
# Change to "up" if vehicles normally move upward
expected_direction = "down"

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

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

                x1, y1, x2, y2 = box

                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                track_id = int(track_id)

                if track_id not in vehicle_positions:
                    vehicle_positions[track_id] = []

                vehicle_positions[track_id].append(
                    (center_x, center_y)
                )

                positions = vehicle_positions[track_id]

                # Keep recent positions
                if len(positions) > 15:
                    positions.pop(0)

                if len(positions) >= 10:

                    old_y = positions[0][1]
                    new_y = positions[-1][1]

                    movement = new_y - old_y

                    # Detect movement opposite to expected direction
                    if expected_direction == "down":
                        if movement < -30:
                            wrong_way_ids.add(track_id)

                    elif expected_direction == "up":
                        if movement > 30:
                            wrong_way_ids.add(track_id)

    frame = results[0].plot()

    cv2.putText(
        frame,
        "Wrong Way Vehicles: " + str(len(wrong_way_ids)),
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    if len(wrong_way_ids) > 0:
        cv2.putText(
            frame,
            "WRONG WAY ALERT!",
            (30, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

    cv2.imshow("Wrong Way Vehicle Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
