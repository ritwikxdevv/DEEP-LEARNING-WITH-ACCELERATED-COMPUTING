
import cv2
import math
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("traffic.mp4")

vehicle_history = {}
wrong_turn_ids = set()

# Set the prohibited turn
PROHIBITED_TURNS = {"left"}

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Intersection detection area
    x1, y1 = 650, 450
    x2, y2 = 1250, 850

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

                # Check whether vehicle is inside intersection
                if x1 < center_x < x2 and y1 < center_y < y2:

                    if track_id not in vehicle_history:
                        vehicle_history[track_id] = []

                    vehicle_history[track_id].append(
                        (center_x, center_y)
                    )

                    points = vehicle_history[track_id]

                    # Keep recent positions only
                    if len(points) > 12:
                        points.pop(0)

                    if len(points) >= 10:

                        p1 = points[0]
                        p2 = points[4]
                        p3 = points[9]

                        dx1 = p2[0] - p1[0]
                        dy1 = p2[1] - p1[1]

                        dx2 = p3[0] - p2[0]
                        dy2 = p3[1] - p2[1]

                        cross = dx1 * dy2 - dy1 * dx2
                        dot = dx1 * dx2 + dy1 * dy2

                        angle = math.degrees(
                            math.atan2(cross, dot)
                        )

                        turn = ""

                        if angle > 45:
                            turn = "right"

                        elif angle < -45:
                            turn = "left"

                        # Detect prohibited turn
                        if turn in PROHIBITED_TURNS:
                            if track_id not in wrong_turn_ids:

                                wrong_turn_ids.add(track_id)

                                print(
                                    "WRONG TURN DETECTED:",
                                    track_id,
                                    turn
                                )

    frame = results[0].plot()

    # Draw intersection area
    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Wrong Turns: " + str(len(wrong_turn_ids)),
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    if len(wrong_turn_ids) > 0:
        cv2.putText(
            frame,
            "WRONG TURN ALERT!",
            (30, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

    cv2.imshow("Wrong Turn Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
