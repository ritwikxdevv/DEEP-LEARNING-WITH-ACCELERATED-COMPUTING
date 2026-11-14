
import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("traffic.mp4")

entry_count = 0
exit_count = 0

previous_positions = {}

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    line_y = 650

    results = model.track(
        frame,
        persist=True,
        classes=[2, 3, 5, 7],
        tracker="bytetrack.yaml"
    )

    for result in results:
        if result.boxes.id is not None:

            ids = result.boxes.id.cpu().numpy()
            boxes = result.boxes.xyxy.cpu().numpy()

            for box, track_id in zip(boxes, ids):

                x1, y1, x2, y2 = box

                center_x = int((x1 + x2) / 2)
                center_y = int(y2)

                track_id = int(track_id)

                if track_id in previous_positions:

                    previous_y = previous_positions[track_id]

                    if previous_y < line_y and center_y >= line_y:
                        entry_count += 1
                        print("ENTRY DETECTED:", track_id)

                    elif previous_y > line_y and center_y <= line_y:
                        exit_count += 1
                        print("EXIT DETECTED:", track_id)

                previous_positions[track_id] = center_y

    frame = results[0].plot()

    cv2.line(
        frame,
        (0, line_y),
        (frame.shape[1], line_y),
        (0, 255, 255),
        3
    )

    cv2.putText(
        frame,
        "Entry: " + str(entry_count),
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "Exit: " + str(exit_count),
        (30, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    cv2.imshow("Entry Exit Vehicle Count", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
