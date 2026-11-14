
import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("traffic.mp4")

triple_riding_ids = set()

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    results = model.track(
        frame,
        persist=True,
        classes=[0, 3],
        tracker="bytetrack.yaml",
        conf=0.3
    )

    for result in results:

        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()

        if result.boxes.id is not None:
            ids = result.boxes.id.cpu().numpy()
        else:
            ids = [-1] * len(boxes)

        people = []
        motorcycles = []

        # Separate people and motorcycles
        for box, cls, track_id in zip(boxes, classes, ids):

            x1, y1, x2, y2 = box
            track_id = int(track_id)

            if int(cls) == 0:
                people.append(box)

            elif int(cls) == 3:
                motorcycles.append((box, track_id))

        # Check people near each motorcycle
        for bike_box, bike_id in motorcycles:

            bx1, by1, bx2, by2 = bike_box

            bike_width = bx2 - bx1
            bike_height = by2 - by1

            # Approximate rider region around motorcycle
            region_x1 = bx1 - bike_width * 0.4
            region_x2 = bx2 + bike_width * 0.4

            region_y1 = by1 - bike_height * 1.5
            region_y2 = by2 + bike_height * 0.2

            person_count = 0

            for person_box in people:

                px1, py1, px2, py2 = person_box

                person_center_x = (px1 + px2) / 2
                person_bottom_y = py2

                if (
                    region_x1 < person_center_x < region_x2
                    and region_y1 < person_bottom_y < region_y2
                ):
                    person_count += 1

            # More than two persons detected
            if person_count > 2:

                triple_riding_ids.add(bike_id)

                cv2.putText(
                    frame,
                    "TRIPLE RIDING ALERT!",
                    (int(bx1), max(int(by1) - 30, 30)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

                cv2.rectangle(
                    frame,
                    (int(bx1), int(by1)),
                    (int(bx2), int(by2)),
                    (0, 0, 255),
                    3
                )

            else:

                cv2.rectangle(
                    frame,
                    (int(bx1), int(by1)),
                    (int(bx2), int(by2)),
                    (0, 255, 0),
                    2
                )

            cv2.putText(
                frame,
                "Persons: " + str(person_count),
                (int(bx1), int(by2) + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2
            )

    cv2.putText(
        frame,
        "Triple Riding Alerts: " + str(len(triple_riding_ids)),
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    cv2.imshow("Triple Riding Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
