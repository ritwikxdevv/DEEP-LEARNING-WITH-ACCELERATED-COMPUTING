
import cv2

cap = cv2.VideoCapture("traffic.mp4")

if not cap.isOpened():
    print("ERROR: Cannot open video!")

else:
    print("Traffic video loaded successfully!")

    # Create a manually resizable window
    window_name = "VisionPort - Traffic Video"

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Force window dimensions
    cv2.resizeWindow(window_name, 640, 360)

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Video playback completed!")
            break

        # Resize frame to exactly 640 x 360
        resized_frame = cv2.resize(frame, (640, 360))

        print(
            "Original:", frame.shape,
            "| Display:", resized_frame.shape
        )

        cv2.imshow(window_name, resized_frame)

        # Press Q to quit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            print("Video closed!")
            break

cap.release()
cv2.destroyAllWindows()

print("Video resources released!")
