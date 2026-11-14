
import cv2
import os

video = cv2.VideoCapture("traffic.mp4")

os.makedirs("frames", exist_ok=True)

count = 0

while True:
    ret, frame = video.read()

    if not ret:
        break

    cv2.imwrite("frames/frame%d.jpg" % count, frame)
    count += 1

video.release()

print("Frames extracted successfully")
print("Total frames:", count)
