
import cv2

# Read the captured image
img = cv2.imread("traffic_image.jpg")

# Check whether image is loaded
if img is None:
    print("Error: Image not found!")

else:
    print("Image loaded successfully!")

    # Display the image
    cv2.imshow("VisionPort - Traffic Image", img)

    # Wait until a key is pressed
    cv2.waitKey(0)

    # Close all OpenCV windows
    cv2.destroyAllWindows()

    print("Image window closed successfully!")