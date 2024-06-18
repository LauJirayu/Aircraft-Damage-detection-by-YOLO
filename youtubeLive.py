import cv2
from ultralytics import YOLO
import streamlink

# Load YOLO model
model = YOLO("yolov8n.pt")

# Get the stream URL using streamlink
streams = streamlink.streams("https://www.twitch.tv/willyfan0104")
stream_url = streams["best"].url

# Initialize video capture from live stream
capture = cv2.VideoCapture(stream_url)

while True:
    # Read frame from live stream
    ret, frame = capture.read()

    # Perform object detection and tracking on the frame
    results = model.track(frame, device="0", conf=0.2)

    # Display annotated frame with detected objects
    annotated_frame = results[0].plot()
    cv2.imshow("YOLOv8 Inference", annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
capture.release()
cv2.destroyAllWindows()
