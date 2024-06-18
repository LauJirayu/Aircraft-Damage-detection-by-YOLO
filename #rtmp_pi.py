import cv2
import subprocess
from ultralytics import YOLO

# Load YOLO model
model = YOLO("C:/Users/USER/runs/detect/9c1000/weights/best.pt")

video_path = "rtmp://163.17.23.31:1935/live/test2"
cap = cv2.VideoCapture(video_path)


# Provided Twitch RTMP URL
#https://www.twitch.tv/yoloiot
twitch_rtmp_url = "rtmp://163.17.23.31:1935/live/view"

# Parse the Twitch RTMP URL to extract the server URL and stream key
parts = twitch_rtmp_url.split("/")
rtmp_server = "/".join(parts[:-1]) + "/"
stream_key = parts[-1]

# Construct the ffmpeg command to capture the entire screen
ffmpeg_command = [
    "ffmpeg",
    "-f", "gdigrab",
    "-framerate", "30",
    "-i", "desktop",
    "-vcodec", "libx264",
    "-preset", "ultrafast",
    "-pix_fmt", "yuv420p",
    "-f", "flv",
    rtmp_server + stream_key
]

# Start streaming
stream_process = subprocess.Popen(ffmpeg_command)

while True:
    # Read frame from live stream
    ret, frame = cap.read()

    # Perform object detection and tracking on the frame
    results = model.track(frame, device="0", conf=0.2)

    # Display annotated frame with detected objects
    annotated_frame = results[0].plot()
    cv2.imshow("YOLOv9 Inference", annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
