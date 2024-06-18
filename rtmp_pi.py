import cv2
import subprocess
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8x.pt")

# Source RTMP stream (input)
video_path = "rtmp://163.17.23.31:1935/live/test2"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Destination RTMP stream (output)
twitch_rtmp_url = "rtmp://163.17.23.31:1935/live/view"

# Parse the Twitch RTMP URL to extract the server URL and stream key
parts = twitch_rtmp_url.split("/")
rtmp_server = "/".join(parts[:-1]) + "/"
stream_key = parts[-1]

# Get frame dimensions
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Construct the ffmpeg command to stream the processed frames
ffmpeg_command = [
    "ffmpeg",
    "-y",  # Overwrite output files without asking
    "-f", "rawvideo",
    "-vcodec", "rawvideo",
    "-pix_fmt", "bgr24",
    "-s", f"{frame_width}x{frame_height}",  # Frame size
    "-r", "30",  # Frame rate
    "-i", "-",  # Input from stdin
    "-c:v", "libx264",
    "-preset", "ultrafast",
    "-tune", "zerolatency",  # Tune for low latency
    "-pix_fmt", "yuv420p",
    "-f", "flv",
    rtmp_server + stream_key
]

try:
    # Start the ffmpeg process
    stream_process = subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE)
except FileNotFoundError:
    print("Error: FFmpeg not found. Ensure FFmpeg is installed and in your PATH.")
    exit()
except Exception as e:
    print(f"Error starting FFmpeg: {e}")
    exit()

while True:
    # Read frame from live stream
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read frame from stream.")
        break

    # Perform object detection on the frame
    try:
        results = model(frame, device="0", conf=0.2)
    except Exception as e:
        print(f"Error during model inference: {e}")
        break

    # Get annotated frame with detected objects
    annotated_frame = results[0].plot()

    # Write the frame to ffmpeg's stdin
    try:
        stream_process.stdin.write(annotated_frame.tobytes())
    except BrokenPipeError:
        print("Error: Broken pipe while writing to FFmpeg.")
        break
    except Exception as e:
        print(f"Error writing frame to FFmpeg: {e}")
        break

    # Display annotated frame with detected objects
    cv2.imshow("YOLOv9 Inference", annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
if stream_process.stdin:
    stream_process.stdin.close()
stream_process.wait()
cv2.destroyAllWindows()
