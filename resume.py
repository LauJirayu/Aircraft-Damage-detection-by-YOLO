from ultralytics import YOLO
import os
import torchvision
import torch
import multiprocessing

def main():
    # Set environment variable to avoid 'OMP: Error #15: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized.' issue
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    # Initialize YOLO model with GPU and yolov8n.yaml configuration
    model = YOLO("C:/Users/USER/runs/detect/train5/weights/last.pt")

    # Train the model with GPU
    results = model.train(resume=True, amp=False, patience=0)
if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()  