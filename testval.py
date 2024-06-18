from ultralytics import YOLO
import os
import torchvision
import torch
import multiprocessing

def main():
    # Load a model
    model = YOLO("C:/Users/USER/runs/detect/9c1000/weights/best.pt")


    # Customize validation settings
    validation_results = model.val(data='C:/Users/USER/Downloads/Innovation Hangar v2.v1-aug.yolov9/data.yaml',
                               #imgsz=640,
                               batch=2,
                               #conf=0.25,
                               #iou=0.6,
                               save_json=True,
                               #save_hybrid=True,
                               device='0')
    validation_results.box.maps 
if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()