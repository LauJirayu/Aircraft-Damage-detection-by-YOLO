from ultralytics import YOLO
import os
import torchvision
import torch
import multiprocessing

data_path = "C:/Users/USER/runs/data/Custom dataset.v2i.yolov9/data.yaml"
n_epochs = 500
bs = 4
gpu_id = 0
verbose = True
rng = 0
validate = True


def main():
    # Set environment variable to avoid 'OMP: Error #15: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized.' issue
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    # Initialize YOLO model with GPU and yolov8n.yaml configuration
    model = YOLO("C:/Users/USER/runs/detect/train13/weights/best.pt")
    #model = YOLO("C:/Users/USER/runs/detect/train2/weights/best.pt" )
    # Train the model with GPU
    results = model.train(
    data=data_path,
    epochs=n_epochs,
    batch=bs,

    
    patience=0,
    amp=False,
    device=0
    # save_dir=str(save_dir)
)
    #results = model.train(data="C:/Users/user/Desktop/New folder (2)/Aircraft/data.yaml", epochs=1000, amp=False)
if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()



