
from Yolo import YoloSeg

import torch
print(torch.cuda.is_available())

# Load a model
model = YoloSeg('./yolov8n-seg.pt')  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data='./cityscapes-seg.yaml', epochs=100, imgsz=640,batch=64)

# model.val()