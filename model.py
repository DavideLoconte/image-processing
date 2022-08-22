import numpy as np
import torch.hub

CLASS = 0

def get_yolo(path):
    return torch.hub.load('ultralytics/yolov5', 'custom', path=path, force_reload=True)

def predict(model, input, confidence):
    df = model(input).pandas().xyxy[0]
    df = df.loc[df['class'] == 0]
    df = df.loc[df['confidence'] > confidence]
    df = df.filter(('xmin', 'ymin', 'xmax', 'ymax'))
    return df.to_numpy()
