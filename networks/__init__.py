import torch.hub

CLASS = 0

def get_yolo(path):
    return torch.hub.load('ultralytics/yolov5', 'custom', path=path, force_reload=True)

def predict(model, input, confidence):
    for prediction in  model(input).pandas().xyxy[0].values.tolist():
        if prediction[5] == CLASS and prediction[4] >= confidence:
            yield (int(prediction[0]), int(prediction[1]), int(prediction[2]), int(prediction[3]))
