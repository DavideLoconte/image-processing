import sys
import cv2
import os
import numpy as np
import torch
import time

from sugar import get_webcam, read_image, denoise
from coco import COCO_CLASSES, COCO_COLORS

MODEL_PATH = os.path.join('res', 'model.pt')



def main():
    """Entry point"""
    running = True
    # torch.cuda.is_available = lambda : False

    cam = get_webcam(1, width=800, height=600)
    # background = read_image(cam)
    model = torch.hub.load('ultralytics/yolov5', 'yolov5x6')

    while running:
        frame_start = time.time_ns()
        image = read_image(cam)
        input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = model(input_image).xyxy[0].cpu().numpy()

        for detection in result:
            index = int(detection[5])
            name = COCO_CLASSES[index + 1]
            color = COCO_COLORS[index]
            pt1 = (int(detection[0]), int(detection[1]))
            pt2 = (int(detection[2]), int(detection[3]))
            name_xy = (int(detection[0]-10), int(detection[1]-10))
            cv2.rectangle(image, pt1, pt2, color)
            cv2.putText(image, name, name_xy, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            print(f"Detected a {name} in area between {pt1} {pt2} in {(time.time_ns() - frame_start) / 1_000_000} ms")

        cv2.imshow("Webcam", cv2.resize(image, (640,480)))

        if cv2.waitKey(1) == ord('q'):
            running = False

    cam.release()
    cv2.destroyAllWindows()
    return 0

if __name__ == '__main__':
    sys.exit(main())