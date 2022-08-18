# import dataset
# import networks
# import time
# import evaluation


# Evaluation

# dataset = dataset.KorteRaw("KORTE")
# network = networks.get_yolo('networks/yolov5x6.pt')
# start = time.time_ns()
# eval = evaluation.evaluate_box(network, dataset)
# print(f"Eval in time {(time.time_ns() - start) / 1_000_000_000} s")
# print(eval)


import cv2
import numpy as np
import prespective.checkers
import prespective.manual
from visualize import visualize

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

first_points = None

while True:
    # Capture frame-by-frame
    ret, image = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    points = prespective.checkers.find_checkerboard(image)

    if first_points is None:
        first_points = points

    elif first_points is not None and points is not None:
        H = prespective.checkers.get_homography_from_points(points, first_points)
        image = prespective.checkers.apply_homography(image, H)


    # if points is not None:
    #     image = cv2.drawChessboardCorners(image, (7, 10), points, True)

    # Display the resulting frame
    cv2.imshow('frame', image)
    if cv2.waitKey(1) == ord('q'):
        break