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

source = cv2.imread('source.JPG')
target = cv2.imread('target.JPG')

cv2.resize(source, (1280, 720))
cv2.resize(target, (1280, 720))

H = prespective.checkers.get_homography(source, target)
image = prespective.checkers.apply_homography(source, H)
cv2.imshow('Hello', image)
cv2.waitKey(0)