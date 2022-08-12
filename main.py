import dataset
import networks
import time
import evaluation


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

image = cv2.imread('image.png')

points = prespective.checkers.points(image)
visualize(image, checkers = points)
cv2.imshow("Prova", image)
# image = prespective.four_point_transform(image, points)
# cv2.imshow("prova", image)
cv2.waitKey(0)