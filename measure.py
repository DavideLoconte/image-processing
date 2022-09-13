import cv2
import time
import numpy as np
from perspective import transform_points
import scipy.spatial.distance as dist


def get_distances(image, predictions, homography, distance):
    """Detect social distancing violations"""
    start = time.time_ns()
    points = get_points(predictions)
    if len(points) == 0:
        return np.array([])
    points = transform_points(points, homography)
    end = time.time_ns()
    print(f"Perspective correction time: {(end - start)/1_000_000} ms")
    start = time.time_ns()
    distances = dist.cdist(points, points, metric="euclidean") * float(distance)
    distances = distances - np.diag(np.diag(distances))
    end = time.time_ns()
    print(f"Measurement time: {(end - start)/1_000_000} ms")
    return distances

def get_violations(distances, threshold=1000):
    violations = (distances < threshold).astype(int)
    return violations - np.diag(np.diag(violations))

def get_points(predictions):
    """Transform array of predictions into array of points """
    result = []
    for prediction in predictions:
        x = ((prediction[2] - prediction[0]) / 2) + prediction[0]
        y = prediction[3]
        result.append([x,y])
    return np.asarray(result)
