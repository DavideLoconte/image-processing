import cv2
import numpy as np
import scipy.spatial.distance as dist


def get_distances(image, predictions, homography, distance):
    """Detect social distancing violations"""
    points = get_points(predictions)
    distances = dist.cdist(points, points, metric="euclidean") * distance
    distances = distances - np.diag(np.diag(distances))
    return distances

def get_violations(distances, threshold=1000):
    return (distances > threshold).astype(int)

def get_points(predictions):
    """Transform array of predictions into array of points """
    y = predictions[:, 3]
    x = (predictions[:, 2] - predictions[:, 0]) / 2
    return np.vstack((x, y)).transpose()