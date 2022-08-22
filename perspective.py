"""Module to correct perspective of the image using checkerboards from www.calib.io"""
import pickle
from tabnanny import check

import cv2
import numpy as np

def single_checkerboard_calibration(points, pattern, size):
    """Find homography matrix H and lenght per pixel"""

    if points is None:
        return None, None

    points = get_corners(points, pattern)
    print(points)
    print(pattern)
    distance = abs(points[3][0] - points[1][0]) * (pattern[0]-1) / (pattern[1]-1)
    y0 = points[1][1] - distance
    y2 = points[3][1] - distance

    target = np.array([
        [points[1][0], y0], points[1],
        [points[3][0], y2], points[3]
    ])

    source = np.array([
        [points[0][0], points[0][1]],
        [points[1][0], points[1][1]],
        [points[2][0], points[2][1]],
        [points[3][0], points[3][1]]
    ])

    H, _ = cv2.findHomography(source, target)
    return H, ((size * pattern[1]) / abs(points[0][0] - points[2][0])) / 2 # Homography, mm/px

def get_corners(points, pattern):
    """Return the corners of the checkerboard starting from the points"""
    return points[np.array([0, pattern[0]-1, -pattern[0], -1])]

def find_checkerboard(image, pattern, flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE):
    """Return the checkerboard points of the given pattern from the image"""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    T, result = cv2.findChessboardCorners(image, pattern, flags)
    return np.squeeze(result) if T else None

def apply_homography(image, homography: np.ndarray):
    """Apply homography to a images"""
    return cv2.warpPerspective(image, homography, (image.shape[1], int(image.shape[0])))

def transform_points(points, homography):
    """Apply homography to points and return the transformed ones"""
    points = np.float32(points).reshape(-1,1,2)
    return cv2.perspectiveTransform(points, homography)

def save_homography(filename, homography, pixel_length):
    """Save homography and pixel_length in file for future uses"""
    try:
        with open(filename, 'wb') as f:
            pickle.dump((homography, pixel_length), f)
            return True
    except OSError:
        print("Cannot save homography data")
        return False

def load_homography(filename):
    """Load homography and pixel_length from file"""
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except OSError:
        print("Cannot load homography data")
        return None, None

