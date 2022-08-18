import cv2
import numpy as np

def find_checkerboard(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
    T, result = cv2.findChessboardCorners(image, (7, 10), flags)
    return result if T else None

def get_homography_from_points(source, target):
    H, _ = cv2.findHomography(source, target)
    return H

def get_homography(source, target):
    source = find_checkerboard(source)
    target = find_checkerboard(target)

    if source is None or target is None:
        raise ValueError("Cannot find checkerboards in both images")

    return get_homography_from_points(source, target)

def apply_homography(image, homography: np.ndarray):
    return cv2.warpPerspective(image, homography, (image.shape[1], int(image.shape[0])))
