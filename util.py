from scipy.spatial import distance as dist
import numpy as np
import cv2

def to_grayscale(image):
    """Convert image to grayscale"""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def denoise(image, strength = 10):
    """Denoise image"""
    return cv2.fastNlMeansDenoising(image, h=strength)
