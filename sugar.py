"""Function naming of cv2 is awful. These function simplify the access to some core utils"""
import cv2

def to_grayscale(image):
    """Convert image to grayscale"""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def denoise(image, strength = 10):
    """Denoise image"""
    return cv2.fastNlMeansDenoising(image, h=strength)

def get_webcam(index = 0, width = 320, height = 240):
    """Return the webcam object"""
    cam = cv2.VideoCapture(index)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cam

def read_image(webcam):
    """Return an image from webcam"""
    return webcam.read()[1]
