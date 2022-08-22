import os
import cv2

def get_directory(path: str):
    for filename in os.listdir(path):
        yield get_image(filename)

def get_image(path: str):
    yield cv2.imread(path)

__cap = None
def get_camera(index: int):
    global __cap
    if __cap is None:
        __cap = (index, cv2.VideoCapture(index))
    elif __cap[0] != index or not __cap[1].cap.isOpened():
        __cap[1].release()
        __cap = (index, cv2.VideoCapture(index))

    if not __cap[1].isOpened():
        raise OSError("Cannot open video capture")

    while True:
        ret, frame = __cap[1].read()
        if not ret:
            break
        yield frame

    __cap[1].release()
    __cap = None


def get_video(path: str):
    video = cv2.VideoCapture(path)

    if not video.isOpened():
        raise OSError("Cannot open video capture")

    while True:
        ret, frame = video.read()
        if not ret:
            break
        yield frame

    video.release()
