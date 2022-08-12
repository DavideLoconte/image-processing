import cv2
import numpy as np

_img = None

def points(image):
    winname = "Manual Calibration"
    global _img
    _img = image.copy()
    def callback(event, x, y, flags, params):
        global _img
        if event == cv2.EVENT_LBUTTONDOWN:
            _img = cv2.circle(_img,(x,y),0,(255,0,255),20)
            points.append([x, y])

    points = []
    cv2.namedWindow(winname)
    cv2.setMouseCallback(winname, callback)

    while len(points) < 4:
        cv2.imshow(winname, _img)
        key = cv2.waitKey(20) & 0xFF

    cv2.destroyAllWindows()
    return np.array(points)