import cv2

def annotate_label(image,
                   label,
                   eyes_color=(200, 0, 0),
                   head_color=(0,0,200),
                   shoulder_color=(0,200,0),
                   torso_color=(0,200,200)):

    def mark_point(image, x, y, color, thickness=5):
        shape = image.shape
        if x == -1 or y == -1:
            return image
        return cv2.rectangle(image, (x-thickness, y-thickness), (x+thickness, y+thickness), color, thickness)

    x0, y0, x1, y1, x2, y2, x3, y3  = label
    image = mark_point(image, x0, y0, (200,0,0))
    image = mark_point(image, x1, y1, (0,0,200))
    image = mark_point(image, x2, y2, (0,200,0))
    image = mark_point(image, x3, y3, (0,200,200))
    return image

def annotate_prediction(image, prediction):
    x0, y0, x1, y1 = int(prediction[0]), int(prediction[1]), int(prediction[2]), int(prediction[3])
    return cv2.rectangle(image, (x0, y0), (x1, y1), (255,0,255), 5)

def visualize(image, labels = None, predictions = None, checkers = None):
    if labels is not None:
        for label in labels:
            image = annotate_label(image, label)
    if predictions is not None:
        for prediction in predictions:
            image = annotate_prediction(image, prediction)
    if checkers is not None:
            if len(checkers) == 4:
                image = cv2.rectangle(image, (int(checkers[0][0]), int(checkers[0][1])), (int(checkers[-1][0]), int(checkers[-1][1])), (255,0,0), 2)
            else:
                image = cv2.drawChessboardCorners(image, (7, 10), checkers, True)

    cv2.imshow("Results", image)
    cv2.waitKey(0)

