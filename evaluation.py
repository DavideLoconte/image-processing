import math

import cv2
import tqdm
import model


class Evaluation:
    def __init__(self, tp, fp, fn) -> None:
        self._tp = tp
        self._fp = fp
        self._fn = fn

    @property
    def accuracy(self):
        return self._tp / (self._tp + self._fn + self._fp)

    @property
    def precision(self):
        return self._tp / (self._tp + self._fp)

    @property
    def recall(self):
        return self._tp / (self._tp + self._fn)

    @property
    def f1(self):
        return 2 * self.precision * self.recall / (self.precision + self.recall)

    def __str__(self) -> str:
        return f"""
        P\tN\t<-- Classified as
        {self._tp}\t{self._fn}\t| Positive
        {self._fp}\t0\t| Negative
        \n
        ---
        \n
        Accuracy: {self.accuracy}
        Precision: {self.precision}
        Recall: {self.recall}
        """

    def __repr__(self) -> str:
        return self.__str__()

def area(rect) -> float:
    return abs(rect[0] - rect[2]) * abs(rect[1] - rect[3])

def contains(point, rect):
    if point[0] == -1 or point[1] == -1:
        return 1
    return 1 if (rect[0] < point[0] and point[0] < rect[2] and rect[1] < point[1] and point[1] < rect[3]) else 0

def prediction_contains(label, prediction):
    return contains((label[0], label[1]), prediction) +\
           contains((label[2], label[3]), prediction) +\
           contains((label[4], label[5]), prediction) +\
           contains((label[6], label[7]), prediction) > 2

def evaluate_box(model, dataset, confidence=0.75):
    tp, fp, fn = 0, 0, 0
    for img, labels in tqdm.tqdm(dataset):
        result = evaluate_image(model, img, labels, confidence)
        tp += result[0]
        fp += result[1]
        fn += result[2]
    return Evaluation(tp, fp, fn)

def select_label(label1, label2, prediction):

    if label2 is None:
        return label1
    if label1 is None:
        return label2

    def distance(pt1, pt2):
        if (pt1[0] == -1 or pt2[0] == -1):
            return 0
        return math.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] + pt2[1])**2)

    center = ((prediction[0] + prediction[2]) / 2, (prediction[1] + prediction[3]) / 2)
    eyes1 = (label1[0], label1[1])
    head1 = (label1[2], label1[3])
    shoulder1 = (label1[4], label1[5])
    torso1 = (label1[6], label1[7])
    eyes2 = (label2[0], label2[1])
    head2 = (label2[2], label2[3])
    shoulder2 = (label2[4], label2[5])
    torso2 = (label2[6], label2[7])
    d1 = distance(center, eyes1) + distance(center, head1) + distance(center, torso1) + distance(center, shoulder1)
    d2 = distance(center, eyes2) + distance(center, head2) + distance(center, torso2) + distance(center, shoulder2)
    if d1 > d2:
        return label1
    return label2

def evaluate_image(yolo, image, labels, confidence=0.75):
    image = cv2.imread(image)
    predictions = list(model.predict(yolo, image, confidence))
    predictions.sort(key=area, reverse=True)
    tp, fp, fn = 0,0,0
    labels = set(labels)

    while len(labels) != 0 and len(predictions) != 0:
        prediction = predictions.pop()
        victim = None

        for label in labels:
            if prediction_contains(label, prediction):
                victim = select_label(victim, label, prediction)

        if victim is not None:
            labels.remove(victim)
            tp += 1
            continue
        fp += 1

    fn = len(labels)
    return tp, fp, fn
