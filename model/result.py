from typing import Tuple

class Evaluation:

    def __init__(self, tp, fp, fn) -> None:
        self.tp = tp
        self.fn = fn
        self.fp = fp

    @property
    def tp(self) -> int:
        return self._tp

    @property
    def fn(self) -> int:
        return self._fn

    @property
    def fp(self) -> int:
        return self._fp

    @tp.setter
    def tp(self, tp):
        self._tp = int(tp)

    @fn.setter
    def fn(self, fn):
        self._fn = int(fn)

    @fp.setter
    def fp(self, fp):
        self._fp = int(fp)

    @property
    def accuracy(self) -> float:
        return self.tp / (self.fn + self.fp + self.tp) if (self.fn + self.fp + self.tp) != 0 else 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) != 0 else 0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if  (self.tp + self.fn) != 0 else 0

    @property
    def f1(self) -> float:
        return (2 * self.tp) / (2 * self.tp + self.fp + self.fn) if (2 * self.tp + self.fp + self.fn) != 0 else 0

    def __str__(self) -> str:
        return \
        f"""
        Evaluation ===
        Confusion matrix
        ---
        P\tN <-- Classified as
        {self.tp}\t{self.fn}\tP
        {self.fp}\t0\tN
        ---

        Accuracy:\t{self.accuracy}
        Precision:\t{self.precision}
        Recall:\t{self.recall}
        F1:\t{self.f1}
        """