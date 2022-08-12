from multiprocessing.sharedctypes import Value
from typing import Tuple

class Box:

    def __init__(self, x, y, w, h):
        self._x = x
        self._y = y
        self._w = w
        self._h = h

    @property
    def center(self) -> Tuple[int, int]:
        return (self.x + self.w) // 2, (self.y + self.h) // 2

    @property
    def size(self) -> Tuple[int, int]:
        return self.w, self.h

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, x: int):
        self._x = int(x)

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, y: int):
        self._y = int(y)

    @property
    def w(self):
        return self._w

    @w.setter
    def w(self, w: int):
        if int(w) < 0:
            raise ValueError("Invalid width size")
        self._w = int(w)

    @property
    def h(self):
        return self._h

    @h.setter
    def h(self, h: int):
        if int(h) < 0:
            raise ValueError("Invalid height size")
        self._h = int(h)

    @property
    def top(self):
        return self._y

    @top.setter
    def top(self, top: int):
        self.y = top

    @property
    def left(self):
        return self._x

    @left.setter
    def left(self, left: int):
        self.x = left

    @property
    def bottom(self):
        return self._y + self._h

    @bottom.setter
    def bottom(self, bottom: int):
        self.h = bottom - self._y

    @property
    def right(self) -> int:
        return self._x + self._w

    @right.setter
    def right(self, left: int):
        self.w = left - self._x

    def __repr__(self) -> str:
        return str((self.left, self.top, self.right, self.bottom))

    def __str__(self) -> str:
        return self.__repr__()