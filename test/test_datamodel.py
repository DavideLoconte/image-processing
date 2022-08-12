from unittest import TestCase
from unittest.mock import patch, MagicMock

import model.geometry
import model.result

class TestGeometry(TestCase):

    def test_box(self):

        box = model.geometry.Box(0,0,10,10)
        self.assertEqual(box.x, 0)
        self.assertEqual(box.y, 0)
        self.assertEqual(box.w, 10)
        self.assertEqual(box.h, 10)

        self.assertEqual(box.top, 0)
        self.assertEqual(box.left, 0)
        self.assertEqual(box.bottom, 10)
        self.assertEqual(box.right, 10)

        box.left = 10
        box.top = 20

        self.assertEqual(box.x, 10)
        self.assertEqual(box.y, 20)
        self.assertEqual(box.w, 10)
        self.assertEqual(box.h, 10)

        self.assertEqual(box.top, 20)
        self.assertEqual(box.left, 10)
        self.assertEqual(box.bottom, 30)
        self.assertEqual(box.right, 20)

        box.bottom = 25

        self.assertEqual(box.h, 5)
        self.assertEqual(box.w, 10)
        self.assertEqual(box.x, 10)
        self.assertEqual(box.y, 20)

    def test_evaluation(self):
        result = model.result.Evaluation(1, 2, 3)
        self.assertEqual(result.tp, 1)
        self.assertEqual(result.fp, 2)
        self.assertEqual(result.fn, 3)
        self.assertEqual(result.accuracy, 1/6)
        self.assertEqual(result.precision, 1/3)
        self.assertEqual(result.recall, 1/4)
        result.tp = 4
        self.assertEqual(result.tp, 4)
        self.assertEqual(result.accuracy, 4/9)
        self.assertEqual(result.precision, 4/6)
        self.assertEqual(result.recall, 4/7)