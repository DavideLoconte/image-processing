from unittest import TestCase
from unittest.mock import patch, MagicMock

import numpy as np

import util

class TestUtils(TestCase):

    def test_order(self):
        x = np.array(
            [
                [0,0],
                [1,1],
                [1,0],
                [0,1]
            ]
        )
        self.assertTrue(np.array_equal(
            util.order_points(x), np.array([
                [0,0], [1,0], [1,1], [0,1]
            ], dtype='float32')
        ))

    