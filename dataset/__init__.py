import csv
import os.path

from os import PathLike
from pathlib import Path
from typing import List, Dict, Optional, Callable, Tuple, Set
from collections import namedtuple

import pandas as pd
from torch import Tensor

from torch.utils import data
from torchvision.io import read_image
from torchvision.transforms import Resize
from torchvision.transforms.functional import crop

# Types
KorteImage = namedtuple('KorteItem', ('location', 'image_width', 'image_height'))
KorteLabel = namedtuple('KorteLabel', ('x', 'y', 'c'))
PreprocessFunction = Callable[[Tensor], Tensor]
KORTE_CLASSES = ['Eyes','Torso','Shoulder','Head']

class KorteRaw(data.Dataset):

    """KORTE dataset"""
    # Constants
    DESCRIPTOR_FILENAME = 'body_pixel_locations.csv'

    def __init__(self, dataset_dir: PathLike, preprocess: Optional[PreprocessFunction] = None) -> None:
        self.preprocess = preprocess
        self.data_dir: PathLike = os.path.join(dataset_dir, 'data')
        self.image_metadata: Dict[str, KorteImage] = dict()
        self.image_names: List[str] = list()
        self.image_label: Dict[str, Set[KorteLabel]] = dict()
        self.__load_descriptor(os.path.join(dataset_dir, 'labels', self.DESCRIPTOR_FILENAME))
        self.width, self.height = 640, 480
        self.resize = Resize((self.height, self.width))

    def class_index(self, class_name: str) -> int:
        """Return the default index for the output class"""
        return KORTE_CLASSES.index(class_name)

    def class_name(self, class_index: int) -> str:
        """Return the name for the output class index"""
        return KORTE_CLASSES[class_index]

    def __load_descriptor(self, descriptor_path: PathLike) -> None:
        """Opens the csv descriptor having the specified path and loads image filenames and labels"""

        # TODO slow loader, use pandas dataframe to speed up
        # descriptor_df: pd.DataFrame = pd.read_csv(descriptor_path)

        with open(descriptor_path, encoding='utf-8') as csv_file:
            reader = csv.reader(csv_file)
            for line in reader:
                image_name = os.path.join(self.data_dir, line[4])

                if not os.path.exists(image_name):
                    continue

                if image_name not in self.image_metadata:
                    self.image_names.append(image_name)
                    self.image_label[image_name] = set()
                    self.image_metadata[image_name] = KorteImage(
                        location=line[0],
                        image_width=int(line[5]),
                        image_height=int(line[6])
                    )

                self.image_label[image_name].add(KorteLabel(
                    x=int(line[2]),
                    y=int(line[3]),
                    c=self.class_index(line[1])
                ))

    def __getitem__(self, index: int) -> Tuple[Tensor, Tuple[int, int, int]]:
        image_name: str = self.image_names[index]
        image_labels: tuple = tuple(self.image_label[image_name])
        image = read_image(image_name)

        _, h1, w1 = image.shape
        image = self.resize(image)

        if self.preprocess is not None:
            image = self.preprocess(image)

        new_labels: List[KorteLabel] = []
        for label in image_labels:
            new_labels.append(KorteLabel(x=label.x * self.width/w1, y=label.y * self.height/h1, c=label.c))

        return image, tuple(new_labels)

    def __len__(self) -> int:
        return len(self.image_names)


class KorteWindowed(data.Dataset):
    def __init__(self, korte_raw: KorteRaw, sizes = (15, 30, 60, 120, 240), increments = (2, 4, 8, 16, 32), epsilon = 1) -> None:
        super().__init__()
        self.raws = korte_raw
        self.windows = self.__compute_windows(sizes, increments)
        self.traversed_labels = set()
        self.epsilon = epsilon
        self.cache_index = None
        self.cache = None

    def __getitem__(self, index: int):
        image_index = index // len(self.windows)

        if self.cache_index != image_index:
            self.cache_index = image_index
            self.cache = self.raws[image_index]

        image, labels = self.cache

        window = self.windows[index % len(self.windows)]

        size_x = window[2] - window[0]
        size_y = window[3] - window[1]

        center_window = (
            window[0] + size_x/2 - self.epsilon, window[1] + size_y/2 - self.epsilon,
            window[0] + size_x/2 + self.epsilon, window[1] + size_y/2 + self.epsilon,
        )

        cropped_image = crop(image, window[1], window[0], size_y, size_x)

        head = None
        contains_shoulder = False
        for label in labels:
            if self.__is_head(label) and self.__contains(center_window, label):
                head = label
            if self.__is_shoulder(label) and self.__contains(window, label):
                contains_shoulder = True
            if head is not None and contains_shoulder:
                break

        if head is not None and contains_shoulder and (image_index, head) not in self.traversed_labels:
            self.traversed_labels.add((image_index, head))
            return cropped_image, 1.0
        return cropped_image, 0.0

    def __len__(self) -> int:
        return len(self.raws) * len(self.windows)

    def __compute_windows(self, sizes, increments):
        windows = []
        for size, increment in zip(sizes, increments):
            x, y = 0, 0
            while y + size < self.raws.height:
                windows.append((x, y, x + size, y + size))
                x += increment
                if x + size > self.raws.width:
                    x, y = 0, y+increment
        return windows

    def __contains(self, window, point):
        if window[0] < point[0] and point[0] < window[2]:
            if window[1] < point[1] and point[1] < window[3]:
                return True
        return False

    def __is_head(self, label):
        return label[2] == 3

    def __is_shoulder(self, label):
        return label[2] == 2
