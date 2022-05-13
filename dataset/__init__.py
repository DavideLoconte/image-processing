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

        return image/255.0, tuple(new_labels)

    def __len__(self) -> int:
        return len(self.image_names)
