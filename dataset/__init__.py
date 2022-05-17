import csv
import os.path
import random

from os import PathLike
from typing import List, Dict, Tuple

from torch import Tensor

from torch.utils import data
from torchvision.io import read_image
from torchvision.transforms import Resize, Grayscale
from torchvision.transforms.functional import crop

from affinity.utils import *

class KorteRaw(data.Dataset):

    """KORTE dataset"""
    # Constants
    DESCRIPTOR_FILENAME = 'body_pixel_locations.csv'

    def __init__(self, dataset_dir: PathLike, grid_size: int,  size: Tuple[int, int], device = None) -> None:
        self.grid_size = grid_size
        self.data_dir: PathLike = os.path.join(dataset_dir, 'data')
        self.images: List[str] = []
        self.labels: Dict[str, Dict[str, Dict[str, Tuple[int, int]]]] = dict()
        self.resize = Resize(size)
        self.device = device
        self.grayscale = Grayscale()

        self.__load_descriptor(os.path.join(dataset_dir, 'labels', self.DESCRIPTOR_FILENAME))

    def __load_descriptor(self, descriptor_path: PathLike) -> None:
        """Opens the csv descriptor having the specified path and loads image filenames and labels"""

        with open(descriptor_path, encoding='utf-8') as csv_file:
            reader = csv.reader(csv_file)
            for line in reader:
                image_name = os.path.join(self.data_dir, line[4])

                if not os.path.exists(image_name):
                    continue
                
                if image_name not in self.images:
                    self.images.append(image_name)
                    self.labels[image_name] = {}
                
                w1, h1 = line[-2], line[-1]
                w2, h2 = self.resize.size
                wf, hf = int(w2)/int(w1), int(h2)/int(h1)
                
                if line[0] not in self.labels[image_name]:
                    self.labels[image_name][line[0]] = {}
                
                self.labels[image_name][line[0]][line[1]] = (
                    int(int(line[2]) * wf),
                    int(int(line[3]) * hf),
                )

    def __getitem__(self, index: int) -> Tuple[Tensor, Tuple[int, int, int]]:
        filename = self.images[index]
        raw_labels = self.labels[filename]
        image = self.resize(read_image(filename).to(self.device))/255
        cols, rows = grid(self.grid_size, image.shape)

        labels = []

        for c in range(cols):
            labels.append([])
            for r in range(rows):
                labels[-1].append([random.random(),random.random(),random.random(),random.random(), 0.0])

        for _, parts in raw_labels.items():
            if "Head" in parts.keys() and "Shoulder" in parts.keys():
                c, r, xh, yh = position(self.grid_size, image.shape, parts['Head'])
                xs, ys = relposition(self.grid_size, (c, r), parts["Shoulder"])
                labels[r][c] = [
                    xh, yh,
                    xs, ys,
                    1.0
                ]

        return self.grayscale(image), Tensor(labels).to(self.device)

    def __len__(self) -> int:
        return len(self.images)

