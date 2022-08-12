import csv
import os.path
from os import PathLike

class KorteRaw():
    """KORTE dataset"""
    # Constants
    DESCRIPTOR_FILENAME = 'body_pixel_locations.csv'

    def __init__(self, dataset_dir: PathLike) -> None:
        self.data_dir: PathLike = os.path.join(dataset_dir, 'data')
        self.labels = {}
        self.sizes = {}
        self.images = []
        self.__load_descriptor(os.path.join(dataset_dir, 'labels', self.DESCRIPTOR_FILENAME))

    def __load_descriptor(self, descriptor_path: PathLike) -> None:
        """Opens the csv descriptor having the specified path and loads image filenames and labels"""

        with open(descriptor_path, encoding='utf-8') as csv_file:
            reader = csv.reader(csv_file)
            for line in reader:
                filename = os.path.join(self.data_dir, line[4])
                if not os.path.exists(filename):
                    continue

                person = line[0]
                label = line[1]
                x, y = line[2], line[3]
                w, h = line[5], line[6]

                if filename not in self.labels.keys():
                    self.labels[filename] = {}

                if person not in self.labels[filename]:
                    self.labels[filename][person] = {}

                self.labels[filename][person][label] = (int(x), int(y))
                self.sizes[filename] = (int(w), int(h))
        self.images = [x for x in self.labels.keys()]

    def __getitem__(self, index: int):
        filename = self.images[index]
        labels = []
        for person, parts in self.labels[filename].items():
            x0, y0 = parts.get('Eyes', (-1,-1))
            x1, y1 = parts.get('Head', (-1,-1))
            x2, y2 = parts.get('Shoulder', (-1,-1))
            x3, y3 = parts.get('Torso', (-1,-1))
            labels.append((x0, y0, x1, y1, x2, y2, x3, y3))
        return filename, tuple(labels)

    def __len__(self) -> int:
        return len(self.images)
