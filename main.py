from torch import tensor
from torch import concat

import matplotlib.pyplot as plt

import dataset
import affinity

korte = dataset.KorteRaw('Korte')
slicer = affinity.AffinitySlicer(size=60)
network = affinity.AffinityNetwork()
imgs = network(korte[0][0])
# for image in imgs:
for x in imgs:
    for y in x:
        plt.imshow(y.cpu().permute(1,2,0).numpy())
        plt.show()
