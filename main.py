import cv2
import matplotlib.pyplot as plt
import numpy as np

import dataset
from affinity import AffinityNetwork
from torch import nn
from torch import optim

korte = dataset.KorteRaw('Korte')
windowed = dataset.KorteWindowed(korte)



full_image = korte[0][0]
plt.imshow(full_image.cpu().permute(1,2,0).numpy())
plt.show()

for i in range(0, len(windowed)):
    image, label = windowed[i]
    if label > 0:
        print ("Trovato! " + str(i))
        plt.imshow(image.cpu().permute(1,2,0).numpy())
        plt.show()
