from functools import lru_cache
from math import ceil, floor

from torch import nn, relu, flatten, sigmoid, cat
from torchvision.transforms import functional as F

from affinity.utils import grid


class AffinityNetwork(nn.Module):
    """Small neural network for feature detection in images"""
    def __init__(self, grid_size, image_size):
        super(AffinityNetwork, self).__init__()
        self.size = grid_size
        
        self.w, self.h = image_size
        self.c, self.r = grid(grid_size, (self.h, self.w))
        self.classes = 1
        self.labels = 4

        self.pool = nn.MaxPool2d(2)

        self.conv1a = nn.Conv2d(1, 16, self.size)
        self.conv1b = nn.Conv2d(16, 16, 30)

        self.conv2a = nn.Conv2d(16, 32, 12)
        self.conv2b = nn.Conv2d(32, 32, 6)

        self.conv3a = nn.Conv2d(32, 48, 8)
        self.conv3b = nn.Conv2d(48, 48, 4)
        
        output = 19536
        self.linear1 = nn.Linear(output, self.r * self.c * self.labels)
        self.linear2 = nn.Linear(output, self.r * self.c * self.classes)

        
    def forward(self, x):
        # Input must be batched. If none, create a batch having only one lement inside
        batched = True
        if len(x.shape) == 3:
            batched = False
            x = x.unsqueeze(0)
        batch_size = x.shape[0]
        
        x = relu(self.conv1a(x))
        x = relu(self.conv1b(x))
        x = relu(self.conv2a(x))
        x = relu(self.conv2b(x))
        x = relu(self.conv3a(x))
        x = relu(self.conv3b(x))
        
        x = flatten(x, start_dim=1, end_dim=3)
        x = sigmoid(self.linear2(x)).reshape([batch_size, self.r, self.c, self.classes])

        # If input was not batched, squeeze vector and get rid of batch
        
        if not batched:
            x = x.squeeze(0)
        
        return x


class AffinityLoss(nn.Module):
    
    def __init__(self, grid_size, image_size) -> None:
        super(AffinityLoss, self).__init__()
        self.size = grid_size
        self.w, self.h = image_size
        
        self.ce = nn.BCELoss()
        self.mse = nn.MSELoss()

    def forward(self, y_pred, y_true):
        c_true = y_true[:, :, :, -1].reshape(-1)
        c_pred = y_pred[:, :, :, -1].reshape(-1)
        
        return self.ce(c_true, c_pred)