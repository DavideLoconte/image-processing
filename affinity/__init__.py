from functools import lru_cache
from math import ceil, floor

from torch import eq, nn, permute, reshape, relu, flatten, sigmoid
from torchvision.transforms import functional as F

class AffinitySlicer(nn.Module):
    
    def __init__(self, size = 60) -> None:
       super().__init__()
       self.size = size

    def forward(self, x):
        x = F.pad(x, self.size_to_padding(x.size))
        x = x.unfold(-2, self.size, self.size).unfold(-2, self.size, self.size)
        image_axis = [-4,-3,-5,-2,-1]
        other_axis = [x for x in range(len(x.shape) - 5)] 
        x = permute(x, other_axis + image_axis)
        return x
    
    def location(self, img, r, c):
        pad = self.size_to_padding(img.size)
        x = (r * self.size) - pad[0]
        y = (c * self.size) - pad[1]
        return x, y 

    @lru_cache(maxsize=8)
    def size_to_padding(self, size):
        return (
            floor((self.size - size(-1) % self.size) / 2) if size(-1) % self.size != 0 else 0,
            floor((self.size - size(-2) % self.size) / 2) if size(-2) % self.size != 0 else 0, # Top
            ceil((self.size - size(-1) % self.size) / 2) if size(-1) % self.size != 0 else 0, # Right
            ceil((self.size - size(-2) % self.size) / 2) if size(-2) % self.size != 0 else 0  # Top
        )
    
    def __hash__(self) -> int:
        return super().__hash__()

    def __eq__(self, __o: object) -> bool:
        return self.size == __o.size
        
        

class AffinityNetwork(nn.Module):

    """Small neural network for feature detection in images"""
    def __init__(self):
        super(AffinityNetwork, self).__init__()
        # Input tensor shape
        # [B, Ch, W, H]
        self.slicer = AffinitySlicer(size=60)
        # Input tensor shape after slicing
        # [B, R, Co, Ch, W, H]
        self.conv1 = nn.Conv2d(3, 8, 10)
        self.conv2 = nn.Conv2d(8, 16, 10)
        self.conv3 = nn.Conv2d(16, 24, 8)
        self.conv4 = nn.Conv2d(24, 32, 4)
        self.conv5 = nn.Conv2d(32, 40, 2)

        self.lin1 = nn.Linear(38440, 38440/2)
        self.lin2 = nn.Linear(38440/2, 38440/4)
        self.lin2 = nn.Linear(38440/2, 6)
        # Output tensor shape
        # [B, R, Co, 5]

    def forward(self, x):

        # Input must be batched. If none, create a batch having only one lement inside
        batched = True
        if len(x.shape) == 3:
            batched = False
            x = x.unsqueeze(0)

        print(x.shape)
        x = self.slicer(x)
        batch_size = x.shape[0]
        rows = x.shape[1]
        columns = x.shape[2]
        
        x = reshape(x, (-1, x.shape[3], x.shape[4], x.shape[5]))
        x = relu(self.conv1(x))
        x = relu(self.conv2(x))
        x = relu(self.conv3(x))
        x = relu(self.conv4(x))
        x = relu(self.conv5(x))
        x = flatten(x, start_dim=1, end_dim=3)
        
        print(x.shape)

        x = relu(self.lin1(x))
        x = relu(self.lin2(x))
        x = sigmoid(self.lin3(x))
    
        # If input was not batched, squeeze vector and get rid of batch
        if not batched:
            x = x.squeeze(0)
        return x
