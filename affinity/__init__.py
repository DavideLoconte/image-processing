from functools import lru_cache
from math import ceil, floor

from torch import eq, nn, permute, reshape
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

        # Output tensor shape
        # [B, R, Co, 5]

    def forward(self, x):
        x = self.slicer(x)
        return x
