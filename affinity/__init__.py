from torch import nn

class AffinityNetwork(nn.Module):

    # Convolutional layers
    KERNEL_SIZE = 5
    INPUT_CHANNELS = 3 # RGB
    LEVEL_1_CHANNELS = 6
    LEVEL_2_CHANNELS = 16
    LEVEL_3_CHANNELS = 32

    # Linear levels
    INPUT_LINEAR = LEVEL_3_CHANNELS * (KERNEL_SIZE ** 2)
    # Increase to add features to the linear network
    LINEAR_1 = 120
    LINEAR_2 = 84
    LINEAR_3 = 1 # Output: confidence

    # Pooling size between conv layers
    POOL_SIZE = 2 # Increase to decrease features to the network

    """Small neural network for feature detection in images"""
    def __init__(self):
        super(AffinityNetwork, self).__init__()
        self.conv1 = nn.Conv2d(self.INPUT_CHANNELS, self.LEVEL_1_CHANNELS, self.KERNEL_SIZE)
        self.conv2 = nn.Conv2d(self.LEVEL_1_CHANNELS, self.LEVEL_2_CHANNELS, self.KERNEL_SIZE)
        self.conv3 = nn.Conv2d(self.LEVEL_2_CHANNELS, self.LEVEL_3_CHANNELS, self.KERNEL_SIZE)
        self.linear1 = nn.Linear(self.INPUT_LINEAR, self.LINEAR_1)# 5x5 from image dimension
        self.linear2 = nn.Linear(self.LINEAR_1, self.LINEAR_2)
        self.linear3 = nn.Linear(self.LINEAR_2, self.LINEAR_3)

        self.pool = nn.MaxPool2d((self.POOL_SIZE, self.POOL_SIZE))

    def forward(self, x):
        x = self.pool(nn.ReLU(self.conv1(x)))
        x = self.pool(nn.ReLU(self.conv2(x)))
        x = self.pool(nn.ReLU(self.conv3(x)))

        x = x.flatten()

        x = nn.ReLU(self.linear1(x))
        x = nn.ReLU(self.linear2(x))
        x = self.linear3(x)
        return x
