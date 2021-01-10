## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs

        # Input size: 224 x 224
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 11, 2),   # output size = (W-F)/S + 1 = (224-11)/2 + 1 = 107 -- (32, 107, 107)
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),          # (32, 53, 53)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5, 1),  # output size = (W-F)/S + 1 = (53-5)/1 + 1 = 49 -- (64, 49, 49)
            nn.ReLU(), 
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),          # (64, 24, 24)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1),  # output size = (W-F)/S + 1 = (24-3)/1 + 1 = 22 -- (128, 22, 22)
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),           # (128, 11, 11)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1),  # output size = (W-F)/S + 1 = (11-3)/1 + 1 = 9 -- (256, 9, 9)
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),           # (256, 4, 4)
        )

        self.fc1 = nn.Linear(256 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 68*2)

    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.3)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.3)
        x = self.fc3(x)

        return x
