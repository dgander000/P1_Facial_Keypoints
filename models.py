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
        self.conv1 = nn.Conv2d(1, 32, 7, 1) # output size = (W-F)/S + 1 = (224-7)/1 + 1 = 218 -- (32, 218, 218)
        self.pool1 = nn.MaxPool2d(2)        # (32, 109, 109)
        self.drop1 = nn.Dropout(p = 0.1)

        self.conv2 = nn.Conv2d(32, 64, 5, 1) # output size = (W-F)/S + 1 = (109-5)/1 + 1 = 105 -- (64, 105, 105)
        self.pool2 = nn.MaxPool2d(2)         # (64, 52, 52)
        self.drop2 = nn.Dropout(p = 0.2)
        
        self.conv3 = nn.Conv2d(64, 128, 5, 1) # output size = (W-F)/S + 1 = (52-5)/1 + 1 = 48 -- (128, 48, 48)
        self.pool3 = nn.MaxPool2d(2)          # (128, 24, 24)
        self.drop3 = nn.Dropout(p = 0.2)

        self.conv4 = nn.Conv2d(128, 256, 3, 1) # output size = (W-F)/S + 1 = (24-3)/1 + 1 = 22 -- (256, 22, 22)
        self.pool4 = nn.MaxPool2d(2)          # (256, 11, 11)
        self.drop4 = nn.Dropout(p = 0.2)

        self.conv5 = nn.Conv2d(256, 512, 1, 1) # output size = (W-F)/S + 1 = (11-1)/1 + 1 = 11 -- (256, 11, 11)
        self.pool5 = nn.MaxPool2d(2)          # (512, 5, 5)
        self.drop5 = nn.Dropout(p = 0.3)

        self.fc1 = nn.Linear(512*5*5, 1024)
        self.drop6 = nn.Dropout(p = 0.4)
        self.fc2 = nn.Linear(1024, 68*2)

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        
        # a modified x, having gone through all the layers of your model, should be returned
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.drop1(x)
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.drop2(x)
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.drop3(x)
        x = self.pool4(F.relu(self.conv4(x)))
        x = self.drop4(x)
        x = self.pool5(F.relu(self.conv5(x)))
        x = self.drop5(x)

        # prep for linear layer
        # this line of code is the equivalent of Flatten in Keras
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.drop6(x)
        x = self.fc2(x)
        # final output
        return x
