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

        self.conv1 = nn.Conv2d(1, 32, 4, 1) 
        self.pool1 = nn.MaxPool2d(2)        
        self.drop1 = nn.Dropout(0.1)

        self.conv2 = nn.Conv2d(32, 64, 3, 1)  
        self.pool2 = nn.MaxPool2d(2)         
        self.drop2 = nn.Dropout(0.2)

        self.conv3 = nn.Conv2d(64, 128, 2, 1) 
        self.pool3 = nn.MaxPool2d(2)          
        self.drop3 = nn.Dropout(0.3) 

        self.conv4 = nn.Conv2d(128, 256, 1, 1) 
        self.pool4 = nn.MaxPool2d(2)          
        self.drop4 = nn.Dropout(0.4)

        self.fc1 = nn.Linear(6400, 1024)
        self.drop5 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(1024, 1024)
        self.drop6 = nn.Dropout(0.6)

        # finally, create 68*2 output channels (for the 68 keypoints)
        self.fc3 = nn.Linear(1024, 68*2)

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        
        # a modified x, having gone through all the layers of your model, should be returned
        x = self.drop1(self.pool1(F.elu(self.conv1(x))))
        x = self.drop2(self.pool2(F.elu(self.conv2(x))))
        x = self.drop3(self.pool3(F.elu(self.conv3(x))))
        x = self.drop4(self.pool4(F.elu(self.conv4(x))))

        # prep for linear layer
        # this line of code is the equivalent of Flatten in Keras
        x = x.view(x.size(0), -1)
        x = self.drop5(F.elu(self.fc1(x)))
        x = self.drop6(F.elu(self.fc2(x))) 
        x = self.fc3(x)
        
        # final output
        return x
