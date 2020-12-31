## TODO: define the convolutional neural network architecture

import torch
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
        
        # image size: (224,224,1)
        # First Convolutional layer: 1 input channel(gray scale image),
        # 32 output channel,  kernel size = 5, stride =1 padding=0
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        # Output of this layer: (64,220,220)
        # Max pooling: kernel size = 2, stride = 2
        self.pool1 = nn.MaxPool2d(2, 2)
        # output of max pooling: (32,110,110)
        
        # Second Convolution layer: 32 input channel, 64 Output channel
        # kernel size 3, stride 1, padding 0
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        # Output of this layer: (64,108,108)
        # Max Pooling: kernel size 2, stride =2
        self.pool2 = nn.MaxPool2d(2,2)
        # Output of max pooling: (64,54,54)
        
        
        # Third Convolution layer: 64 input channel, 128 output channel
        # kernel size 3, stride 1, padding 0 
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        # Output of this layer: (128,52,52)
        # Max Pooling: kernel size 2,stride 2
        self.pool3 = nn.MaxPool2d(2,2)
        # Output of max pooling: (128,26,26)
        
        # Fourth Convolution layer: 128 input channel, 256 output channel
        # kernel size 3, stride 1, padding 0 
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3)
        # Output of this layer: (256,24,24)
        # Max Pooling: kernel size 2,stride 2
        self.pool4 = nn.MaxPool2d(2,2)
        # Output of max pooling: (256,12,12)

        # Fifth Convolution layer: 256 input channel, 512 output channel
        # kernel size 1 padding 0 and stride 1
        self.conv5 = nn.Conv2d(256,512,1)
        # Output of this layer: (512,12,12)
        # Max Pooling: kernel size 2 stride 2
        self.pool5 = nn.MaxPool2d(2,2)
        # Output of max pooling: (512,6,6)

        self.fc1 = nn.Linear(512*6*6, 1024)
         # dropout with p=0.4
        self.fc1_drop = nn.Dropout(p=0.4)
        self.fc2 = nn.Linear(1024, 136)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))
        x = self.pool5(F.relu(self.conv5(x)))
        #print("x shape",x.shape)
        x = x.view(x.size(0), -1)        
        # two linear layers with dropout in between
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = self.fc2(x)
        
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
