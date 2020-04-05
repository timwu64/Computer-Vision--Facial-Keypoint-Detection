## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        ## Parameter definition
        # W - input image (previous layer) width/height
        # F - kernel size/filter size
        # K - out_channels : the number of filters in the convolutional layer
        # S - the stride of the convolution
        # P - the padding
        
        # For any convolutional layer, the output feature maps equation looks like: output_dim = (W-F)/S + 1
        # maxpooling layers, multiple conv layers, fully-connected layers, and dropout to avoid overfitting
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        
        # Convolution Layer 
        # first conv layer: 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        ## output size = (W-F)/S +1 = (224-5)/1 +1 = 220
        # the output Tensor for one image, will have the dimensions: (32, 220, 220)
        # after one pool layer, this becomes (32, 110, 110)
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.norm1 = nn.BatchNorm2d(32)
        
        # second conv layer: 32 inputs, 64 outputs, 5x5 conv
        ## output size = (W-F)/S +1 = (110-3)/1 +1 = 108
        # the output tensor will have dimensions: (64, 108, 108)
        # after another pool layer this becomes (64, 54, 54);
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.norm2 = nn.BatchNorm2d(64)
        
        # third conv layer: 64 inputs, 128 outputs, 3x3 conv
        ## output size = (W-F)/S +1 = (54-3)/1 +1 = 52
        # the output tensor will have dimensions: (128, 52, 52)
        # after another pool layer this becomes (128, 26, 26);
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.norm3 = nn.BatchNorm2d(128)
        
        # forth conv layer: 128 inputs, 256 outputs, 3x3 conv
        ## output size = (W-F)/S +1 = (26-3)/1 +1 = 24
        # the output tensor will have dimensions: (256, 24, 24)
        # after another pool layer this becomes (256, 12, 12);
        self.conv4 = nn.Conv2d(128, 256, 3)
        self.norm4 = nn.BatchNorm2d(256)
        
        # fifty conv layer: 256 inputs, 512 outputs, 1x1 conv
        ## output size = (W-F)/S +1 = (12-3)/1 +1 = 12
        # the output tensor will have dimensions: (512, 12, 12)
        # after another pool layer this becomes (512, 6, 6);
        self.conv5 = nn.Conv2d(256, 512, 1)
        self.norm5 = nn.BatchNorm2d(512)
        
        # Maxpool layer
        # pool with kernel_size=2, stride=2
        self.pool = nn.MaxPool2d(2, 2)
        
        # Drop Out Layer
        self.fc_drop = nn.Dropout(p=0.3)
        
        # Fully Connected Layer
        # 512 outputs * the 5*5 filtered/pooled map size
        self.fc1 = nn.Linear(512*6*6, 1024)        
        # finally, create 136 output channels
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 68*2)
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        # Five conv/relu + pool layers
        x = self.pool(self.norm1(F.relu(self.conv1(x))))
        x = self.pool(self.norm2(F.relu(self.conv2(x))))
        x = self.pool(self.norm3(F.relu(self.conv3(x))))
        x = self.pool(self.norm4(F.relu(self.conv4(x))))
        x = self.pool(self.norm5(F.relu(self.conv5(x))))

        # prep for linear layer
        # this line of code is the equivalent of Flatten in Keras
        x = x.view(x.size(0), -1)
        
        # two linear layers with dropout in between
        x = F.relu(self.fc1(x))
        x = self.fc_drop(x)
        x = F.relu(self.fc2(x))
        x = self.fc_drop(x)
        x = self.fc3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
