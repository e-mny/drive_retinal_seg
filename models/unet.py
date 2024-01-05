import torch
import torch.nn as nn
import torch.nn.functional as F

"""
The network architecture is illustrated in Figure 1. It consists of a contracting
path (left side) and an expansive path (right side). The contracting path follows
the typical architecture of a convolutional network. It consists of the repeated
application of two 3x3 convolutions (unpadded convolutions), each followed by
a rectified linear unit (ReLU) and a 2x2 max pooling operation with stride 2
for downsampling. At each downsampling step we double the number of feature
channels. Every step in the expansive path consists of an upsampling of the
feature map followed by a 2x2 convolution (“up-convolution”) that halves the
number of feature channels, a concatenation with the correspondingly cropped
feature map from the contracting path, and two 3x3 convolutions, each fol-
lowed by a ReLU. The cropping is necessary due to the loss of border pixels in
every convolution. At the final layer a 1x1 convolution is used to map each 64-
component feature vector to the desired number of classes. In total the network
has 23 convolutional layers.
To allow a seamless tiling of the output segmentation map (see Figure 2), it
is important to select the input tile size such that all 2x2 max-pooling operations
are applied to a layer with an even x- and y-size.
"""

# class ConvBlock():
#    [(W−K+2P)/S]+1
#    W = width
#    K = kernel size
#    P = padding
#    S = stride
   
#    572 - 3 + 2 (0) / 1 + 1
   
# def createLHSConvBlock(in_channels: int, out_channels: int):
#     # Inputshape = (3, 572, 572)
#     num_channel, height, width = input_shape
#     nn.Conv2d(num_channel, 64, (3, 3), stride=1, padding=0)
#     relu = nn.ReLU(inplace=False)
#     nn.Conv2d(64, 64, (3, 3), stride=1, padding=0)
#     relu = nn.ReLU(inplace=False)
    
    
# def createMaxPoolLayer():
#     return nn.MaxPool2d((2, 2), stride=2) # Downsampling
    
    

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        
        self.encoder1_1 = nn.Conv2d(3, 64, (3, 3), stride=1, padding=0)
        self.encoder1_2 = nn.Conv2d(64, 64, (3, 3), stride=1, padding=0)
        
        self.encoder2_1 = nn.Conv2d(64, 128, (3, 3), stride=1, padding=0)
        self.encoder2_2 = nn.Conv2d(128, 128, (3, 3), stride=1, padding=0)
        
        self.encoder3_1 = nn.Conv2d(128, 256, (3, 3), stride=1, padding=0)
        self.encoder3_2 = nn.Conv2d(256, 256, (3, 3), stride=1, padding=0)
        
        self.encoder4_1 = nn.Conv2d(256, 512, (3, 3), stride=1, padding=0)
        self.encoder4_2 = nn.Conv2d(512, 512, (3, 3), stride=1, padding=0)
        
        self.decoder1_1 = nn.Conv2d(512, 1024, (3, 3), stride=1, padding=0)
        self.decoder1_2 = nn.Conv2d(1024, 1024, (3, 3), stride=1, padding=0)
        
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d((2, 2), stride=2) # Downsampling
        
    def createLHSBlock1(self, x):
        x = self.encoder1_1(x)
        x = self.relu(x)
        x = self.encoder1_2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x
    
    def createLHSBlock2(self, x):
        x = self.encoder2_1(x)
        x = self.relu(x)
        x = self.encoder2_2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x

    def createLHSBlock3(self, x):
        x = self.encoder3_1(x)
        x = self.relu(x)
        x = self.encoder3_2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x

    def createLHSBlock4(self, x):
        x = self.encoder4_1(x)
        x = self.relu(x)
        x = self.encoder4_2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x
    
    def createRHSBlock1(self, x):
        x = self.decoder1_1(x)
        x = self.relu(x)
        x = self.decoder1_2(x)
        x = self.relu(x)
        return x
        
        
    def forward(self, x):
        
        # LHS (Contracting Side)
        """
        It consists of the repeated application of two 3x3 convolutions (unpadded convolutions), 
        each followed by a rectified linear unit (ReLU) and 
        a 2x2 max pooling operation with stride 2 for downsampling.
        At each downsampling step we double the number of feature channels. 
        """
        
        
        
        # RHS (Expansive Side)
        """ 
        Every step in the expansive path consists of an upsampling of the feature map 
        followed by a 2x2 convolution (“up-convolution”) that halves the number of feature channels, 
        a concatenation with the correspondingly cropped feature map from the contracting path, and 
        two 3x3 convolutions, each followed by a ReLU.
        """
        
        
        # Last Layer
        """
        At the final layer a 1x1 convolution is used to map each 64-component feature vector to the desired number of classes. 
        In total the network has 23 convolutional layers.
        """
        
        
        return x