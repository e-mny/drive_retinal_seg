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

    
    

class UNet(nn.Module):
    def __init__(self, out_channels):
        super(UNet, self).__init__()
        
        
        self.encoder1_1 = nn.Conv2d(3, 64, (3, 3), stride=1, padding=1)
        self.encoder1_2 = nn.Conv2d(64, 64, (3, 3), stride=1, padding=1)
        
        self.encoder2_1 = nn.Conv2d(64, 128, (3, 3), stride=1, padding=1)
        self.encoder2_2 = nn.Conv2d(128, 128, (3, 3), stride=1, padding=1)
        
        self.encoder3_1 = nn.Conv2d(128, 256, (3, 3), stride=1, padding=1)
        self.encoder3_2 = nn.Conv2d(256, 256, (3, 3), stride=1, padding=1)
        
        self.encoder4_1 = nn.Conv2d(256, 512, (3, 3), stride=1, padding=1)
        self.encoder4_2 = nn.Conv2d(512, 512, (3, 3), stride=1, padding=1)
        
        self.decoder1_1 = nn.Conv2d(512, 1024, (3, 3), stride=1, padding=1)
        self.decoder1_2 = nn.Conv2d(1024, 1024, (3, 3), stride=1, padding=1)
        self.upconv1 = nn.ConvTranspose2d(1024, 512, (2, 2), stride=2)
        
        self.decoder2_1 = nn.Conv2d(1024, 512, (3, 3), stride=1, padding=1)
        self.decoder2_2 = nn.Conv2d(512, 512, (3, 3), stride=1, padding=1)
        self.upconv2 = nn.ConvTranspose2d(512, 256, (2, 2), stride=2)
        
        self.decoder3_1 = nn.Conv2d(512, 256, (3, 3), stride=1, padding=1)
        self.decoder3_2 = nn.Conv2d(256, 256, (3, 3), stride=1, padding=1)
        self.upconv3 = nn.ConvTranspose2d(256, 128, (2, 2), stride=2)
        
        self.decoder4_1 = nn.Conv2d(256, 128, (3, 3), stride=1, padding=1)
        self.decoder4_2 = nn.Conv2d(128, 128, (3, 3), stride=1, padding=1)
        self.upconv4 = nn.ConvTranspose2d(128, 64, (2, 2), stride=2)
        
        self.decoder5_1 = nn.Conv2d(128, 64, (3, 3), stride=1, padding=1)
        self.decoder5_2 = nn.Conv2d(64, 64, (3, 3), stride=1, padding=1)
        self.finalconv = nn.Conv2d(64, out_channels, (1, 1), stride=1, padding=0)
        
        
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d((2, 2), stride=2) # Downsampling
        
    def createLHSBlock1(self, x):
        x = self.encoder1_1(x)
        x = self.relu(x)
        x = self.encoder1_2(x)
        x = self.relu(x)
        return x
    
    def createLHSBlock2(self, x):
        x = self.encoder2_1(x)
        x = self.relu(x)
        x = self.encoder2_2(x)
        x = self.relu(x)
        return x

    def createLHSBlock3(self, x):
        x = self.encoder3_1(x)
        x = self.relu(x)
        x = self.encoder3_2(x)
        x = self.relu(x)
        return x

    def createLHSBlock4(self, x):
        x = self.encoder4_1(x)
        x = self.relu(x)
        x = self.encoder4_2(x)
        x = self.relu(x)
        return x
    
    def createRHSBlock1(self, x): # Lowest layer
        x = self.decoder1_1(x)
        x = self.relu(x)
        x = self.decoder1_2(x)
        x = self.relu(x)
        return x
    
    def createRHSBlock2(self, x):
        x = self.decoder2_1(x)
        x = self.relu(x)
        x = self.decoder2_2(x)
        x = self.relu(x)
        return x
    
    def createRHSBlock3(self, x):
        x = self.decoder3_1(x)
        x = self.relu(x)
        x = self.decoder3_2(x)
        x = self.relu(x)
        return x
    
    def createRHSBlock4(self, x):
        x = self.decoder4_1(x)
        x = self.relu(x)
        x = self.decoder4_2(x)
        x = self.relu(x)
        return x
    
    def createRHSBlock5(self, x):
        x = self.decoder5_1(x)
        x = self.relu(x)
        x = self.decoder5_2(x)
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
        lhs1 = self.createLHSBlock1(x)
        lhsp1 = self.maxpool(lhs1)
        lhs2 = self.createLHSBlock2(lhsp1)
        lhsp2 = self.maxpool(lhs2)
        lhs3 = self.createLHSBlock3(lhsp2)
        lhsp3 = self.maxpool(lhs3)
        lhs4 = self.createLHSBlock4(lhsp3)
        lhsp4 = self.maxpool(lhs4)
        
        
        
        # RHS (Expansive Side)
        """ 
        Every step in the expansive path consists of an upsampling of the feature map 
        followed by a 2x2 convolution (“up-convolution”) that halves the number of feature channels, 
        a concatenation with the correspondingly cropped feature map from the contracting path, and 
        two 3x3 convolutions, each followed by a ReLU.
        """
        rhs1 = self.createRHSBlock1(lhsp4)
        rhsp1 = self.upconv1(rhs1)
        rhsc1 = torch.cat([lhs4, rhsp1], dim=1) # Concat along the channels dimension

        rhs2 = self.createRHSBlock2(rhsc1)
        rhsp2 = self.upconv2(rhs2)
        rhsc2 = torch.cat([lhs3, rhsp2], dim=1) # Concat along the channels dimension

        rhs3 = self.createRHSBlock3(rhsc2)
        rhsp3 = self.upconv3(rhs3)
        rhsc3 = torch.cat([lhs2, rhsp3], dim=1) # Concat along the channels dimension

        rhs4 = self.createRHSBlock4(rhsc3)
        rhsp4 = self.upconv4(rhs4)
        rhsc4 = torch.cat([lhs1, rhsp4], dim=1) # Concat along the channels dimension
        
        rhs5 = self.createRHSBlock5(rhsc4)

        
        # Last Layer
        """
        At the final layer a 1x1 convolution is used to map each 64-component feature vector to the desired number of classes. 
        In total the network has 23 convolutional layers.
        """
        output = self.finalconv(rhs5)
        
        
        return output