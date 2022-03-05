import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, intermediate_channels, identity_downsample=None):
        super(ConvBlock, self).__init__()
        self.identity_downsample = identity_downsample
        self.expansion = 4
        
        self.conv1 = nn.Conv2d(in_channels, intermediate_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(intermediate_channels)
        
        self.conv2 = nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(intermediate_channels)
        
        self.conv3 = nn.Conv2d(intermediate_channels, intermediate_channels*self.expansion, kernel_size=1, stride=1, padding=0, bias=False)
        self.batchnorm3 = nn.BatchNorm2d(intermediate_channels*self.expansion)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        identity = x.clone()
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        
        x = self.relu(self.batchnorm1(self.conv1(x)))
        x = self.relu(self.batchnorm2(self.conv2(x)))
        x = self.batchnorm3(self.conv3(x))
        x = x + identity
        x = self.relu(x)
        return x
    

class ResNet(nn.Module):
    def __init__(self):
        pass
    
    def forward(self):
        pass
    
    def _layer(self):
        pass
        