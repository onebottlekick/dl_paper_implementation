import math

import torch.nn as nn


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2, bias=bias)
        
    def forward(self, x):
        return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, num_features, kernel_size, stride, bias, res_scale=1):
        super().__init__()
        
        self.layer = nn.Sequential(
            Conv2d(num_features, num_features, kernel_size, stride, bias=bias),
            nn.ReLU(True),
            Conv2d(num_features, num_features, kernel_size, stride, bias=bias)
        )
        
        self.res_scale = res_scale
        
    def forward(self, x):
        _x = x
        x = self.layer(x)*self.res_scale
        x = x + _x
        
        return x
    
    
class Upsampler(nn.Module):
    def __init__(self, num_features, lr_scale, bias):
        super().__init__()
        
        self.num_features = num_features
        self.lr_scale = lr_scale
        self.bias = bias
        
        self.layer = nn.Sequential(*self._make_layer())
        
    def _make_layer(self):
        blocks = []
        if self.lr_scale&(self.lr_scale - 1) == 0:
            for _ in range(int(math.log(self.lr_scale, 2))):
                blocks.append(Conv2d(self.num_features, 4*self.num_features, 3, bias=self.bias))
                blocks.append(nn.PixelShuffle(2))
                blocks.append(nn.ReLU(True))
                
        elif self.lr_scale == 3:
            blocks.append(Conv2d(self.num_features, 9*self.num_features, 3, bias=self.bias))
            blocks.append(nn.PixelShuffle(3))
            blocks.append(nn.ReLU(True))
            
        return blocks
    
    def forward(self, x):
        x = self.layer(x)
        
        return x