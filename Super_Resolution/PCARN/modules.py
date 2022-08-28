import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, activation=None, groups=1):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups)            
        self.activation = activation() if activation else None
        
    def forward(self, x):
        if self.activation:
            x = self.activation(self.conv(x))
        else:
            x = self.conv(x)
        
        return x
    
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv1 = ConvBlock(in_channels, out_channels, activation=nn.ReLU)
        self.conv2 = ConvBlock(out_channels, out_channels)
        self.relu = nn.ReLU(True)
        
    def forward(self, x):
        _x = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + _x
        x = self.relu(x)
        
        return x
    

class CascadingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.res_block1 = ResidualBlock(in_channels, out_channels)
        self.conv1 = nn.Conv2d(out_channels*2, out_channels, kernel_size=1, stride=1)
        
        self.res_block2 = ResidualBlock(out_channels, out_channels)
        self.conv2 = nn.Conv2d(out_channels*3, out_channels, kernel_size=1, stride=1)
        
        self.res_block3 = ResidualBlock(out_channels, out_channels)
        self.conv3 = nn.Conv2d(out_channels*4, out_channels, kernel_size=1, stride=1)
        
    def forward(self, x):
        inp = x
        
        r1 = self.res_block1(x)
        o1 = self.conv1(torch.cat([inp, r1], 1))
        
        r2 = self.res_block2(o1)
        o2 = self.conv2(torch.cat([inp, r1, r2], 1))
        
        r3 = self.res_block3(o2)
        o3 = self.conv3(torch.cat([inp, r1, r2, r3], 1))
        
        return o3
    
    
class EResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, groups):
        super().__init__()
        
        self.conv1 = ConvBlock(in_channels, out_channels, activation=nn.ReLU, groups=groups)
        self.conv2 = ConvBlock(out_channels, out_channels, activation=nn.ReLU, groups=groups)
        self.conv3 = ConvBlock(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(True)
        
    def forward(self, x):
        _x = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x + _x
        x = self.relu(x)
        
        return x
    
    
class ECascadingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, groups):
        super().__init__()
        
        self.res_block = EResidualBlock(in_channels, out_channels, groups)
        self.conv1 = nn.Conv2d(out_channels*2, out_channels, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(out_channels*3, out_channels, kernel_size=1, stride=1)
        self.conv3 = nn.Conv2d(out_channels*4, out_channels, kernel_size=1, stride=1)
        
    def forward(self, x):
        inp = x
        
        r1 = self.res_block(x)
        o1 = self.conv1(torch.cat([inp, r1], 1))
        
        r2 = self.res_block(o1)
        o2 = self.conv2(torch.cat([inp, r1, r2], 1))
        
        r3 = self.res_block(o2)
        o3 = self.conv3(torch.cat([inp, r1, r2, r3], 1))
        
        return o3
    
    
class UpsampleBlock(nn.Module):
    def __init__(self, channels, scale, groups):
        super().__init__()
        self.scale = scale
        
        self.scale2 = nn.Sequential(
            ConvBlock(channels, 4*channels, activation=nn.ReLU, groups=groups),
            nn.PixelShuffle(2)
        )
        
        self.scale3 = nn.Sequential(
            ConvBlock(channels, 9*channels, activation=nn.ReLU, groups=groups),
            nn.PixelShuffle(3)
        )
        
        self.scale4 = nn.Sequential(
            self.scale2,
            self.scale2
        )
                
    def forward(self, x):
        if self.scale == 2:
            return self.scale2(x)
        elif self.scale == 3:
            return self.scale3(x)
        elif self.scale == 4:
            return self.scale4(x)
        else:
            raise NotImplementedError