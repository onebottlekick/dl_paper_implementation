import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, activation=None, batchnorm=False):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.batchnorm = nn.BatchNorm2d(out_channels) if batchnorm else nn.Identity()
        self.activation = activation if activation is not None else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.activation(x)

        return x
    

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = ConvBlock(in_channels, out_channels, activation=nn.PReLU())
        self.conv2 = ConvBlock(out_channels, out_channels)

    def forward(self, x):
        _x = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + _x

        return x
    
    
class DenseBlock(nn.Module):
    def __init__(self, beta=0.2, in_channels=64, growth_channels=32):
        super().__init__()
        self.beta = beta
        
        self.conv_blocks = nn.ModuleList([ConvBlock(in_channels + i*growth_channels, growth_channels, activation=nn.LeakyReLU(0.2)) if i != 4 \
        else ConvBlock(in_channels + i*growth_channels, in_channels) for i in range(5)])
        
    def forward(self, x):
        _x = x
        for conv_block in self.conv_blocks[:-1]:
            out = conv_block(x)
            x = torch.cat((x, out), dim=1)
        x = self.conv_blocks[-1](x)
                    
        return _x + self.beta*x
    

class RRDB(nn.Module):
    def __init__(self, beta=0.2, num_dense_blocks=3):
        super().__init__()
        self.beta = beta
        
        self.dense_blocks = nn.ModuleList([DenseBlock(self.beta) for _ in range(num_dense_blocks)])
        
    def forward(self, x):
        for dense_block in self.dense_blocks:
            _x = dense_block(x)
        
        return x + self.beta*_x
    

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, lr_scale=2, activation=nn.LeakyReLU(0.2)):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.pixel_shuffle = nn.PixelShuffle(lr_scale)
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.activation(x)

        return x
    
    
if __name__ == '__main__':
    x = torch.randn(1, 64, 256, 256)
    model = RRDB()
    y = model(x)
    print(y.shape)