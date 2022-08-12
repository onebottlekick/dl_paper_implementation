import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, activation=None):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.activation = activation if activation is not None else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
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
    def __init__(self, beta=0.2):
        super().__init__()
        self.beta = beta
        
        self.conv1 = ConvBlock(64, 32, activation=nn.LeakyReLU(0.2))
        self.conv2 = ConvBlock(64 + 32, 32, activation=nn.LeakyReLU(0.2))
        self.conv3 = ConvBlock(64 + 2*32, 32, activation=nn.LeakyReLU(0.2))
        self.conv4 = ConvBlock(64 + 3*32, 32, activation=nn.LeakyReLU(0.2))
        self.conv5 = ConvBlock(64 + 4*32, 64)
        
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        
        return x + self.beta*x5
    

class RRDB(nn.Module):
    def __init__(self, beta=0.2, num_dense_blocks=3):
        super().__init__()
        self.beta = beta
        
        self.dense_blocks = [DenseBlock(self.beta) for _ in range(num_dense_blocks)]
        
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