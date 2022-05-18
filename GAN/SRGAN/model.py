import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, discriminator=False, use_activation=True, use_batchnorm=True, **kwargs):
        super(ConvBlock, self).__init__()
        self.use_activation = use_activation
        
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs, bias=not use_activation)
        self.batchnorm = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        self.activation = nn.LeakyReLU(0.2, True) if discriminator else nn.PReLU(out_channels)

    def forward(self, x):
        if self.use_activation:
            return self.activation(self.batchnorm(self.conv(x)))
        return self.batchnorm(self.conv(x))


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, scale_factor=2):
        super(UpsampleBlock, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, in_channels*scale_factor**2, kernel_size=3, stride=1, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.activation = nn.PReLU(in_channels)
        
    def forward(self, x):
        return self.activation(self.pixel_shuffle(self.conv(x)))

    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        
        self.module1 = ConvBlock(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.module2 = ConvBlock(in_channels, in_channels, kernel_size=3, stride=1, padding=1, use_activation=False)

    def forward(self, x):
        out = self.module1(x)
        out = self.module2(x)
        
        return out + x
    
    
class Generator(nn.Module):
    def __init__(self, in_channels=3, num_channels=64, num_blocks=16):
        super(Generator, self).__init__()
        
        self.init = ConvBlock(in_channels, num_channels, kernel_size=9, stride=1, padding=4, use_batchnorm=False)
        self.residual = nn.Sequential(*[ResidualBlock(num_channels) for _ in range(num_blocks)])
        self.conv = ConvBlock(num_channels, num_channels, kernel_size=3, stride=1, padding=1, use_activation=False)
        self.upsample = nn.Sequential(*[UpsampleBlock(num_channels) for _ in range(2)])
        self.final_conv = nn.Conv2d(num_channels, in_channels, kernel_size=9, stride=1, padding=4)
        
    def forward(self, x):
        init = self.init(x)
        x = self.residual(init)
        x = self.conv(x) + init
        x = self.upsample(x)
        x = self.final_conv(x)
        x = torch.tanh(x)
        
        return x
    

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 64, 128, 128, 256, 256, 512, 512]):
        super(Discriminator, self).__init__()
        self.in_channels = in_channels
        self.features = features
        
        self.blocks = nn.Sequential(*self._blocks())
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Linear(512*6*6, 1024),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.blocks(x)
        x = self.classifier(x)
        
        return x
        
    def _blocks(self):
        blocks = []
        in_channels = self.in_channels
        for idx, feature in enumerate(self.features):
            blocks.append(ConvBlock(in_channels, feature, kernel_size=3, stride=1+idx%2, padding=1, discriminator=True,use_batchnorm=False if idx == 0 else True))
            in_channels = feature
        
        return blocks


if __name__ == '__main__':
    x = torch.randn(1, 3, 100, 100)
    generator = Generator()
    discriminator = Discriminator()
    
    gen = generator(x)
    assert gen.shape == (1, 3, 400, 400)
    assert discriminator(gen).shape == (1, 1)