import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super(Discriminator, self).__init__()

        self.initial = nn.Sequential(
            nn.Conv2d(in_channels*2, features[0], kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(0.2)
        )
        
        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(self._block(in_channels, feature, stride=1 if feature == features[-1] else 2))
            in_channels = feature
            
        layers.append(nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode='reflect'))
            
        self.model = nn.Sequential(*layers)
        
    def forward(self, x, y):
        x = torch.cat((x, y), dim=1)
        x = self.initial(x)
        return self.model(x)
    
    def _block(self, in_channels, out_channels, stride=2):
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, bias=False, padding_mode='reflect'),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )        
        return layer
    
    
class Generator(nn.Module):
    def __init__(self, in_channels=3, features=64):
        super(Generator, self).__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(0.2)
        )
        
        self.down1 = self._block(features, features*2)
        self.down2 = self._block(features*2, features*4)
        self.down3 = self._block(features*4, features*8)
        self.down4 = self._block(features*8, features*8)
        self.down5 = self._block(features*8, features*8)
        self.down6 = self._block(features*8, features*8)
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features*8, features*8, kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
            nn.ReLU()
        )
        
        self.up1 = self._block(features*8, features*8, downsample=False, activation='relu', dropout=0.5)
        self.up2 = self._block(features*8*2, features*8, downsample=False, activation='relu', dropout=0.5)
        self.up3 = self._block(features*8*2, features*8, downsample=False, activation='relu', dropout=0.5)
        self.up4 = self._block(features*8*2, features*8, downsample=False, activation='relu')
        self.up5 = self._block(features*8*2, features*4, downsample=False, activation='relu')
        self.up6 = self._block(features*4*2, features*2, downsample=False, activation='relu')
        self.up7 = self._block(features*2*2, features, downsample=False, activation='relu')
        
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features*2, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
    
    def forward(self, x):
        down1 = self.initial(x)
        down2 = self.down1(down1)
        down3 = self.down2(down2)
        down4 = self.down3(down3)
        down5 = self.down4(down4)
        down6 = self.down5(down5)
        down7 = self.down6(down6)
        
        bottleneck = self.bottleneck(down7)
        
        up1 = self.up1(bottleneck)
        up2 = self.up2(torch.cat((up1, down7), dim=1))
        up3 = self.up3(torch.cat((up2, down6), dim=1))
        up4 = self.up4(torch.cat((up3, down5), dim=1))
        up5 = self.up5(torch.cat((up4, down4), dim=1))
        up6 = self.up6(torch.cat((up5, down3), dim=1))
        up7 = self.up7(torch.cat((up6, down2), dim=1))
        
        return self.final_up(torch.cat((up7, down1), dim=1))
    
    def _block(self, in_channels, out_channels, downsample=True, activation='leaky_relu', dropout=None):
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False, padding_mode='reflect') if downsample
            else nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout) if dropout else nn.Identity(),
            nn.ReLU() if activation == 'relu' else nn.LeakyReLU(0.2)
        )
        return layer
    
    
if __name__ == '__main__':
    x = torch.randn((1, 3, 256, 256))
    y = torch.randn((1, 3, 256, 256))
    discriminator = Discriminator()
    generator = Generator()
    assert discriminator(x, y).shape == (1, 1, 26, 26)
    assert generator(x).shape == x.shape