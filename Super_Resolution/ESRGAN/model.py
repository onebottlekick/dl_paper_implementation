import torch
import torch.nn as nn

from configs import args
from modules import ResidualBlock, DenseBlock, RRDB, UpsampleBlock, ConvBlock
from utils import calc_shape


class ESRGAN(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.init_conv = nn.Conv2d(args.img_channels, 64, kernel_size=3, stride=1, padding=1)
        
        basic_block = ResidualBlock(64, 64) if args.basic_block.lower() == 'residual'\
                      else DenseBlock(args.beta) if args.basic_block.lower() == 'dense'\
                      else RRDB(args.beta, args.num_dense_blocks) if args.basic_block.lower() == 'rrdb' else None
                      
        self.blocks = nn.Sequential(*[basic_block for _ in range(args.num_blocks)])
        self.mid_conv = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upsample1 = UpsampleBlock(64, 256, lr_scale=args.lr_scale//2)
        self.upsample2 = UpsampleBlock(256//((args.lr_scale//2)**2), 256, lr_scale=args.lr_scale//2)
        self.hr_conv = ConvBlock(64, 64, activation=nn.LeakyReLU(0.2))
        self.final_conv = nn.Conv2d(64, args.img_channels, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        x = self.init_conv(x)
        _x = x
        x = self.blocks(x)
        x = self.mid_conv(x)
        x = x + _x
        x = self.upsample1(x)
        x = self.upsample2(x)
        x = self.hr_conv(x)
        x = self.final_conv(x)
        
        return x
    

class Discriminator(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.conv1 = ConvBlock(args.img_channels, 64, kernel_size=3, batchnorm=False, activation=nn.LeakyReLU(0.2))
        self.conv2 = ConvBlock(64, 64, stride=2, activation=nn.LeakyReLU(0.2), batchnorm=True)
        self.conv3 = ConvBlock(64, 128, activation=nn.LeakyReLU(0.2), batchnorm=True)
        self.conv4 = ConvBlock(128, 128, stride=2, activation=nn.LeakyReLU(0.2), batchnorm=True)
        self.conv5 = ConvBlock(128, 256, activation=nn.LeakyReLU(0.2), batchnorm=True)
        self.conv6 = ConvBlock(256, 256, stride=2, activation=nn.LeakyReLU(0.2), batchnorm=True)
        self.conv7 = ConvBlock(256, 512, activation=nn.LeakyReLU(0.2), batchnorm=True)
        self.conv8 = ConvBlock(512, 512, stride=2, activation=nn.LeakyReLU(0.2), batchnorm=True)
        h, w = calc_shape(args.img_size)
        self.fc1 = nn.Linear(512*h*w, 1024)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(1024, 1)
        # self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.leakyrelu(x)
        x = self.fc2(x)
        # x = self.sigmoid(x)
        
        return x


if __name__ == '__main__':
    model = ESRGAN(args)
    x = torch.randn(1, args.img_channels, *args.img_size)
    y = model(x)
    
    assert y.shape == x.shape[:2] + (args.img_size[0]*args.lr_scale, args.img_size[1]*args.lr_scale)