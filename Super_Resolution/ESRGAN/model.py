import torch
import torch.nn as nn

from configs import args
from modules import ResidualBlock, DenseBlock, RRDB, UpsampleBlock, ConvBlock


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
    

# TODO build Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        pass


if __name__ == '__main__':
    model = ESRGAN(args)
    x = torch.randn(1, args.img_channels, *args.img_size)
    y = model(x)
    
    assert y.shape == x.shape[:2] + (args.img_size[0]*args.lr_scale, args.img_size[1]*args.lr_scale)