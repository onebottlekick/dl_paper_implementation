import torch
import torch.nn as nn

from modules import CascadingBlock, ECascadingBlock, UpsampleBlock


class PCARN(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.init_conv = nn.Conv2d(args.img_channels, 64, kernel_size=3, stride=1, padding=1)
        
        if args.mobile:
            self.block1 = ECascadingBlock(64, 64, args.groups)
            self.block2 = ECascadingBlock(64, 64, args.groups)
            self.block3 = ECascadingBlock(64, 64, args.groups)
        else:
            self.block1 = CascadingBlock(64, 64)
            self.block2 = CascadingBlock(64, 64)
            self.block3 = CascadingBlock(64, 64)
            
        self.conv1 = nn.Conv2d(64*2, 64, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(64*3, 64, kernel_size=1, stride=1)
        self.conv3 = nn.Conv2d(64*4, 64, kernel_size=1, stride=1)
            
        self.up = UpsampleBlock(64, args.scale, args.groups)
        
        self.final_conv = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        x = self.init_conv(x)
        
        b1 = self.block1(x)
        o1 = self.conv1(torch.cat([x, b1], 1))
        
        b2 = self.block2(o1)
        o2 = self.conv2(torch.cat([x, b1, b2], 1))
        
        b3 = self.block3(o2)
        o3 = self.conv3(torch.cat([x, b1, b2, b3], 1))
        
        x = o3 + x
        
        x = self.up(x)
        x = self.final_conv(x)
        
        return x


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--scale', type=int, default=3)
    parser.add_argument('--img_channels', type=int, default=3)
    parser.add_argument('--groups', type=int, default=1)
    parser.add_argument('--mobile', action='store_true')
    
    args = parser.parse_args()
    
    model = PCARN(args)
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print(y.shape)