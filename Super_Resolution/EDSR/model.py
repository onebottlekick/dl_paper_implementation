import torch.nn as nn

from modules import Conv2d, ResidualBlock, Upsampler


class EDSR(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.init_layer = Conv2d(args.img_channels, args.num_features, args.kernel_size)
        
        self.residual_layer = nn.Sequential(
            *[ResidualBlock(args.num_features, args.kernel_size, stride=args.stride, bias=args.bias, res_scale=args.residual_scale) for _ in range(args.num_residual_blocks)]
        )
        
        self.upsample = nn.Sequential(
            Upsampler(args.num_features, args.lr_scale, args.bias),
            Conv2d(args.num_features, args.img_channels, args.kernel_size)
        )
        
    def forward(self, x):
        x = self.init_layer(x)
        _x = x
        x = self.residual_layer(x)
        x = x + _x
        x = self.upsample(x)
        
        return x
    
    
class MDSR(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        self.init_layer = Conv2d(args.img_channels, args.num_features, args.kernel_size)
        
        self.preprocess = nn.ModuleList([
            nn.Sequential(
                ResidualBlock(args.num_features, args.preprocess_kernel_size, stride=args.stride, bias=args.bias, res_scale=args.residual_scale),
                ResidualBlock(args.num_features, args.preprocess_kernel_size, stride=args.stride, bias=args.bias, res_scale=args.residual_scale)
            ) for _ in args.lr_scale
        ])
        
        self.residual_layer = nn.Sequential(
            *[ResidualBlock(args.num_features, args.kernel_size, stride=args.stride, bias=args.bias, res_scale=args.residual_scale) for _ in range(args.num_residual_blocks)]
        )
        
        self.upsample = nn.ModuleList([
            Upsampler(args.num_features, scale, args.bias) for scale in args.lr_scale
        ])
        
        self.out_layer = Conv2d(args.num_features, args.img_channels, args.kernel_size)
        
    def forward(self, x, scale_idx):
        x = self.init_layer(x)
        x = self.preprocess[scale_idx](x)
        
        _x = x
        x = self.residual_layer(x)
        x = x + _x
        
        x = self.upsample[scale_idx](x)
        
        x = self.out_layer(x)
        
        return x