import math

import torch.nn as nn


class VDSR(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        self.first_layer = nn.Conv2d(self.args.img_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.residual_layer = nn.Sequential(*[self._block() for _ in range(self.args.num_res_blocks)])
        self.last_layer = nn.Conv2d(64, self.args.img_channels, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.relu = nn.ReLU(True)
        
        self.upsample = nn.Upsample(scale_factor=args.lr_scale, mode='bicubic')
        
        self.apply(self._init_weights)
        
    def _block(self):
        block = nn.Sequential(
            nn.Conv2d(self.args.res_channels, self.args.res_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True)
        )
        return block
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0]*m.kernel_size[1]*m.out_channels
            m.weight.data.normal_(0, math.sqrt(2./n))
                
    def forward(self, x):
        x = self.upsample(x)
        _x = x
        x = self.relu(self.first_layer(x))
        x = self.residual_layer(x)
        x = self.last_layer(x)
        x = x + _x
        
        return x