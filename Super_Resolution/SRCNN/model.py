import torch.nn as nn


def conv2d(in_channels, out_channels, kernel_size):
    layer = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
    return layer
    
    
class SRCNN(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.patch_extraction = conv2d(args.img_channels, args.num_filters[0], args.filter_size[0])
        self.non_linear_mapping = conv2d(args.num_filters[0], args.num_filters[1], args.filter_size[1])
        self.reconstruction = conv2d(args.num_filters[1], args.img_channels, args.filter_size[2])

        self.relu = nn.ReLU(True)
        
        self.upsample = nn.Upsample(scale_factor=args.lr_scale, mode='bicubic')
        
        self.apply(self._init_weights)
        
    def forward(self, x):
        x = self.upsample(x)
        
        x = self.relu(self.patch_extraction(x))
        x = self.relu(self.non_linear_mapping(x))
        x = self.reconstruction(x)
        
        return x
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, mean=0, std=0.001)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.zeros_(m.bias)