import numpy as np
import torch
import torch.nn as nn

from modules import *
from utils import *


class ViT(nn.Module):
    def __init__(self, img_channels, img_size, patch_size, num_heads, num_layers, mlp_size, dropout):
        super().__init__()
        
        img_shape = (img_channels,) + pair(img_size)
        num_patches = int(np.sqrt(np.prod(img_shape)/(patch_size**2*img_channels)))
        patch_size = img_size//num_patches
        token_dim = img_channels*patch_size**2
        
        self.to_patch = nn.Conv2d(img_channels, token_dim, kernel_size=patch_size, stride=patch_size)
        self.flatten_patches = nn.Flatten(2)
        self.linear_projection = nn.Linear(token_dim, token_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, token_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches**2 + 1, token_dim))
        self.transformer_encoder = TransformerEncoder(num_layers, token_dim, num_heads, mlp_size, dropout)
        self.mlp_head = MLP(token_dim, mlp_size, dropout)
        
    def forward(self, x, mask=None):
        # make patches and flatten
        x = self.to_patch(x)
        x = self.flatten_patches(x).transpose(1, 2)
        
        # linear projection of flattend patches
        x = self.linear_projection(x)
        
        # concat cls token
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        
        # patch + position embedding
        x = x + self.pos_embedding
        
        # transformer encoder
        x = self.transformer_encoder(x)
        
        # mlp head
        x = self.mlp_head(x)
        
        return x
        
        
if __name__ == '__main__':
    img = torch.randn(1, 3, 224, 224).cuda()
    vit = ViT(3, 224, 16, 12, 12, 3072, 0.1).cuda()
    assert vit(img).shape == (1, 197, 768)