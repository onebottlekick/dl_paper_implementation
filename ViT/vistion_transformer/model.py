import numpy as np
import torch
import torch.nn as nn

from configs import *
from modules import *
from utils import *


class ViT(nn.Module):
    def __init__(self, img_channels, img_size, num_classes, patch_size, num_heads, num_layers, mlp_size, embed_dim, dropout):
        super().__init__()
        
        img_shape = (img_channels,) + pair(img_size)
        num_patches = int(np.prod(img_shape)/(patch_size**2*img_channels))
        patch_size = img_size//int(np.sqrt(num_patches))
        token_dim = img_channels*patch_size**2
        
        # self.to_patch = nn.Conv2d(img_channels, token_dim, kernel_size=patch_size, stride=patch_size)
        self.to_patch = nn.Unfold(kernel_size=patch_size, stride=patch_size)
        self.flatten_patches = nn.Flatten(2)
        self.linear_projection = nn.Linear(token_dim, embed_dim)
        token_dim = embed_dim
        self.cls_token = nn.Parameter(torch.zeros(1, 1, token_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, token_dim))
        self.transformer_encoder = TransformerEncoder(num_layers, token_dim, num_heads, mlp_size, dropout)
        self.mlp_head = nn.Linear(token_dim, num_classes)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.normal_(m.bias, std=1e-6)
        nn.init.normal_(self.pos_embedding, std=0.02)
        nn.init.constant_(self.cls_token, 0)
        
    def forward(self, x, mask=None):
        # input x: (batch_size, img_channels, img_size, img_size)
        
        # make patches and flatten
        # (batch_size, token_dim, sqrt(num_patches), sqrt(num_patches))
        x = self.to_patch(x)
        # (batch_size, token_dim, sqrt(num_patches), sqrt(num_patches)) -> (batch_size, token_dim, num_patches) -> (batch_size, num_patches, token_dim)
        # x = self.flatten_patches(x).transpose(1, 2)
        x = x.transpose(1, 2)
        
        # linear projection of flattend patches
        # (batch_size, num_patches, token_dim)
        x = self.linear_projection(x)
        
        # concat cls token
        # (batch_size, num_patches + 1, token_dim)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        
        # patch + position embedding
        # (batch_size, num_patches + 1, token_dim)
        x = x + self.pos_embedding
        
        # transformer encoder
        # (batch_size, num_patches + 1, token_dim)
        x, attentions = self.transformer_encoder(x)
        
        # take cls token out
        x = x[:, 0, :]
        
        # mlp head
        # (batch_size, 1, num_classes)
        x = self.mlp_head(x)
        
        return x, attentions
        
        
if __name__ == '__main__':
    config = b16_config()
    model = build_model(ViT, config).cuda()    
    img = torch.randn((1, config['img_channels']) + pair(config['img_size'])).cuda()
    
    output, _ = model(img)
    assert output.shape == (1, config['num_classes'])
