import numpy as np
import torch
import torch.nn as nn

from modules import TransformerEncoder


class DeiT(nn.Module):
    def __init__(self, img_channels, img_size, num_classes, patch_size, num_heads, num_layers, mlp_size, embed_dim, dropout, token_type='cls+distil'):
        super().__init__()
        self.token_type = token_type
        
        img_shape = (img_channels, img_size, img_size)
        num_patches = int(np.prod(img_shape)/(patch_size**2*img_channels))
        patch_size = img_size//int(np.sqrt(num_patches))
        token_dim = img_channels*patch_size**2
        
        self.to_patch = nn.Conv2d(img_channels, token_dim, kernel_size=patch_size, stride=patch_size)
        self.flatten_patches = nn.Flatten(2)
        self.linear_projection = nn.Linear(token_dim, embed_dim)
        token_dim = embed_dim
        self.cls_token = nn.Parameter(torch.randn(1, 1, token_dim))
        self.distil_token = nn.Parameter(torch.randn(1, 1, token_dim))       
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches+2, token_dim))
        self.transformer_encoder = TransformerEncoder(num_layers, token_dim, num_heads, mlp_size, dropout)
        self.mlp_head = nn.Linear(token_dim, num_classes)
        
    def forward(self, x):
        x = self.to_patch(x)
        x = self.flatten_patches(x).transpose(1, 2)
        x = self.linear_projection(x)        
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x, self.distil_token.expand(x.shape[0], -1, -1)), dim=1)
        x = x + self.pos_embedding
        x, attentions = self.transformer_encoder(x)
        cls, distil = x[:, 0, :], x[:, -1, :]
        cls, distil = self.mlp_head(cls), self.mlp_head(distil)
        
        return cls, distil, attentions
    
    
if __name__ == '__main__':
    from configs import b16_config
    
    config = b16_config()
    model = DeiT(**config).cuda()
    img = torch.randn(1, config['img_channels'], config['img_size'], config['img_size']).cuda()
    
    cls, distil, _ = model(img)

    assert cls.shape == distil.shape == (1, config['num_classes'])