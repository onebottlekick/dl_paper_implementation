import torch
import torch.nn as nn

from modules import T2T_Module, TransformerEncoder


# TODO comment tensor shapes
class T2T_ViT(nn.Module):
    def __init__(self, img_channels, img_size, num_classes, embed_dim, mlp_size, num_layers, num_heads, dropout=0.1, kernel_sizes=[7, 3, 3], strides=[4, 2, 2], paddings=[2, 1, 1]):
        super().__init__()
        img_shape = (img_channels, img_size, img_size)
        
        self.tokens_to_token = T2T_Module(img_channels, img_shape, mlp_size, dropout, kernel_sizes, strides, paddings)
        
        token_dim, num_patches = self.tokens_to_token.patch_shape
        self.linear_projection = nn.Linear(token_dim, embed_dim)
        token_dim = embed_dim
        self.cls_token = nn.Parameter(torch.randn(1, 1, token_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, token_dim))
        self.transformer_encoder = TransformerEncoder(num_layers, token_dim, num_heads, mlp_size, dropout)
        self.mlp_head = nn.Linear(token_dim, num_classes)
        
    def forward(self, x):
        x = self.tokens_to_token(x)
        x = self.linear_projection(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.pos_embedding
        x, attentions = self.transformer_encoder(x)
        x = x[:, 0, :]
        x = self.mlp_head(x)
        return x, attentions
    
    
if __name__ == '__main__':
    from configs import base_config
    
    config = base_config()
    model = T2T_ViT(**config).cuda()
    x = torch.randn(1, 3, 224, 224).cuda()
    y, _ = model(x)
    assert y.shape == (1, config['num_classes'])