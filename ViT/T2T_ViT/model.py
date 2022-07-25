import torch
import torch.nn as nn

from modules import T2T_Module, TransformerEncoder


class T2T_ViT(nn.Module):
    def __init__(self, img_channels, num_classes, embed_dim, mlp_size, num_layers, num_heads, dropout=0.1, kernel_sizes=[7, 3, 3], strides=[4, 2, 2], paddings=[2, 1, 1]):
        super().__init__()
        
        self.tokens_to_token = T2T_Module(img_channels, mlp_size, dropout, kernel_sizes, strides, paddings)
        
        # TODO auto calc
        batch_size, num_patches, token_dim = [1, 196, 11907]
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
    model = T2T_ViT(3, 1000, 1024, 1024, 6, 8, 0.1)
    x = torch.randn(1, 3, 224, 224)
    y, _ = model(x)
    print(y.shape)