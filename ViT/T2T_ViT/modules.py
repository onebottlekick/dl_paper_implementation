import numpy as np
import torch
import torch.nn as nn

from utils import get_patch_shape


class MSA(nn.Module):
    def __init__(self, token_dim, num_heads, dropout):
        super().__init__()
        self.n_heads = num_heads
        
        self.fc_q = nn.Linear(token_dim, token_dim)
        self.fc_k = nn.Linear(token_dim, token_dim)
        self.fc_v = nn.Linear(token_dim, token_dim)
        
        self.fc_o = nn.Linear(token_dim, token_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, num_patches, token_dim = x.shape
        head_dim = token_dim // self.n_heads
        
        query = self.fc_q(x).view(batch_size, num_patches, self.n_heads, head_dim).permute(0, 2, 1, 3)
        key = self.fc_k(x).view(batch_size, num_patches, self.n_heads, head_dim).permute(0, 2, 1, 3)
        value = self.fc_v(x).view(batch_size, num_patches, self.n_heads, head_dim).permute(0, 2, 1, 3)
        
        energy = query@key.transpose(-2, -1)/np.sqrt(head_dim)
        
        attention = torch.softmax(energy, dim=-1)
        
        x = self.dropout(energy)@value
        
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, num_patches, token_dim)
        
        x = self.fc_o(x)
        
        return x, attention
    
    
class MLP(nn.Module):
    def __init__(self, token_dim, mlp_size, dropout):
        super().__init__()
        
        self.fc1 = nn.Linear(token_dim, mlp_size)
        self.fc2 = nn.Linear(mlp_size, token_dim)
        
        self.activation = nn.GELU()
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.dropout(self.activation(self.fc1(x)))
        x = self.dropout(self.fc2(x))
        
        return x
    
    
class TransformerEncoderBlock(nn.Module):
    def __init__(self, token_dim, num_heads, mlp_size, dropout):
        super().__init__()
        
        self.msa = MSA(token_dim, num_heads, dropout)
        self.mlp = MLP(token_dim, mlp_size, dropout)
        
        self.dropout = nn.Dropout(dropout)
        
        self.layer_norm1 = nn.LayerNorm(token_dim)
        self.layer_norm2 = nn.LayerNorm(token_dim)
        
    def forward(self, x):
        _x, attention = self.msa(self.layer_norm1(x))
        _x = self.dropout(_x)
        x = x + _x
        
        _x = self.dropout(self.mlp(self.layer_norm2(x)))
        x = x + _x
        
        return x, attention
    
    
class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, token_dim, num_heads, mlp_size, dropout):
        super().__init__()
        
        self.blocks = nn.ModuleList([TransformerEncoderBlock(token_dim, num_heads, mlp_size, dropout) for _ in range(num_layers)])
        
    def forward(self, x):
        attentions = []
        for block in self.blocks:
            x, attention = block(x)
            attentions.append(attention)
            
        return x, attentions


class SS(nn.Module):
    def __init__(self, kernel_size, stride, padding):
        super().__init__()
        
        self.layer = nn.Unfold(kernel_size=kernel_size, stride=stride, padding=padding)
    
    def forward(self, x):
        x = self.layer(x)
        
        return x
    

# TODO make reshape module
class Reshape:
    def __init__(self, batch_size, num_channels, height, width):
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.height = height
        self.width = width
        
    def __call__(self, x):
        x = x.reshape(self.batch_size, self.num_channels, self.height, self.width)
        
        return x
    
    
class T2T_Module(nn.Module):
    def __init__(self, img_channels, img_size, mlp_size, dropout=0.1, kernel_sizes=[7, 3, 3], strides=[4, 2, 2], paddings=[2, 1, 1]):
        super().__init__()
        self.patch_shape = get_patch_shape(img_size, kernel_sizes, strides, paddings)
        
        self.init_soft_split = SS(kernel_sizes[0], strides[0], paddings[0])
        token_dim = img_channels*kernel_sizes[0]**2
        
        self.transformer1 = TransformerEncoderBlock(token_dim, 1, mlp_size//4, dropout)
        
        self.soft_split1 = SS(kernel_sizes[1], strides[1], paddings[1])
        token_dim = token_dim*kernel_sizes[1]**2
        
        self.transformer2 = TransformerEncoderBlock(token_dim, 1, mlp_size//4, dropout)
        
        self.soft_split2 = SS(kernel_sizes[2], strides[2], paddings[2])
        
    def forward(self, x):
        x = self.init_soft_split(x).transpose(1, 2)
        
        x, _ = self.transformer1(x)
        batch_size, num_patches, token_dim = x.shape
        x = x.transpose(1, 2).reshape(batch_size, token_dim, int(np.sqrt(num_patches)), int(np.sqrt(num_patches)))
        
        x = self.soft_split1(x).transpose(1, 2)
        
        x, _ = self.transformer2(x)
        batch_size, num_patches, token_dim = x.shape
        x = x.transpose(1, 2).reshape(batch_size, token_dim, int(np.sqrt(num_patches)), int(np.sqrt(num_patches)))
        
        x = self.soft_split2(x).transpose(1, 2)
        
        return x
              