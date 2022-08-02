import numpy as np
import torch
import torch.nn as nn


class MSA(nn.Module):
    def __init__(self, token_dim, num_heads, dropout):
        super().__init__()
        self.n_heads = num_heads
        
        self.fc_q = nn.Linear(token_dim, token_dim)
        self.fc_k = nn.Linear(token_dim, token_dim)
        self.fc_v = nn.Linear(token_dim, token_dim)
        
        self.fc_o = nn.Linear(token_dim, token_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        batch_size, num_patches, token_dim = x.shape
        head_dim = token_dim//self.n_heads
        
        # num_patches: number of patches(cls_token included)
        # (batch_size, num_patches, n_heads, head_dim) -> (batch_size, n_heads, num_patches, head_dim)
        query = self.fc_q(x).view(batch_size, num_patches, self.n_heads, head_dim).permute(0, 2, 1, 3)
        key = self.fc_k(x).view(batch_size, num_patches, self.n_heads, head_dim).permute(0, 2, 1, 3)
        value = self.fc_v(x).view(batch_size, num_patches, self.n_heads, head_dim).permute(0, 2, 1, 3)
        
        # (batch_size, n_heads, num_patches, head_dim)@(batch_size, n_heads, head_dim, num_patches) -> (batch_size, n_heads, num_patches, num_patches)
        energy = query@key.transpose(-2, -1)/np.sqrt(head_dim)
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        
        # (batch_size, n_heads, num_patches, num_patches)
        attention = torch.softmax(energy, dim=-1)
        
        # (batch_size, n_heads, num_patches, num_patches)@(batch_size, n_heads, num_patches, head_dim) -> (batch_size, n_heads, num_patches, head_dim)
        x = self.dropout(energy)@value
        # (batch_size, num_patches, n_heads, head_dim)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, num_patches, token_dim)
        
        # (batch_size, num_patches, token_dim)
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
        
    def forward(self, x, mask=None):
        _x, attention = self.msa(self.layer_norm1(x), mask)
        _x = self.dropout(_x)
        x = x + _x
        _x = self.dropout(self.mlp(self.layer_norm2(x)))
        x = x + _x
        
        return x, attention
    

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, token_dim, num_heads, mlp_size, dropout):
        super().__init__()
        
        self.blocks = nn.ModuleList([TransformerEncoderBlock(token_dim, num_heads, mlp_size, dropout) for _ in range(num_layers)])
        
    def forward(self, x, mask=None):
        attentions = []
        for block in self.blocks:
            x, attention = block(x, mask)
            attentions.append(attention)
        
        return x, attentions
