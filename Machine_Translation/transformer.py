import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, n_heads, dropout, device):
        super(MultiHeadAttention, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim//n_heads
        
        self.fc_q = nn.Linear(hidden_dim, hidden_dim)
        self.fc_k = nn.Linear(hidden_dim, hidden_dim)
        self.fc_v = nn.Linear(hidden_dim, hidden_dim)

        self.fc_o = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.tensor([self.head_dim], dtype=torch.float32, device=device))
        
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2))/self.scale
        
        if mask is not None:
            energy = energy.masked_fill(mask==0, -1e10)
        
        attention = torch.softmax(energy, dim=-1)
        
        x = torch.matmul(self.dropout(attention), V)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.hidden_dim)
        x = self.fc_o(x)
        
        return x, attention
    
    
class PositionWiseFeedForward(nn.Module):
    def __init__(self, hidden_dim, ff_dim, dropout):
        super(PositionWiseFeedForward, self).__init__()
        
        self.fc_1 = nn.Linear(hidden_dim, ff_dim)
        self.fc_2 = nn.Linear(ff_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.dropout(self.relu(self.fc_1(x)))
        x = self.fc_2(x)
        
        return x
    

class EncoderLayer(nn.Module):
    def __init__(self, hidden_dim, n_heads, ff_dim, dropout, device):
        super(EncoderLayer, self).__init__()
        
        self.self_attn_layer_norm = nn.LayerNorm(hidden_dim)
        self.ff_layer_norm = nn.LayerNorm(hidden_dim)
        self.self_attention = MultiHeadAttention(hidden_dim, n_heads, dropout, device)
        self.positionwise_ff = PositionWiseFeedForward(hidden_dim, ff_dim, dropout)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, source, source_mask):
        _source, _ = self.self_attention(source, source, source, source_mask)
        source = self.self_attn_layer_norm(source + self.dropout(_source))
        
        _source = self.positionwise_ff(source)
        source = self.ff_layer_norm(source + self.dropout(_source))
        
        return source
    
    
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, n_heads, ff_dim, dropout, device, max_length=100):
        super(Encoder, self).__init__()
        
        self.device = device
        self.token_embedding = nn.Embedding(input_dim, hidden_dim)
        self.position_embedding = nn.Embedding(max_length, hidden_dim)
        
        self.layers = nn.ModuleList([EncoderLayer(hidden_dim, n_heads, ff_dim, dropout, device) for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.tensor([hidden_dim], dtype=torch.float32, device=device))
        
    def forward(self, source, source_mask):
        batch_size = source.shape[0]
        source_length = source.shape[1]
        
        position = torch.arange(0, source_length, device=self.device).unsqueeze(0).repeat(batch_size, 1)

        source = self.dropout((self.token_embedding(source)*self.scale) + self.position_embedding(position))

        for layer in self.layers:
            source = layer(source, source_mask)
        
        return source
    
    
class DecoderLayer(nn.Module):
    def __init__(self, hidden_dim, n_heads, ff_dim, dropout, device):
        super(DecoderLayer, self).__init__()
        
        self.self_attn_layer_norm = nn.LayerNorm(hidden_dim)
        self.encoder_layer_norm = nn.LayerNorm(hidden_dim)
        self.ff_layer_norm = nn.LayerNorm(hidden_dim)
        self.self_attention = MultiHeadAttention(hidden_dim, n_heads, dropout, device)
        self.encoder_attention = MultiHeadAttention(hidden_dim, n_heads, dropout, device)
        self.positionwise_ff = PositionWiseFeedForward(hidden_dim, ff_dim, dropout)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, target, encoded_source, target_mask, source_mask):
        _target, _ = self.self_attention(target, target, target, target_mask)
        target = self.self_attn_layer_norm(target + self.dropout(_target))
        
        _target, attention = self.encoder_attention(target, encoded_source, encoded_source, source_mask)
        target = self.encoder_layer_norm(target + self.dropout(_target))

        _target = self.positionwise_ff(target)
        target = self.ff_layer_norm(target + self.dropout(_target))

        return target, attention
    

class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, n_layers, n_heads, ff_dim, dropout, device, max_length=100):
        super(Decoder, self).__init__()
        
        self.device = device
        
        self.token_embedding = nn.Embedding(output_dim, hidden_dim)
        self.position_embedding = nn.Embedding(max_length, hidden_dim)
        
        self.layers = nn.ModuleList([DecoderLayer(hidden_dim, n_heads, ff_dim, dropout, device) for _ in range(n_layers)])

        self.fc_out = nn.Linear(hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.tensor([hidden_dim], dtype=torch.float32, device=device))

    def forward(self, target, encoded_source, target_mask, source_mask):
        batch_size = target.shape[0]
        target_length = target.shape[1]

        position = torch.arange(0, target_length, device=self.device).unsqueeze(0).repeat(batch_size, 1)

        target = self.dropout((self.token_embedding(target)*self.scale) + self.position_embedding(position))

        for layer in self.layers:
            target, attention = layer(target, encoded_source, target_mask, source_mask)

        output = self.fc_out(target)
        
        return output, attention
    

class Transformer(nn.Module):
    def __init__(self, encoder, decoder, source_pad_idx, target_pad_idx, device):
        super(Transformer, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        
        self.source_pad_idx = source_pad_idx
        self.target_pad_idx = target_pad_idx
        
        self.device = device
        
    def make_source_mask(self, source):
        source_mask = (source != self.source_pad_idx).unsqueeze(1).unsqueeze(2)
        return source_mask
    
    def make_target_mask(self, target):
        target_length = target.shape[1]
        
        target_pad_mask = (target != self.target_pad_idx).unsqueeze(1).unsqueeze(2)
        target_subsequent_mask = torch.tril(torch.ones((target_length, target_length), device=self.device)).bool()
        
        target_mask = target_pad_mask & target_subsequent_mask
        
        return target_mask
    
    def forward(self, source, target):
        source_mask = self.make_source_mask(source)
        target_mask = self.make_target_mask(target)
        
        encoded_source = self.encoder(source, source_mask)
        
        output, attention = self.decoder(target, encoded_source, target_mask, source_mask)
        
        return output, attention
    

if __name__ == '__main__':
    input_dim = 300
    output_dim = 400
    hidden_dim = 256
    n_layers = 3
    n_heads = 8
    ff_dim = 512
    dropout = 0.1

    encoder = Encoder(input_dim, hidden_dim, n_layers, n_heads, ff_dim, dropout, 'cpu')
    decoder = Decoder(output_dim, hidden_dim, n_layers, n_heads, ff_dim, dropout, 'cpu')
    model = Transformer(encoder, decoder, 0, 0, 'cpu').to('cpu')

    batch_size = 10
    source_length = 15
    target_length = 17

    source = torch.randint(input_dim, size=(batch_size, source_length))
    target = torch.randint(output_dim, size=(batch_size, target_length))

    output, attention = model(source, target)
    assert output.shape == (batch_size, target_length, output_dim)