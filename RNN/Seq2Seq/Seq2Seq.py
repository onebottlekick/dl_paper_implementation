import random

import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, dropout_ratio):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.dropout = nn.Dropout(dropout_ratio)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dropout_ratio)
        
    def forward(self, x):
        embedding = self.dropout(self.embedding(x))
        outputs, (hidden, cell) = self.lstm(embedding)
        return hidden, cell
    
    
class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers, dropout_ratio):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.dropout = nn.Dropout(dropout_ratio)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dropout_ratio)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, hidden, cell):
        x = x.unsqueeze(0)
        
        embedding = self.dropout(self.embedding(x))
        outputs, (hidden, cell) = self.lstm(embedding, (hidden, cell))
        predictions = self.fc(outputs).squeeze(0)
        return predictions, hidden, cell
    

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, source, target, target_vocab_size, teacher_force_ratio=0.5):
        batch_size = source.shape[1]
        target_length = target.shape[0]
        target_vocab_size = target_vocab_size
        
        outputs = torch.zeros(target_length, batch_size, target_vocab_size).to(next(self.parameters()).device)
        
        hidden, cell = self.encoder(source)
        
        x = target[0]
        
        for t in range(1, target_length):
            output, hidden, cell = self.decoder(x, hidden, cell)
            outputs[t] = output
            
            best_guess = output.argmax(1)
            
            x = target[t] if random.random() < teacher_force_ratio else best_guess
            
        return outputs
    

if __name__ == '__main__':
    input_dim = 300
    output_dim = 400
    embedding_size = 256
    hidden_size = 512
    batch_size = 10
    source_length = 15
    target_length = 17

    encoder = Encoder(input_dim, embedding_size, hidden_size, 2, 0.2)
    decoder = Decoder(output_dim, embedding_size, hidden_size, output_dim, 2, 0.2)
    seq2seq = Seq2Seq(encoder, decoder)
    
    source = torch.randint(input_dim, size=(source_length, batch_size))
    target = torch.randint(output_dim, size=(target_length, batch_size))

    y = seq2seq(source, target, output_dim)
    assert y.shape == (target_length, batch_size, output_dim)