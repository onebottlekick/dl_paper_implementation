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
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dropout_ratio, bidirectional=True)
        
        self.fc_hidden = nn.Linear(hidden_size*2, hidden_size)
        self.fc_cell = nn.Linear(hidden_size*2, hidden_size)
        
    def forward(self, x):
        embedding = self.dropout(self.embedding(x))
        encoder_states, (hidden, cell) = self.lstm(embedding)
        
        hidden = self.fc_hidden(torch.cat((hidden[0:1], hidden[1:2]), dim=2))
        cell = self.fc_hidden(torch.cat((cell[0:1], cell[1:2]), dim=2))
        
        return encoder_states, hidden, cell
    
    
class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers, dropout_ratio):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.dropout = nn.Dropout(dropout_ratio)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.lstm = nn.LSTM(hidden_size*2+embedding_size, hidden_size, num_layers, dropout=dropout_ratio)
        
        self.energy = nn.Linear(hidden_size*3, 1)
        self.softmax = nn.Softmax(dim=0)
        self.relu = nn.ReLU()
        
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, encoder_states, hidden, cell):
        x = x.unsqueeze(0)
        
        embedding = self.dropout(self.embedding(x))
        
        sequence_length = encoder_states.shape[0]
        h_reshape = hidden.repeat(sequence_length, 1, 1)
        
        energy = self.relu(self.energy(torch.cat((h_reshape, encoder_states), dim=2)))
        attention = self.softmax(energy)
        attention = attention.permute(1, 2, 0)
        encoder_states = encoder_states.permute(1, 0, 2)
        context_vector = torch.bmm(attention, encoder_states).permute(1, 0, 2)
        
        rnn_input = torch.cat((context_vector, embedding), dim=2)
        
        outputs, (hidden, cell) = self.lstm(rnn_input, (hidden, cell))
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
        
        encoder_states, hidden, cell = self.encoder(source)
        
        x = target[0]
        
        for t in range(1, target_length):
            output, hidden, cell = self.decoder(x, encoder_states, hidden, cell)
            outputs[t] = output
            
            best_guess = output.argmax(1)
            
            x = target[t] if random.random() < teacher_force_ratio else best_guess
            
        return outputs