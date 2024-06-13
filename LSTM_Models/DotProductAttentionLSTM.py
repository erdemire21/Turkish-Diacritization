from torch.nn import functional as F
import torch.nn as nn
import torch

class Attention(nn.Module):
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        attention_weights = F.softmax(self.linear(x), dim=2)
        return x * attention_weights

class DiacritizationModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(DiacritizationModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.lstm1 = nn.LSTM(embedding_dim, 256, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(512, 128, bidirectional=True, batch_first=True)
        self.lstm3 = nn.LSTM(256, 64, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(128, vocab_size)
        
        self.attention = Attention(128)

    def forward(self, x):
        x = self.embedding(x)
        
        x, _ = self.lstm1(x)
        
        x, _ = self.lstm2(x)
        
        x, _ = self.lstm3(x)
        
        # Attention layer
        x = self.attention(x)
        
        x = self.fc(x)
        
        return x