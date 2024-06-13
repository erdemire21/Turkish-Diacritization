from torch.nn import functional as F
import torch.nn as nn
import torch
import numpy as np
from torch.nn import functional as F
class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.scaling_factor = np.sqrt(input_dim // num_heads)
        
        self.query_linear = nn.Linear(input_dim, input_dim)
        self.key_linear = nn.Linear(input_dim, input_dim)
        self.value_linear = nn.Linear(input_dim, input_dim)
        self.out_linear = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        query = self.query_linear(x).view(batch_size, seq_len, self.num_heads, -1)
        key = self.key_linear(x).view(batch_size, seq_len, self.num_heads, -1)
        value = self.value_linear(x).view(batch_size, seq_len, self.num_heads, -1)
        
        # Compute scaled dot product of query with key
        attention_logits = torch.matmul(query, key.transpose(-2, -1)) / self.scaling_factor
        attention_weights = F.softmax(attention_logits, dim=-1)
        
        # Compute weighted sum of value
        attention_output = torch.matmul(attention_weights, value)
        attention_output = attention_output.view(batch_size, seq_len, -1)
        
        return self.out_linear(attention_output)

class DiacritizationModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads =64):
        super(DiacritizationModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.lstm1 = nn.LSTM(embedding_dim, 256, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(512, 128, bidirectional=True, batch_first=True)
        self.lstm3 = nn.LSTM(256, 64, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(128, vocab_size)
        
        self.attention = MultiHeadAttention(128, num_heads)

    def forward(self, x):
        x = self.embedding(x)
        
        x, _ = self.lstm1(x)
        
        x, _ = self.lstm2(x)
        
        x, _ = self.lstm3(x)
        
        # Attention layer
        x = self.attention(x)

        x = self.fc(x)
        
        return x