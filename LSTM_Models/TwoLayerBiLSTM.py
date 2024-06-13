import torch
import torch.nn as nn

class DiacritizationModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(DiacritizationModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.lstm1 = nn.LSTM(embedding_dim, 256, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(512, 128, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(256, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = self.fc(x)
        return x