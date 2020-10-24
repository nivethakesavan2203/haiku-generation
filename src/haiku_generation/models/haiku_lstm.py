import torch
from torch import nn


class HaikuLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim=128):
        super(HaikuLSTM, self).__init__()
        self.model = nn.LSTM(embedding_dim, hidden_dim)