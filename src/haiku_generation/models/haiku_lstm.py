import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence


class HaikuLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim=128):
        super(HaikuLSTM, self).__init__()
        self.model = nn.LSTM(embedding_dim, hidden_dim)