import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class SimpleLSTM(nn.Module):
    """Simple recurrent neural network classifier using LSTM.
    Supports variable length sequences.
    """

    def __init__(
        self,
        in_channels=12,
        hidden_size=200,
        num_layers=2,
        dropout=0.02,
        bidirectional=True,
        num_classes=9,
    ):
        super(SimpleLSTM, self).__init__()

        self.lstm1 = nn.LSTM(
            in_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,  # input tensors as (batch, seq, feature)
            bidirectional=bidirectional,
        )

        inp_size = hidden_size * num_layers
        if bidirectional:
            inp_size *= 2
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(inp_size, num_classes)

    def forward(self, batch):
        x = batch["signal"]  # (seq, batch, feature)
        lens = batch["len"]
        packed_x = pack_padded_sequence(x, lens, enforce_sorted=False)

        _packed_out, (h_n, _c_n) = self.lstm1(packed_x)
        # h_n = (num_layers * num_directions, batch, hidden_size)
        h_n = torch.transpose(h_n, 0, 1)
        h_n = self.flatten(h_n)
        out = self.fc(h_n)

        return out
