import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


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

        self.num_directions = int(bidirectional) + 1
        self.hidden_size = hidden_size

        self.lstm1 = nn.LSTM(
            in_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        # output: (seq, batch, num_directions * hidden_size)

        inp_size = hidden_size * self.num_directions
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(inp_size, num_classes)

    def forward(self, batch):
        x = batch["signal"]  # (seq, batch, feature)
        batch_size = x.size(1)
        lens = batch["len"]
        packed_x = pack_padded_sequence(x, lens, enforce_sorted=False)

        packed_out, _ = self.lstm1(packed_x)
        padded_out, _ = pad_packed_sequence(packed_out, total_length=x.size(0))

        # separate out the forward/backward if required
        seq_batch_dir_hidden = padded_out.view(
            x.size(0), batch_size, self.num_directions, self.hidden_size
        )

        # forward is 0, backward is 1
        to_fc = []
        for idx, length in enumerate(lens):
            seq_dir_hidden = seq_batch_dir_hidden[:,idx]
            out_hidden = seq_dir_hidden[length - 1, 0]
            if self.num_directions > 1:
                back_hidden = seq_dir_hidden[0, 1]
                out_hidden = torch.cat((out_hidden, back_hidden))
            to_fc.append(out_hidden)
        to_fc = torch.stack(to_fc)
        out = self.fc(to_fc)
        return out
