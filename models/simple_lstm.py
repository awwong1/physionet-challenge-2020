# -*- coding: utf-8 -*-
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class SimpleLSTM(nn.Module):
    def __init__(self, in_channels=12, hidden_size=10, output_size=9, num_layers=4, dropout=0.05, max_seq_len=2000, bidirectional=True):
        super(SimpleLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_size,
            num_layers=4,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )

        classifier_hidden_size = hidden_size
        if bidirectional:
            classifier_hidden_size = hidden_size * 2
        self.fc1 = nn.Linear(classifier_hidden_size, 1)
        self.flatten = nn.Flatten()
        self.fc2 = nn.Linear(max_seq_len, output_size)


    def forward(self, x_padded, x_lens):
        """
        Args:
            **x_padded** Padded signal inputs (batch_first=True)
            **x_lens** Actual lengths of padded signal inputs
        """
        x_packed = pack_padded_sequence(
            x_padded, x_lens, batch_first=True, enforce_sorted=False)

        output_packed, (_hn, _cn) = self.lstm(x_packed)
        output_padded, output_lengths = pad_packed_sequence(output_packed, batch_first=True)

        output_padded = self.fc1(output_padded)
        output_padded = self.flatten(output_padded)

        # todo, better support for ignoring the padding
        output = self.fc2(output_padded)

        return output
