import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class SignalFourierTransformLSTM(nn.Module):
    """Due to pack_padded_sequence, https://github.com/pytorch/pytorch/issues/31422
    only training with one GPU is supported.
    """
    def __init__(
        self,
        in_channels=12,
        num_classes=9,
        num_layers=2,
        dropout=0.1,
        hidden_size=200,
        bidirectional=True,
    ):
        super(SignalFourierTransformLSTM, self).__init__()

        self.bidirectional = bidirectional
        self.lstm_sig = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
        )

        self.lstm_fft = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
        )

        lstm_hidden_size = hidden_size * 2
        if bidirectional:
            lstm_hidden_size *= 2

        self.classify = nn.Sequential(
            nn.BatchNorm1d(lstm_hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(dropout, inplace=True),
            nn.Linear(lstm_hidden_size, num_classes),
        )

    def forward(self, batch):
        sig = batch["signal"]
        fft = batch["fft"]
        sig_lens = batch["signal_len"]
        fft_lens = batch["fft_len"]

        lstm_sig_in = pack_padded_sequence(sig, sig_lens, enforce_sorted=False)
        _, (sig_hidden, _) = self.lstm_sig(lstm_sig_in)
        # out, lens = pad_packed_sequence(packed_out)

        lstm_fft_in = pack_padded_sequence(fft, fft_lens, enforce_sorted=False)
        _, (fft_hidden, _) = self.lstm_sig(lstm_fft_in)

        if self.bidirectional:
            # concat the forward and backward hidden states
            sig_hidden = torch.cat((sig_hidden[-2, :, :], sig_hidden[-1, :, :]), dim=1)
            fft_hidden = torch.cat((fft_hidden[-2, :, :], fft_hidden[-1, :, :]), dim=1)
        else:
            sig_hidden = sig_hidden[-1, :, :]
            fft_hidden = fft_hidden[-1, :, :]

        hidden = torch.cat((sig_hidden, fft_hidden), dim=1)

        out = self.classify(hidden)

        return out
