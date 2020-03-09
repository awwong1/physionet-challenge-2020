# CNN feature extraction to LSTM + Transformer + FC, classification
import torch
import torch.nn as nn


class FeatureCNN(nn.Module):
    def __init__(self, in_channels=12, num_classes=9, p_dropout=0.1):
        super(FeatureCNN, self).__init__()

        # b, c, l
        # (1, 12, 4000)
        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(in_channels, in_channels, 10),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels, in_channels, 20),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels, 2 * in_channels, 30),
            nn.LeakyReLU(),
            nn.Dropout(p_dropout, inplace=True),
            # Block 2
            nn.Conv1d(2 * in_channels, 2 * in_channels, 30),
            nn.LeakyReLU(),
            nn.Conv1d(2 * in_channels, 2 * in_channels, 60),
            nn.LeakyReLU(),
            nn.Conv1d(2 * in_channels, 3 * in_channels, 90),
            nn.LeakyReLU(),
            nn.Dropout(p_dropout, inplace=True),
            # Block 3
            nn.Conv1d(3 * in_channels, 3 * in_channels, 90),
            nn.LeakyReLU(),
            nn.Conv1d(3 * in_channels, 3 * in_channels, 110),
            nn.LeakyReLU(),
            nn.Conv1d(3 * in_channels, 4 * in_channels, 130),
            nn.LeakyReLU(),
            nn.Dropout(p_dropout, inplace=True),
            # Block 4
            nn.Conv1d(4 * in_channels, 4 * in_channels, 130),
            nn.LeakyReLU(),
            nn.Conv1d(4 * in_channels, 4 * in_channels, 150),
            nn.LeakyReLU(),
            nn.Conv1d(4 * in_channels, 5 * in_channels, 170),
            nn.LeakyReLU(),
            nn.Dropout(p_dropout, inplace=True),
            # Block 5
            nn.Conv1d(5 * in_channels, 5 * in_channels, 170),
            nn.LeakyReLU(),
            nn.Conv1d(5 * in_channels, 5 * in_channels, 190),
            nn.LeakyReLU(),
            nn.Conv1d(5 * in_channels, 6 * in_channels, 210),
            nn.LeakyReLU(),
            nn.Dropout(p_dropout, inplace=True),
            # Block 6
            nn.Conv1d(6 * in_channels, 6 * in_channels, 210),
            nn.LeakyReLU(),
            nn.Conv1d(6 * in_channels, 6 * in_channels, 230),
            nn.LeakyReLU(),
            nn.Conv1d(6 * in_channels, 7 * in_channels, 250),
            nn.LeakyReLU(),
            nn.Dropout(p_dropout, inplace=True),
            # Block 7
            nn.Conv1d(7 * in_channels, 7 * in_channels, 250),
            nn.LeakyReLU(),
            nn.Conv1d(7 * in_channels, 7 * in_channels, 270),
            nn.LeakyReLU(),
            nn.Conv1d(7 * in_channels, 8 * in_channels, 290),
            nn.LeakyReLU(),
            nn.Dropout(p_dropout, inplace=True),
            # Block 8
            nn.Conv1d(8 * in_channels, 8 * in_channels, 290),
            nn.LeakyReLU(),
            nn.Conv1d(8 * in_channels, 8 * in_channels, 310),
            nn.LeakyReLU(),
            nn.Conv1d(8 * in_channels, 9 * in_channels, 330),
            nn.LeakyReLU(),
            nn.Dropout(p_dropout, inplace=True),
        )

        self.lstm = nn.LSTM(
            input_size=9 * in_channels, hidden_size=10 * in_channels, bidirectional=True
        )

        self.transformer = nn.Transformer(
            d_model=10 * in_channels, nhead=in_channels, dropout=p_dropout
        )

        self.classify = nn.Sequential(
            nn.BatchNorm1d(4),
            nn.LeakyReLU(),
            nn.Dropout(p_dropout, inplace=True),
            nn.Flatten(),
            nn.Linear(in_features=4 * in_channels * 10, out_features=num_classes),
        )

    def forward(self, batch):
        x = batch["signal"]
        # x is in shape (batch, seq, features)
        # need to convert into (batch, features, seq)
        x = torch.transpose(x, 1, 2)

        # x: (batch, inp_size, seq_len)
        feat_out = self.features(x)
        feat_out = torch.transpose(
            feat_out, 0, 1
        )  # feat_out: (inp_size, batch, seq_len)
        feat_out = torch.transpose(
            feat_out, 0, 2
        )  # feat_out: (seq_len, batch, inp_size)

        # LSTM layer
        lstm_out, _ = self.lstm(feat_out)
        (seq_len, batch_size, inp_size) = lstm_out.shape
        lstm_out = lstm_out.view(seq_len, batch_size, 2, inp_size // 2)

        # Attention layer
        trans_out = self.transformer(
            lstm_out[:, :, 0, :].view(seq_len, batch_size, inp_size // 2),
            lstm_out[:, :, 1, :].view(seq_len, batch_size, inp_size // 2),
        )

        # Classification layer
        trans_out = torch.transpose(
            trans_out, 0, 1
        )  # trans_out: (batch, seq_len, inp_size)
        out = self.classify(trans_out)

        return out
