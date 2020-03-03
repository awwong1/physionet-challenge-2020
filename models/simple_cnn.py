import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """Simple convolutional neural network classifier.
    Supports 1 dimensional, multi-channel signals of length 4000.
    """

    def __init__(self, in_channels=12, num_classes=9):
        """Assumes each signal length is 4000"""
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 32, 200),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.AvgPool1d(kernel_size=8),
            nn.Conv1d(32, 64, 100),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.AvgPool1d(kernel_size=7),
            nn.Conv1d(64, 128, 50),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AvgPool1d(kernel_size=3),
            nn.Flatten(),
        )

        self.fc = nn.Linear(128, num_classes)

    def forward(self, batch):
        x = batch["signal"]
        # x is in shape (batch, seq, features)
        # need to convert into (batch, features, seq)
        x = torch.transpose(x, 1, 2)
        output = self.features(x)
        output = self.fc(output)
        return output
