import torch.nn as nn
import torch


class ProtC1D(nn.Module):

    def __init__(self,length):
        super().__init__()
        self.protein_sequence = nn.Sequential(
            nn.Conv1d(length, 64,kernel_size=8, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(),
            nn.Conv1d(32, 64,kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(),
        )
        self.numerical = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.mlp = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(64, 1),
        )

    def forward(self, x,y):
        x = self.protein_sequence(x)
        y = self.numerical(y)
        x = torch.cat((x, y), 1)
        x = self.mlp(x)
        return x