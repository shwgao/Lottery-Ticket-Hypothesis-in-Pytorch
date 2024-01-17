import torch
import torch.nn as nn

class fc1(nn.Module):

    def __init__(self, num_classes=3):
        super(fc1, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(15, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        x = self.classifier(x)
        return x
