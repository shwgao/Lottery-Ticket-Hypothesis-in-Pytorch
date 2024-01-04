import torch
import torch.nn as nn

class fc1(nn.Module):

    def __init__(self, num_classes=3):
        super(fc1, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(9, 784),
            nn.ReLU(inplace=True),
            nn.Linear(784, 300),
            nn.ReLU(inplace=True),
            nn.Linear(300, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, num_classes),
        )

    def forward(self, x):
        x = self.classifier(x)
        return x
