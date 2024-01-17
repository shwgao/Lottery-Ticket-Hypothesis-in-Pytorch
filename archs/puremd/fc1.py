import torch
import torch.nn as nn

class fc1_large(nn.Module):

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


class fc1(nn.Module):

    def __init__(self, num_classes=3):
        super(fc1, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(10, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        x = self.classifier(x)
        return x


class fc1_mask(nn.Module):

    def __init__(self, num_classes=3):
        super(fc1_mask, self).__init__()
        self.first_layer = nn.Linear(9, 784)
        self.classifier = nn.Sequential(
            # nn.Linear(9, 784),
            nn.ReLU(inplace=True),
            nn.Linear(784, 300),
            nn.ReLU(inplace=True),
            nn.Linear(300, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, num_classes),
        )
        # create a mask for the fist layer, the mask is trainable
        self.mask = nn.Parameter(torch.ones(9))

    def forward(self, x):
        x = self.mask * x
        x = self.first_layer(x)
        x = self.classifier(x)
        return x
