import torch
from torch import nn

# Define the custom neural network
class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        # Define layers of the neural network
        self.layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
        # Add more layers...
            nn.Linear(128*25*25, 200), # 200 is the number of classes in TinyImageNet
            # nn.ReLU()
        )

    def forward(self, x):
      logits = self.layers(x)
      return logits