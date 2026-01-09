import torch.nn as nn

# Part 1: Bottom Layers (Client) - UNCHANGED
class ModelPartA(nn.Module):
    def __init__(self):
        super(ModelPartA, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv1(x))

# Part 2: Top Layers (Server) - FIXED
class ModelPartB(nn.Module):
    def __init__(self):
        super(ModelPartB, self).__init__()
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2) 
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(9216, 10) 

    def forward(self, x):
        x = self.relu(self.conv2(x))
        x = self.pool(x) 
        x = self.flatten(x)
        return self.fc1(x)