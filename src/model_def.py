import torch.nn as nn
import os

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

# Full Model: Combined architecture for Federated Learning
class FullModel(nn.Module):
    def __init__(self):
        super(FullModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(9216, 10)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.flatten(x)
        return self.fc1(x)

# Helper function to get the appropriate model based on configuration
def get_model(role="client"):
    """
    Returns the appropriate model based on LEARNING_MODE environment variable.
    
    Args:
        role: "client" or "server"
    
    Returns:
        Model instance appropriate for the role and learning mode
    """
    learning_mode = os.getenv("LEARNING_MODE", "split").lower()
    
    if learning_mode == "federated":
        # In federated learning, both client and server use the full model
        return FullModel()
    elif learning_mode == "split":
        # In split learning, client uses part A, server uses part B
        if role == "client":
            return ModelPartA()
        else:  # server
            return ModelPartB()
    else:
        raise ValueError(f"Unknown LEARNING_MODE: {learning_mode}. Use 'split' or 'federated'.")