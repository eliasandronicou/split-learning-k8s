import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pickle
import requests
import os
from model_def import get_model

# Setup
device = "cpu"
learning_mode = os.getenv("LEARNING_MODE", "split").lower()
model = get_model(role="client").to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# Fake Data (MNIST)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

SERVER_URL_SPLIT = "http://split-server.mlflow.svc.cluster.local:8000/forward_pass"
SERVER_URL_FEDERATED = "http://split-server.mlflow.svc.cluster.local:8000/aggregate_weights"

def train_split_learning():
    """Training loop for Split Learning mode"""
    model.train()
    global_step = 0
    for epoch in range(1, 4):  # Train for 3 epochs
        print(f"--- [SPLIT LEARNING] Starting Epoch {epoch} ---")
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            activations = model(data)

            # Send activations to server
            payload = {
                "activations": activations.clone().detach(), 
                "labels": target,
                "step": global_step 
            }
            serialized_data = pickle.dumps(payload)

            try:
                response = requests.post(SERVER_URL_SPLIT, data=serialized_data)
                
                if response.status_code != 200:
                    print(f"Error {response.status_code}: {response.content}")
                    continue

                server_grads = pickle.loads(response.content).to(device)
                activations.backward(server_grads)
                optimizer.step()
                
                if global_step % 10 == 0:
                     print(f"Epoch {epoch} | Step {global_step} | Updated")
                
                global_step += 1

            except Exception as e:
                print(f"Communication failed: {e}")

def train_federated_learning():
    """Training loop for Federated Learning mode"""
    model.train()
    global_step = 0
    
    for epoch in range(1, 4):  # Train for 3 epochs
        print(f"--- [FEDERATED LEARNING] Starting Epoch {epoch} ---")
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            # Complete forward and backward pass locally
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            if global_step % 10 == 0:
                print(f"Epoch {epoch} | Step {global_step} | Loss: {loss.item():.4f}")
            
            global_step += 1
        
        # After each epoch, send model weights to server for aggregation
        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch} complete. Average Loss: {avg_loss:.4f}")
        print(f"Sending model weights to server for aggregation...")
        
        try:
            # Send model state dict to server
            payload = {
                "model_state": model.state_dict(),
                "epoch": epoch,
                "loss": avg_loss,
                "step": global_step - 1  # Last step of the epoch
            }
            serialized_data = pickle.dumps(payload)
            
            response = requests.post(SERVER_URL_FEDERATED, data=serialized_data)
            
            if response.status_code != 200:
                print(f"Error {response.status_code}: {response.content}")
                continue
            
            # Receive aggregated model from server
            aggregated_state = pickle.loads(response.content)
            model.load_state_dict(aggregated_state)
            print(f"Received aggregated model from server")
            
        except Exception as e:
            print(f"Communication failed: {e}")

def train():
    """Main training function that selects the appropriate training loop"""
    print(f"Starting training in {learning_mode.upper()} mode...")
    
    if learning_mode == "split":
        train_split_learning()
    elif learning_mode == "federated":
        train_federated_learning()
    else:
        raise ValueError(f"Unknown LEARNING_MODE: {learning_mode}. Use 'split' or 'federated'.")

if __name__ == "__main__":
    train()