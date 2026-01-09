import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pickle
import requests
from model_def import ModelPartA

# Setup
device = "cpu"
model = ModelPartA().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Fake Data (MNIST)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

SERVER_URL = "http://split-server.mlflow.svc.cluster.local:8000/forward_pass"

def train():
    model.train()
    global_step = 0
    for epoch in range(1, 4): # Train for 3 epochs
        print(f"--- Starting Epoch {epoch} ---")
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            activations = model(data)

            # ✅ Add 'step' to payload
            payload = {
                "activations": activations.clone().detach(), 
                "labels": target,
                "step": global_step 
            }
            serialized_data = pickle.dumps(payload)

            try:
                response = requests.post(SERVER_URL, data=serialized_data)
                
                # Simple error check
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

if __name__ == "__main__":
    train()