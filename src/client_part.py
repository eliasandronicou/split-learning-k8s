import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pickle
import requests
import os
import boto3
from botocore.exceptions import ClientError
from model_def import get_model

# Setup
device = "cpu"
learning_mode = os.getenv("LEARNING_MODE", "split").lower()
model = get_model(role="client").to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# SeaweedFS S3 configuration
s3_endpoint = os.getenv("S3_ENDPOINT_URL", "http://seaweedfs.mlflow.svc.cluster.local:8333")
s3_access_key = os.getenv("AWS_ACCESS_KEY_ID", "test")
s3_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY", "test")
bucket_name = "mlops-bucket"
s3_key = "datasets/mnist_dataset.pkl"

# Initialize S3 client
s3_client = boto3.client(
    's3',
    endpoint_url=s3_endpoint,
    aws_access_key_id=s3_access_key,
    aws_secret_access_key=s3_secret_key,
    region_name='us-east-1'
)

# Load MNIST dataset from S3 or download
try:
    print(f"Checking if MNIST dataset exists in S3 bucket '{bucket_name}'...")
    s3_client.head_object(Bucket=bucket_name, Key=s3_key)
    
    print("✓ MNIST dataset found in S3! Downloading from cache...")
    
    # Download from S3
    local_s3_path = '/tmp/mnist_from_s3.pkl'
    s3_client.download_file(bucket_name, s3_key, local_s3_path)
    
    # Load the cached dataset
    with open(local_s3_path, 'rb') as f:
        cached_data = pickle.load(f)
    
    train_dataset = cached_data['train']
    test_dataset = cached_data['test']
    
    print(f"✓ Loaded from S3 cache - Train: {len(train_dataset)}, Test: {len(test_dataset)}")
    
except ClientError as e:
    if e.response['Error']['Code'] == '404':
        print("MNIST dataset not found in S3. Downloading from source...")
        
        # Download MNIST from source
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_dataset = datasets.MNIST(
            './data', 
            train=True, 
            download=True, 
            transform=transform
        )
        
        test_dataset = datasets.MNIST(
            './data', 
            train=False, 
            download=True, 
            transform=transform
        )
        
        print(f"✓ Downloaded - Train: {len(train_dataset)}, Test: {len(test_dataset)}")
        
        # Upload to S3 for future use
        print(f"Uploading MNIST dataset to S3 bucket '{bucket_name}'...")
        
        local_upload_path = '/tmp/mnist_to_upload.pkl'
        with open(local_upload_path, 'wb') as f:
            pickle.dump({
                'train': train_dataset,
                'test': test_dataset
            }, f)
        
        s3_client.upload_file(local_upload_path, bucket_name, s3_key)
        print(f"✓ Uploaded to S3: s3://{bucket_name}/{s3_key}")
    else:
        raise

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

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