import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle
import requests
import os
import boto3
from botocore.exceptions import ClientError
import numpy as np

# Setup
learning_mode = os.getenv("LEARNING_MODE", "split").lower()

# Create TensorFlow/Keras model for client (Part A)
def create_client_model():
    """Client-side model for split learning"""
    model = keras.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, 3, activation='relu'),
    ], name='ClientModel')
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Create full model for federated learning
def create_full_model():
    """Full model for federated learning"""
    model = keras.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, 3, activation='relu'),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(2),
        layers.Flatten(),
        layers.Dense(10, activation='softmax')
    ], name='FullModel')
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Get appropriate model based on learning mode
if learning_mode == "split":
    model = create_client_model()
else:  # federated
    model = create_full_model()

# SeaweedFS S3 configuration
s3_endpoint = os.getenv("S3_ENDPOINT_URL", "http://seaweedfs.mlflow.svc.cluster.local:8333")
s3_access_key = os.getenv("AWS_ACCESS_KEY_ID", "test")
s3_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY", "test")
bucket_name = "mlops-bucket"
s3_key = "datasets/mnist_dataset_tf.pkl"

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
    
    train_images = cached_data['train_images']
    train_labels = cached_data['train_labels']
    test_images = cached_data['test_images']
    test_labels = cached_data['test_labels']
    
    print(f"✓ Loaded from S3 cache - Train: {len(train_images)}, Test: {len(test_images)}")
    
except ClientError as e:
    if e.response['Error']['Code'] == '404':
        print("MNIST dataset not found in S3. Downloading from source...")
        
        # Download MNIST using Keras
        (train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
        
        # Normalize and reshape for CNN input (28, 28, 1)
        train_images = train_images.astype('float32') / 255.0
        test_images = test_images.astype('float32') / 255.0
        
        # Add channel dimension
        train_images = np.expand_dims(train_images, -1)
        test_images = np.expand_dims(test_images, -1)
        
        # Normalize using MNIST stats (for consistency)
        mean = 0.1307
        std = 0.3081
        train_images = (train_images - mean) / std
        test_images = (test_images - mean) / std
        
        print(f"✓ Downloaded - Train: {len(train_images)}, Test: {len(test_images)}")
        
        # Upload to S3 for future use
        print(f"Uploading MNIST dataset to S3 bucket '{bucket_name}'...")
        
        local_upload_path = '/tmp/mnist_to_upload.pkl'
        with open(local_upload_path, 'wb') as f:
            pickle.dump({
                'train_images': train_images,
                'train_labels': train_labels,
                'test_images': test_images,
                'test_labels': test_labels
            }, f)
        
        s3_client.upload_file(local_upload_path, bucket_name, s3_key)
        print(f"✓ Uploaded to S3: s3://{bucket_name}/{s3_key}")
    else:
        raise

SERVER_URL_SPLIT = "http://split-server.mlflow.svc.cluster.local:8000/forward_pass"
SERVER_URL_FEDERATED = "http://split-server.mlflow.svc.cluster.local:8000/aggregate_weights"

print(f"Starting training in {learning_mode.upper()} mode...")

if learning_mode == "split":
    # Split Learning Training Loop
    batch_size = 64
    num_batches = len(train_images) // batch_size
    
    for epoch in range(1, 4):  # 3 epochs
        print(f"--- Epoch {epoch} ---")
        
        # Shuffle data
        indices = np.random.permutation(len(train_images))
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            batch_indices = indices[start_idx:end_idx]
            
            batch_images = train_images[batch_indices]
            batch_labels = train_labels[batch_indices]
            
            # Forward pass through client model
            with tf.GradientTape() as tape:
                activations = model(batch_images, training=True)
            
            # Send activations to server
            payload = {
                "activations": activations.numpy(),
                "labels": batch_labels,
                "step": epoch * num_batches + batch_idx
            }
            serialized_data = pickle.dumps(payload)
            
            try:
                response = requests.post(
                    SERVER_URL_SPLIT,
                    data=serialized_data,
                    timeout=30
                )
                
                if response.status_code != 200:
                    print(f"Error {response.status_code}: {response.content}")
                    continue
                
                # Receive gradients from server
                server_grads = pickle.loads(response.content)
                server_grads_tensor = tf.constant(server_grads, dtype=tf.float32)
                
                # Backward pass
                gradients = tape.gradient(activations, model.trainable_variables, output_gradients=server_grads_tensor)
                model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                
                if batch_idx % 100 == 0:
                    print(f"Epoch {epoch} | Batch {batch_idx}/{num_batches}")
                
            except Exception as e:
                print(f"Communication failed: {e}")
                continue
        
        print(f"Epoch {epoch} complete")
    
    print("Split learning training complete!")

else:  # Federated Learning
    # Federated Learning Training Loop
    print("Training full model locally...")
    
    history = model.fit(
        train_images,
        train_labels,
        batch_size=64,
        epochs=3,
        validation_split=0.1,
        verbose=1
    )
    
    print("Federated learning training complete!")
    
    # Optionally send weights to server for aggregation
    try:
        weights_dict = {f"layer_{i}": w for i, w in enumerate(model.get_weights())}
        payload = {
            "weights": weights_dict,
            "step": 0
        }
        response = requests.post(
            SERVER_URL_FEDERATED,
            data=pickle.dumps(payload),
            timeout=30
        )
        if response.status_code == 200:
            print("Weights sent to server for aggregation")
    except Exception as e:
        print(f"Could not send weights to server: {e}")

print("Training complete!")