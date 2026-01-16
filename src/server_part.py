import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle
import mlflow
import mlflow.tensorflow
import os
import numpy as np
from fastapi import FastAPI, Request, Response

app = FastAPI()

# Setup Device and Learning Mode
learning_mode = os.getenv("LEARNING_MODE", "split").lower()

# Create TensorFlow/Keras model for server (Part B)
def create_server_model():
    """Server-side model for split learning"""
    model = keras.Sequential([
        layers.Input(shape=(26, 26, 32)),  # Output from client's Conv2D(32, 3) on 28x28
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(2),
        layers.Flatten(),
        layers.Dense(10, activation='softmax')
    ], name='ServerModel')
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Initialize model
model = create_server_model()

# MLflow Setup
mlflow.set_tracking_uri("http://mlflow.mlflow.svc.cluster.local:5000")
experiment_name = f"{learning_mode.capitalize()}_Learning_Sim"
mlflow.set_experiment(experiment_name)

active_run = mlflow.start_run(run_name=f"{learning_mode.capitalize()}_Training")

@app.post("/forward_pass")
async def forward_pass(request: Request):
    """
    Endpoint for Split Learning mode.
    Receives activations from client, performs forward/backward pass,
    and returns gradients to client.
    """
    if learning_mode != "split":
        return Response(
            content=f"Error: /forward_pass endpoint is only for split learning mode. Current mode: {learning_mode}",
            status_code=400
        )
    
    try:
        body = await request.body()
        data = pickle.loads(body)
        
        # Convert numpy arrays to TensorFlow tensors
        client_activations = tf.constant(data["activations"], dtype=tf.float32)
        labels = tf.constant(data["labels"], dtype=tf.int64)
        step = data["step"]
        
        # Ensure activations are tracked for gradients
        client_activations = tf.Variable(client_activations, trainable=True)
        
        # Forward pass with gradient tape
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(client_activations)
            outputs = model(client_activations, training=True)
            loss = tf.keras.losses.sparse_categorical_crossentropy(labels, outputs)
            loss = tf.reduce_mean(loss)
        
        # Compute gradients
        gradients_wrt_activations = tape.gradient(loss, client_activations)
        gradients_wrt_model = tape.gradient(loss, model.trainable_variables)
        
        # Apply gradients to model
        model.optimizer.apply_gradients(zip(gradients_wrt_model, model.trainable_variables))
        
        # Log to MLflow
        if step % 100 == 0:
            mlflow.log_metric("loss", float(loss.numpy()), step=step)
            print(f"Step {step} | Loss: {loss.numpy():.4f}")
        
        # Return gradients as numpy array
        grad_numpy = gradients_wrt_activations.numpy()
        
        del tape  # Clean up persistent tape
        
        return Response(
            content=pickle.dumps(grad_numpy),
            media_type="application/octet-stream"
        )
        
    except Exception as e:
        print(f"Error in forward_pass: {e}")
        import traceback
        traceback.print_exc()
        return Response(
            content=f"Internal server error: {str(e)}",
            status_code=500
        )


@app.post("/aggregate_weights")
async def aggregate_weights(request: Request):
    """
    Endpoint for Federated Learning mode.
    Receives model weights from clients and aggregates them.
    """
    if learning_mode != "federated":
        return Response(
            content=f"Error: /aggregate_weights endpoint is only for federated learning mode. Current mode: {learning_mode}",
            status_code=400
        )
    
    try:
        body = await request.body()
        data = pickle.loads(body)
        
        client_weights = data["weights"]
        step = data["step"]
        
        # Convert to TensorFlow format if needed
        if isinstance(client_weights, dict):
            # Set model weights
            model.set_weights([np.array(w) for w in client_weights.values()])
        
        # Log to MLflow
        if step % 100 == 0:
            mlflow.log_metric("aggregation_step", step, step=step)
            print(f"Aggregated weights at step {step}")
        
        return Response(
            content=pickle.dumps({"status": "success"}),
            media_type="application/octet-stream"
        )
        
    except Exception as e:
        print(f"Error in aggregate_weights: {e}")
        import traceback
        traceback.print_exc()
        return Response(
            content=f"Internal server error: {str(e)}",
            status_code=500
        )


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "mode": learning_mode, "framework": "tensorflow"}


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up MLflow run on shutdown"""
    if active_run:
        mlflow.end_run()