import torch
import torch.optim as optim
import pickle
import mlflow
import os
from fastapi import FastAPI, Request, Response
from model_def import get_model

app = FastAPI()

# Setup Device, Model, Optimizer
device = "cpu"
learning_mode = os.getenv("LEARNING_MODE", "split").lower()
model = get_model(role="server").to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

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
    
    body = await request.body()
    data = pickle.loads(body)
    
    client_activations = data["activations"].to(device)
    labels = data["labels"].to(device)
    step = data["step"]

    client_activations.requires_grad_(True)

    optimizer.zero_grad()
    outputs = model(client_activations)
    loss = criterion(outputs, labels)

    loss.backward()
    optimizer.step()

    # Log metric to MLflow
    mlflow.log_metric("loss", loss.item(), step=step)

    cut_layer_gradient = client_activations.grad.clone().detach()
    return Response(content=pickle.dumps(cut_layer_gradient), media_type="application/octet-stream")

@app.post("/aggregate_weights")
async def aggregate_weights(request: Request):
    """
    Endpoint for Federated Learning mode.
    Receives model weights from client, performs aggregation (simple update for single client),
    and returns aggregated model to client.
    """
    if learning_mode != "federated":
        return Response(
            content=f"Error: /aggregate_weights endpoint is only for federated learning mode. Current mode: {learning_mode}",
            status_code=400
        )
    
    body = await request.body()
    data = pickle.loads(body)
    
    client_model_state = data["model_state"]
    epoch = data["epoch"]
    client_loss = data["loss"]
    step = data["step"]
    
    # For single client: simply update server model with client model
    # For multiple clients: this would aggregate multiple client models
    model.load_state_dict(client_model_state)
    
    # Log metrics to MLflow
    mlflow.log_metric("loss", client_loss, step=step)
    mlflow.log_metric("epoch", epoch, step=step)
    
    print(f"Aggregated model at epoch {epoch}, step {step}, loss: {client_loss:.4f}")
    
    # Return the aggregated model (in this case, same as client for single client)
    aggregated_state = model.state_dict()
    return Response(content=pickle.dumps(aggregated_state), media_type="application/octet-stream")

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "mode": learning_mode,
        "model_type": type(model).__name__
    }