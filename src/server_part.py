import torch
import torch.optim as optim
import pickle
import mlflow
from fastapi import FastAPI, Request, Response
from model_def import ModelPartB

app = FastAPI()

# Setup Device, Model, Optimizer
device = "cpu"
model = ModelPartB().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

# MLflow Setup
mlflow.set_tracking_uri("http://mlflow.mlflow.svc.cluster.local:5000")
mlflow.set_experiment("Split_Learning_Sim")

active_run = mlflow.start_run(run_name="Continuous_Training")

@app.post("/forward_pass")
async def forward_pass(request: Request):
    body = await request.body()
    data = pickle.loads(body)
    
    client_activations = data["activations"].to(device)
    labels = data["labels"].to(device)
    step = data["step"]  # ✅ Receive step number from client

    client_activations.requires_grad_(True)

    optimizer.zero_grad()
    outputs = model(client_activations)
    loss = criterion(outputs, labels)

    loss.backward()
    optimizer.step()

    # ✅ Log metric to the SINGLE active run
    # This creates the smooth line chart in the UI
    mlflow.log_metric("loss", loss.item(), step=step)

    cut_layer_gradient = client_activations.grad.clone().detach()
    return Response(content=pickle.dumps(cut_layer_gradient), media_type="application/octet-stream")