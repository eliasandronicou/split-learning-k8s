# Split Learning Simulation on Kubernetes

This repository simulates a **Split Learning** architecture using Kubernetes (k3d). It separates a PyTorch Neural Network into two parts across different pods:
1.  **Client Pod:** Holds data & bottom layers. Computes activations.
2.  **Server Pod:** Holds top layers & labels. Computes gradients and logs loss.

## 🏗 Architecture

| Component | Tech Stack | Role |
|-----------|------------|------|
| **Orchestrator** | Kubernetes (k3d) | Manages container lifecycle |
| **Tracker** | MLflow | Tracks loss curves and metrics |
| **Communication** | FastAPI / HTTP | Transmits "smashed data" (tensors) |
| **Model** | PyTorch | Split CNN (MNIST) |

## 🚀 Prerequisites

* [Docker](https://www.docker.com/)
* [k3d](https://k3d.io/) (for local clusters)
* `kubectl`

## 🛠 Installation & Setup

### 1. Create the Cluster
Create a cluster with port forwarding for the LoadBalancer.
```bash
k3d cluster create mlops-cluster -p "8080:80@loadbalancer" --agents 2
```

2. Deploy Infrastructure
Deploy MLflow, PostgreSQL, and SeaweedFS (S3).


```bash

kubectl apply -f k8s/mlflow-stack.yaml
#Wait for pods to become ready: kubectl get pods -n mlflow
```
3. Build & Import Simulation Image
Build the Docker image containing the Split Learning logic.


```bash

cd src
docker build -t split-sim:latest .
k3d image import split-sim:latest -c mlops-cluster
```

4. Start the Simulation
Deploy the Split Server and Client.
```bash

kubectl apply -f k8s/split-learning.yaml
```

📊 Viewing Results
Port Forward MLflow:
```bash
kubectl port-forward svc/mlflow -n mlflow 5000:5000
# Forward local port 8333 to the service 'seaweedfs' on port 8333
kubectl port-forward svc/seaweedfs -n mlflow 8333:8333
```
Access Dashboard: Open http://localhost:5000 in your browser.

View Experiment: Look for the "Split_Learning_Sim" experiment. You will see real-time loss updates.

Start stop scale replicas 0 to 1:

```bash
kubectl scale deployment split-client split-server --replicas=0 -n mlflow
```
Logs:

```bash
kubectl logs -n mlflow -l app=split-server --tail=50
kubectl logs -n mlflow -l app=split-client -f
```
