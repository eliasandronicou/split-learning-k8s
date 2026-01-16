# Split Learning Simulation on Kubernetes
![Warning](https://img.shields.io/badge/TensorFlow-THIS_IS_THE_TensFlow_VERSION-red?style=for-the-badge&logo=tensorflow)

This repository simulates a **Split Learning** architecture using Kubernetes (k3d). It separates a TensorFlow Neural Network into two parts across different pods:
1. **Client Pod:** Holds data & bottom layers. Computes activations.
2. **Server Pod:** Holds top layers & labels. Computes gradients and logs loss.

## 🏗 Architecture

| Component | Tech Stack | Role |
|-----------|------------|------|
| **Orchestrator** | Kubernetes (k3d) | Manages container lifecycle |
| **Tracker** | MLflow v2.9.2 | Tracks loss curves and metrics |
| **Communication** | FastAPI / HTTP | Transmits "smashed data" (tensors) |
| **Model** | TensorFlow (CPU-only) | Split CNN (MNIST) |
| **Storage** | SeaweedFS (S3) | Artifact storage |
| **Database** | PostgreSQL 13 | MLflow backend store |

## 🚀 Prerequisites

* [Docker](https://www.docker.com/)
* [k3d](https://k3d.io/) (for local clusters)
* `kubectl`
* At least 10GB free disk space

## 🛠 Initial Setup

### 1. Create the k3d Cluster

Create a cluster with port forwarding for the LoadBalancer:

```bash
k3d cluster create mlops-cluster --agents 1 --servers 1 --port "8080:80@loadbalancer"
```

**Verify cluster is running:**
```bash
kubectl get nodes
```

You should see 2 nodes (1 server, 1 agent) in `Ready` state.

### 2. Deploy MLflow Infrastructure

Deploy MLflow, PostgreSQL, and SeaweedFS (S3):

```bash
kubectl apply -f k8s/mlflow-stack.yaml
```

**Wait for all pods to become ready:**
```bash
kubectl get pods -n mlflow -w
```

Expected pods:
- `postgres-0` - Running
- `seaweedfs-0` - Running
- `mlflow-xxxxx` - Running
- `aws-init-job-xxxxx` - Completed

This may take 2-3 minutes for all images to pull and pods to start.

### 3. Build & Import Application Image

Build the optimized Docker image with CPU-only TensorFlow:

```bash
cd src
docker build -t split-learning:optimized .
```

**Import to k3d cluster:**
```bash
k3d image import split-learning:optimized -c mlops-cluster
```

> **Note:** This step takes 3-5 minutes as the image is ~1.6GB.

### 4. Deploy Split Learning Application

Deploy the Split Server and Client:

```bash
cd ..  # Return to project root
kubectl apply -f k8s/split-learning.yaml
```

**Verify deployment:**
```bash
kubectl get pods -n mlflow
```

You should see:
- `split-server-xxxxx` - Running
- `split-client-xxxxx` - Running

### 5. Access MLflow Dashboard

Port forward MLflow to your local machine:

```bash
kubectl port-forward svc/mlflow -n mlflow 5000:5000
```

**Access Dashboard:** Open http://localhost:5000 in your browser.

**View Experiment:** Look for the "Split_Learning_Sim" or "Federated_Learning_Sim" experiment to see real-time loss updates.

---

## 🔄 Updating the Application

### Update Application Code

When you modify Python code in `src/`:

```bash
# 1. Rebuild the Docker image
cd src
docker build -t split-learning:optimized .

# 2. Import to k3d
k3d image import split-learning:optimized -c mlops-cluster

# 3. Restart deployments
kubectl rollout restart deployment split-server split-client -n mlflow

# 4. Verify
kubectl get pods -n mlflow
```

### Update MLflow Version

> **⚠️ Warning:** Changing MLflow version requires database recreation!

```bash
# 1. Update version in three places:
#    - src/requirements.txt
#    - src/Dockerfile (line 27)
#    - k8s/mlflow-stack.yaml (line 248)

# 2. Delete existing postgres database
kubectl delete statefulset postgres -n mlflow
kubectl delete pvc pg-data-postgres-0 -n mlflow

# 3. Reapply MLflow stack
kubectl apply -f k8s/mlflow-stack.yaml

# 4. Rebuild application image
cd src
docker build -t split-learning:optimized .
k3d image import split-learning:optimized -c mlops-cluster

# 5. Restart deployments
kubectl rollout restart deployment split-server split-client mlflow -n mlflow
```

### Update Kubernetes Manifests

When you modify `k8s/*.yaml` files:

```bash
# Apply the changes
kubectl apply -f k8s/mlflow-stack.yaml
# or
kubectl apply -f k8s/split-learning.yaml

# Restart affected deployments
kubectl rollout restart deployment <deployment-name> -n mlflow
```

---

## 📊 Monitoring & Management

### View Logs

**Server logs:**
```bash
kubectl logs -n mlflow -l app=split-server --tail=50 -f
```

**Client logs:**
```bash
kubectl logs -n mlflow -l app=split-client --tail=50 -f
```

**MLflow logs:**
```bash
kubectl logs -n mlflow deployment/mlflow --tail=50 -f
```

### Scale Deployments

**Stop training (scale to 0):**
```bash
kubectl scale deployment split-client split-server --replicas=0 -n mlflow
```

**Start training (scale to 1):**
```bash
kubectl scale deployment split-client split-server --replicas=1 -n mlflow
```

---

## 🧹 Cleanup & Maintenance

### Clean Up Docker Resources

**Remove old images:**
```bash
# List images
docker images | grep split

# Remove specific old image
docker rmi split-sim:latest

# Remove dangling images
docker image prune -f

# Remove all unused images
docker image prune -a -f
```

**Check Docker disk usage:**
```bash
docker system df
```

### Delete the Cluster

**Stop the cluster:**
```bash
k3d cluster stop mlops-cluster
```

**Delete the cluster (removes all data):**
```bash
k3d cluster delete mlops-cluster
```

### Reset MLflow Data

**Delete MLflow experiments (keeps infrastructure):**
```bash
kubectl delete deployment mlflow -n mlflow
kubectl delete pvc pg-data-postgres-0 -n mlflow
kubectl apply -f k8s/mlflow-stack.yaml
```

---

## 📁 Project Structure

```
split-learning-k8s/
├── src/
│   ├── Dockerfile              # Multi-stage build with CPU-only TensorFlow
│   ├── requirements.txt        # Python dependencies (mlflow==2.9.2)
│   ├── .dockerignore          # Files to exclude from Docker build
│   ├── server_part.py         # Server-side model and FastAPI endpoints
│   ├── client_part.py         # Client-side model and training loop
│   └── model_def.py           # Model architecture definitions
├── k8s/
│   ├── mlflow-stack.yaml      # MLflow, Postgres, SeaweedFS deployment
│   └── split-learning.yaml    # Split learning client/server deployment
└── README.md                   # This file
```

---

## 🔧 Configuration

### Environment Variables

**Split Learning Mode:**
Set `LEARNING_MODE=split` in `k8s/split-learning.yaml`

**Federated Learning Mode:**
Set `LEARNING_MODE=federated` in `k8s/split-learning.yaml`

### MLflow Configuration

MLflow tracking URI is configured in `k8s/split-learning.yaml`:
```yaml
- name: MLFLOW_TRACKING_URI
  value: "http://mlflow.mlflow.svc.cluster.local:5000"
```


---

## 🎯 Quick Reference

```bash
# Start everything
k3d cluster create mlops-cluster --agents 1 --servers 1
kubectl apply -f k8s/mlflow-stack.yaml
cd src && docker build -t split-learning:optimized . && cd ..
k3d image import split-learning:optimized -c mlops-cluster
kubectl apply -f k8s/split-learning.yaml
kubectl port-forward svc/mlflow -n mlflow 5000:5000

# Update code
cd src && docker build -t split-learning:optimized . && cd ..
k3d image import split-learning:optimized -c mlops-cluster
kubectl rollout restart deployment split-server split-client -n mlflow

# View logs
kubectl logs -n mlflow -l app=split-server -f

# Stop everything
k3d cluster delete mlops-cluster
```
