# Distributed Parameter Server for Deep Learning

A scalable distributed parameter server system for training ResNet-18 on CIFAR-100, supporting both **synchronous** and **asynchronous** gradient aggregation modes.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [AWS Deployment](#aws-deployment)
- [Running Experiments](#running-experiments)
- [Project Structure](#project-structure)
- [Results](#results)
- [Troubleshooting](#troubleshooting)

---

## Overview

This project implements a distributed parameter server architecture for training deep learning models at scale. It demonstrates data parallelism where multiple workers train on different data partitions and synchronize gradients through a central parameter server.

**Key Capabilities:**

- **Model**: ResNet-18 (11.2M parameters)
- **Dataset**: CIFAR-100 (50K training, 10K test images)
- **Modes**: Synchronous & Asynchronous gradient aggregation
- **Scaling**: 1-32 workers on AWS ECS Fargate
- **Communication**: gRPC with gradient compression

---

## Features

✅ **Dual Training Modes**

- **Sync Mode**: All workers synchronize at each step (higher accuracy, slower)
- **Async Mode**: Workers update independently (faster, potential staleness)

✅ **AWS Cloud Deployment**

- Automated deployment with Terraform
- ECS Fargate (serverless containers)
- Auto-scaling workers
- CloudWatch monitoring

✅ **Production-Ready**

- Docker containerization
- Service discovery (internal DNS)
- Automatic failover
- Comprehensive logging

✅ **Experiment Tracking**

- JSON metrics export
- CloudWatch log parsing
- Visualization scripts
- Performance analysis tools

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    AWS VPC (10.0.0.0/16)                    │
│                                                               │
│  ┌────────────────────┐       ┌──────────────────────────┐  │
│  │  Parameter Server  │◄──────┤  Network Load Balancer   │  │
│  │  (ECS Fargate)     │       │  (Internal)              │  │
│  │                    │       └──────────────────────────┘  │
│  │  - gRPC Server     │                  ▲                  │
│  │  - Model Storage   │                  │                  │
│  │  - Gradient Agg    │                  │                  │
│  └────────────────────┘         Push/Pull Gradients        │
│           │                              │                  │
│           │                    ┌─────────┴─────────┐        │
│           │                    │                   │        │
│     CloudWatch         ┌───────▼──────┐   ┌───────▼──────┐ │
│       Logs             │   Worker 0   │   │   Worker 1   │ │
│                        │  (ECS Task)  │   │  (ECS Task)  │ │
│                        │              │   │              │ │
│                        │ - Data: 0-N  │   │ - Data: N-2N │ │
│                        │ - Local SGD  │   │ - Local SGD  │ │
│                        └──────────────┘   └──────────────┘ │
│                                 ...                         │
│                        ┌──────────────┐   ┌──────────────┐ │
│                        │  Worker N-1  │   │  Worker N    │ │
│                        └──────────────┘   └──────────────┘ │
└─────────────────────────────────────────────────────────────┘

Data Flow:
1. Workers partition CIFAR-100 training data (no overlap)
2. Each worker trains locally for N steps
3. Workers push gradients to parameter server
4. Server aggregates (sync: wait for all, async: immediate)
5. Workers fetch updated parameters
6. Repeat until convergence
```

**Key Components:**

- **Parameter Server**: Central hub for model parameters and gradient aggregation
- **Workers**: Distributed training nodes, each processing a data partition
- **Network Load Balancer**: Routes worker requests to parameter server
- **Service Discovery**: Internal DNS for worker-to-server communication
- **CloudWatch**: Centralized logging and monitoring

---

## Quick Start

### Prerequisites

- Python 3.9+
- pip
- Git

### 1. Clone & Setup

```bash
# Clone repository
git clone <your-repo-url>
cd distributed-parameter-server

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Baseline (Single Machine)

```bash
# Download CIFAR-100 and train baseline
cd baseline/
python baseline_training.py

# Expected: ~65% accuracy in 6 hours (20 epochs)
```

### 3. Generate Visualizations

```bash
cd ..
python create_charts.py
```

---

## AWS Deployment

### Prerequisites

**Required:**

- AWS Account with **LabRole** IAM role
- AWS CLI configured (`aws configure`)
- Terraform >= 1.0
- Docker
- **M1/M2/M3 Mac Users**: Docker Desktop with Rosetta emulation enabled

**IAM Permissions** (LabRole must have):

- ECR: Pull/push images
- ECS: Task execution
- CloudWatch: Logs write
- VPC: Network resources

### Deployment Steps

**Automated deployment with one command:**

```bash
cd terraform
./deploy.sh
```

This automatically:

1. ✅ Initializes Terraform
2. ✅ Creates ECR repositories
3. ✅ Builds Docker images (linux/amd64 for AWS)
4. ✅ Pushes images to ECR
5. ✅ Deploys complete infrastructure (4 workers by default)
6. ✅ Displays monitoring commands

**Duration**: ~10-30 minutes

### Configuration Options

#### Scaling Workers

**Default: 4 workers**

Change worker count:

```bash
# Method 1: One-time override
terraform apply -var 'worker_count=8' -auto-approve

# Method 2: Edit variables.tf
# Change line: default = 4  →  default = 8
terraform apply
```

#### Switch Between Sync/Async Modes

```bash
# Synchronous mode (default)
terraform apply -var 'server_mode=sync' -var 'worker_count=4'

# Asynchronous mode
terraform apply -var 'server_mode=async' -var 'worker_count=8'
```

#### Resource Allocation

**Development (Low Cost):**

```hcl
# Edit terraform/variables.tf
parameter_server_cpu    = 1024  # 1 vCPU
parameter_server_memory = 2048  # 2GB
worker_cpu              = 1024
worker_memory           = 2048
```

**Production (High Performance):**

```hcl
parameter_server_cpu    = 4096  # 4 vCPUs
parameter_server_memory = 8192  # 8GB
worker_cpu              = 4096
worker_memory           = 8192
```

### Monitoring Your Deployment

#### View Real-Time Logs

```bash
# Parameter Server logs
aws logs tail /ecs/distributed-ps-dev-parameter-server --follow

# Worker logs (all workers)
aws logs tail /ecs/distributed-ps-dev-worker --follow

# Filter for specific events
aws logs tail /ecs/distributed-ps-dev-worker --follow --filter-pattern "METRICS_JSON"
```

#### Check Service Status

```bash
# List running tasks
aws ecs list-tasks --cluster distributed-ps-dev-cluster --region us-west-2

# Describe services
aws ecs describe-services \
  --cluster distributed-ps-dev-cluster \
  --services distributed-ps-dev-parameter-server \
  --region us-west-2
```

#### Verify Workers Registered

```bash
# Watch parameter server logs for worker registration
aws logs tail /ecs/distributed-ps-dev-parameter-server --follow | grep "Worker.*registered"

# Expected output:
# Worker 0 registered with ID 0
# Worker 1 registered with ID 1
# ...
```

### Cleanup

**⚠️ Important: Always destroy resources after experiments to avoid AWS charges!**

```bash
cd terraform
./destroy.sh

# Or manually:
terraform destroy -auto-approve
```

**What gets deleted:**

- All ECS tasks (workers & parameter server)
- ECR repositories and Docker images
- VPC, subnets, security groups
- CloudWatch log groups (logs are deleted!)
- Load balancers

---

## Running Experiments

### Step 1: Deploy Infrastructure

```bash
cd terraform
terraform apply -var 'server_mode=sync' -var 'worker_count=4' -auto-approve
```

### Step 2: Monitor Training

```bash
# Watch worker progress
aws logs tail /ecs/distributed-ps-dev-worker --follow

# Look for:
# - "Worker X registered with ID X"
# - "Epoch 1/3 completed"
# - "Final test accuracy: XX.XX%"
# - "METRICS_JSON: {...}"
```

### Step 3: Wait for Completion

**Expected Duration:**

- Sync 4 workers: ~20-30 minutes
- Async 4 workers: ~15-25 minutes
- Sync 8 workers: ~10-15 minutes
- Async 8 workers: ~8-12 minutes

**Signs of completion:**

- Parameter server logs: `PARAMETER SERVER FINAL STATISTICS`
- Worker logs: `Worker X completed training!`
- All workers print `METRICS_JSON: {...}`

### Step 4: Extract Results

```bash
cd ..

# Parse CloudWatch logs to JSON
python scripts/parse_cloudwatch_logs.py \
  --experiment-name sync_4workers \
  --use-terraform \
  --output experiment_results/sync_4workers.json \
  --hours-ago 2

# Verify results
cat experiment_results/sync_4workers.json | jq '.worker_metrics_aggregated.num_workers'
# Should output: 4
```

### Step 5: Destroy & Redeploy

**🚨 CRITICAL: Always destroy between experiments!**

```bash
# Destroy current deployment
cd terraform
terraform destroy -auto-approve

# Wait 1-2 minutes for full cleanup

# Deploy next experiment
terraform apply -var 'server_mode=async' -var 'worker_count=8' -auto-approve
```

**Why destroy between experiments?**

- Prevents worker ID conflicts (workers 0,1,2,3 → 4,5,6,7...)
- Ensures clean state
- Avoids mixing data from different experiments

### Step 6: Visualize Results

```bash
# Generate comparison charts
python scripts/visualize_results.py

# Creates charts in experiment_results/charts/
```

---

## Project Structure

```
distributed-parameter-server/
├── README.md                   # This file
├── DEPLOYMENT.md              # Detailed AWS deployment guide
├── EXPERIMENT_GUIDE.md        # Experiment protocols
├── requirements.txt           # Python dependencies
│
├── baseline/                  # Single-machine baseline
│   ├── baseline_training.py   # ResNet-18 training script
│   └── results/               # Baseline metrics
│
├── src/                       # Core distributed system
│   ├── parameter_server/      # Parameter server implementation
│   │   ├── __init__.py
│   │   └── server.py          # gRPC server, gradient aggregation
│   │
│   ├── workers/               # Worker implementation
│   │   ├── __init__.py
│   │   └── worker.py          # Data loading, local training
│   │
│   └── communication/         # gRPC protocol
│       ├── ps.proto           # Protocol buffer definition
│       ├── ps_pb2.py          # Generated Python code
│       └── ps_pb2_grpc.py     # Generated gRPC stubs
│
├── terraform/                 # AWS infrastructure as code
│   ├── main.tf                # VPC, ECS, load balancer
│   ├── variables.tf           # Configurable parameters
│   ├── outputs.tf             # Deployment outputs
│   ├── deploy.sh              # Automated deployment script
│   ├── destroy.sh             # Cleanup script
│   └── README.md              # Terraform-specific docs
│
├── scripts/                   # Utility scripts
│   ├── parse_cloudwatch_logs.py   # Extract metrics from logs
│   └── visualize_results.py       # Generate charts
│
├── experiment_results/        # Experiment data
│   ├── sync_4workers.json
│   ├── async_4workers.json
│   ├── sync_8workers.json
│   ├── async_8workers.json
│   └── charts/                # Generated visualizations
│
├── Dockerfile.server          # Parameter server container
├── Dockerfile.worker          # Worker container
└── regenerate_proto.sh        # Rebuild gRPC files (if modifying ps.proto)
```

---

## Results

### Baseline Performance

**Single Machine (M1 Mac, CPU-only):**

- Training time: ~17 minutes/epoch
- Total (20 epochs): ~6 hours
- Final accuracy: ~65%

### Distributed Performance

#### Synchronous Mode

| Workers | Total Time | Epoch Time | Final Accuracy | Speedup |
| ------- | ---------- | ---------- | -------------- | ------- |
| 4       | ~22 min    | ~10 min    | 19.2%\*        | 1.0x    |
| 8       | ~16 min    | ~5 min     | 18.5%\*        | 1.4x    |
| 16      | ~10 min    | ~2.5 min   | 17.8%\*        | 2.2x    |

#### Asynchronous Mode

| Workers | Total Time | Epoch Time | Final Accuracy | Speedup |
| ------- | ---------- | ---------- | -------------- | ------- |
| 4       | ~20 min    | ~9 min     | 18.1%\*        | 1.1x    |
| 8       | ~12 min    | ~4 min     | 17.5%\*        | 1.8x    |
| 16      | ~8 min     | ~2 min     | 16.2%\*        | 2.8x    |

**\*Note**: Lower accuracy is due to only 3 epochs (vs baseline 20 epochs). Distributed system focuses on demonstrating scalability and communication efficiency.

### Key Findings

1. **Async is faster** but has slightly lower accuracy due to gradient staleness
2. **Diminishing returns** beyond 16 workers (communication overhead)
3. **Sweet spot**: 8-16 workers for best performance/cost ratio
4. **Scaling efficiency**: ~70-80% with proper configuration

---

## Troubleshooting

### Docker Build Issues

**Problem**: `exec format error` or tasks crash immediately

**Cause**: Architecture mismatch (ARM64 vs AMD64)

**Solution**:

```bash
# Always use --platform linux/amd64 on M1/M2/M3 Macs
docker build --platform linux/amd64 -f Dockerfile.server -t ...

# Or use the automated script
cd terraform && ./deploy.sh
```

### Workers Can't Connect to Server

**Symptoms**: Workers timeout or can't reach parameter server

**Debugging**:

```bash
# 1. Check parameter server is running
aws ecs list-tasks --cluster distributed-ps-dev-cluster --service-name distributed-ps-dev-parameter-server

# 2. Check worker logs
aws logs tail /ecs/distributed-ps-dev-worker --follow

# 3. Verify security groups allow port 8000
terraform output security_group_parameter_server_id
```

**Fix**: Ensure security groups allow inbound TCP on port 8000

### Only Some Workers Complete

**Symptoms**: JSON shows `num_workers: 2` but you deployed 4 workers

**Cause**: Workers crashed or are being replaced by ECS

**Solution**:

1. Check worker logs for errors:

   ```bash
   aws logs tail /ecs/distributed-ps-dev-worker --follow
   ```

2. Set worker `desired_count=0` after training starts:

   ```bash
   aws ecs update-service \
     --cluster distributed-ps-dev-cluster \
     --service distributed-ps-dev-worker \
     --desired-count 0
   ```

3. Always destroy between experiments

### Missing Metrics in JSON

**Problem**: `server_metrics: null` or missing worker data

**Causes**:

1. Workers didn't finish training
2. Time window too narrow (`--hours-ago 1` might miss earlier workers)
3. Workers crashed before printing metrics

**Solution**:

```bash
# Use wider time window
python scripts/parse_cloudwatch_logs.py \
  --experiment-name my_experiment \
  --use-terraform \
  --output results.json \
  --hours-ago 3  # Wider window

# Check for all METRICS_JSON lines
aws logs filter-log-events \
  --log-group-name /ecs/distributed-ps-dev-worker \
  --filter-pattern "METRICS_JSON"
```

### Terraform State Issues

**Problem**: `Resource already exists` or state conflicts

**Solution**:

```bash
# Remove problematic resource from state
terraform state rm aws_ecs_service.worker

# Or destroy and start fresh
terraform destroy -auto-approve
rm -rf .terraform/ terraform.tfstate*
terraform init
```

---

## Development Timeline

- **Week 1**: Baseline, dataset, gRPC skeleton
- **Week 2**: Sync implementation, correctness testing
- **Week 3**: AWS deployment, scaling experiments (1→32 workers)
- **Week 4**: Async implementation, analysis, final report

---

## Requirements

See `requirements.txt`:

```
torch>=1.9.0
torchvision>=0.10.0
numpy
matplotlib
grpcio
grpcio-tools
protobuf
```

---

## Contributing

This is a course project for **CS6650: Building Scalable Distributed Systems** (Fall 2025).

Team Members: Tianjing Liu, Daisy Ding, Xiaoti Hu

---

## License

Academic use only - CS6650 Fall 2025

---

## Additional Resources

- **[DEPLOYMENT.md](DEPLOYMENT.md)**: Detailed AWS deployment guide
- **[EXPERIMENT_GUIDE.md](EXPERIMENT_GUIDE.md)**: How to run experiments
- **[terraform/README.md](terraform/README.md)**: Terraform-specific documentation

---

## Support

For issues:

1. Check [Troubleshooting](#troubleshooting) section
2. Review CloudWatch logs
3. Verify AWS quotas (ECS task limits)
4. Check Terraform outputs for configuration

---

**Happy Distributed Training! 🚀**
