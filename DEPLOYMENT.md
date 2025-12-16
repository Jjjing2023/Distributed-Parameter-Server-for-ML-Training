# AWS Deployment Summary

## Overview

This deployment setup allows you to run the distributed parameter server on AWS ECS Fargate with automatic scaling and service discovery.

## What Was Created

### 1. **Docker Configuration**

- `Dockerfile.server` - Containerizes the parameter server
- `Dockerfile.worker` - Containerizes the worker nodes
- Updated `requirements.txt` - Added gRPC and protobuf dependencies

### 2. **Code Updates**

- `src/parameter_server/server.py` - Now reads `SERVER_PORT` and `TOTAL_WORKERS_EXPECTED` from environment variables
- `src/workers/worker.py` - Now reads `PARAMETER_SERVER_ADDRESS` from environment variable

### 3. **Terraform Infrastructure** (`terraform/`)

#### Core Infrastructure Files:

- **`main.tf`** - Complete AWS infrastructure including:

  - VPC with 2 public subnets across 2 AZs
  - Internet Gateway and routing
  - Security groups for parameter server and workers
  - ECR repositories for Docker images
  - ECS cluster with Fargate
  - CloudWatch log groups
  - Service Discovery for internal DNS
  - ECS task definitions and services

- **`variables.tf`** - Configurable parameters:

  - AWS region
  - Worker count (default: 16)
  - CPU/Memory allocation
  - Network configuration

- **`outputs.tf`** - Useful outputs:
  - ECR repository URLs
  - Service names
  - Log group names
  - DNS names

#### Helper Scripts:

- **`deploy.sh`** - Automated deployment script
- **`destroy.sh`** - Cleanup script
- **`README.md`** - Detailed deployment guide
- **`.gitignore`** - Terraform state files

## Quick Start

### ⚠️ Important for M1/M2/M3 Mac Users

**AWS Fargate requires linux/amd64 architecture**, but M1/M2/M3 Macs use ARM64.

**Good news:** The `deploy.sh` script already includes `--platform linux/amd64` flags, so it works correctly on M1 Macs automatically!

### Option 1: Automated Deployment (Recommended)

**One command to deploy everything:**

```bash
cd terraform
./deploy.sh
```

This script automatically:

1. Initializes Terraform
2. Creates ECR repositories
3. Builds Docker images for linux/amd64 (works on M1 Macs)
4. Pushes images to ECR
5. Deploys complete infrastructure
6. Displays monitoring commands

### Option 2: Manual Deployment (Advanced)

```bash
cd terraform

# 1. Initialize
terraform init

# 2. Create ECR repositories
terraform apply -target=aws_ecr_repository.parameter_server -target=aws_ecr_repository.worker

# 3. Build and push images
PS_REPO=$(terraform output -raw parameter_server_ecr_repository_url)
WORKER_REPO=$(terraform output -raw worker_ecr_repository_url)

aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin $PS_REPO

cd ..

# IMPORTANT: Use --platform linux/amd64 on M1/M2/M3 Macs
docker build --platform linux/amd64 -f Dockerfile.server -t $PS_REPO:latest .
docker push $PS_REPO:latest

docker build --platform linux/amd64 -f Dockerfile.worker -t $WORKER_REPO:latest .
docker push $WORKER_REPO:latest

# 4. Deploy infrastructure
cd terraform
terraform apply
```

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  AWS VPC                        │
│  ┌───────────────┐        ┌──────────────────┐ │
│  │   Subnet 1    │        │    Subnet 2      │ │
│  │  us-west-1a   │        │   us-west-1b     │ │
│  │               │        │                  │ │
│  │ ┌───────────┐ │        │ ┌──────────────┐ │ │
│  │ │Parameter  │ │        │ │  Worker 1    │ │ │
│  │ │  Server   │◄├────────┼─┤              │ │ │
│  │ │  (ECS)    │ │        │ │  (ECS)       │ │ │
│  │ └───────────┘ │        │ └──────────────┘ │ │
│  │               │        │                  │ │
│  │               │        │ ┌──────────────┐ │ │
│  │               │        │ │  Worker 2    │ │ │
│  │               │◄───────┼─┤              │ │ │
│  │               │        │ │  (ECS)       │ │ │
│  │               │        │ └──────────────┘ │ │
│  └───────────────┘        └──────────────────┘ │
│                                                 │
│  Service Discovery:                             │
│  parameter-server.distributed-ps-dev.local:8000│
└─────────────────────────────────────────────────┘
```

## Configuration

### Scaling Workers

Edit `terraform/variables.tf` and change `worker_count`:

```hcl
variable "worker_count" {
  default = 4  # Change this value
}
```

Then apply:

```bash
terraform apply -var="worker_count=8"
```

### Adjusting Resources

**For production workloads:**

```hcl
parameter_server_cpu    = 4096  # 4 vCPUs
parameter_server_memory = 8192  # 8GB
worker_cpu              = 4096
worker_memory           = 8192
```

**For development/testing:**

```hcl
parameter_server_cpu    = 1024  # 1 vCPU
parameter_server_memory = 2048  # 2GB
worker_cpu              = 1024
worker_memory           = 2048
```

## Monitoring

### View Logs

```bash
# Parameter Server logs
aws logs tail /ecs/distributed-ps-dev-parameter-server --follow

# Worker logs
aws logs tail /ecs/distributed-ps-dev-worker --follow
```

### Check Service Status

```bash
# List running tasks
aws ecs list-tasks --cluster distributed-ps-dev-cluster

# Describe service
aws ecs describe-services \
  --cluster distributed-ps-dev-cluster \
  --services distributed-ps-dev-parameter-server
```

## Cleanup

To destroy all AWS resources:

```bash
cd terraform
./destroy.sh
```

Or manually:

```bash
terraform destroy
```

## Cost Optimization

### Estimated Costs (us-west-2):

**Default Configuration (16 workers):**

- 17 Fargate tasks × 2 vCPU × 4GB RAM (1 server + 16 workers)
- ~$0.08/hour per task = **~$1.36/hour** (~$990/month if running 24/7)

**Development Configuration (1 vCPU, 2GB):**

- ~$0.04/hour per task = **~$0.12/hour** (~$87/month)

**Cost Saving Tips:**

1. Stop services when not in use
2. Use smaller instance sizes for testing
3. Reduce worker count during development
4. Enable CloudWatch log retention limits (default: 7 days)

## Troubleshooting

### Task fails to start with "exec format error" or crashes immediately

**Cause**: Architecture mismatch - Docker image was built for ARM64 but AWS Fargate requires linux/amd64.

**Solution**:
1. Delete existing images from ECR
2. Rebuild and redeploy: `cd terraform && ./deploy.sh`
3. Or manually rebuild with: `docker build --platform linux/amd64 -f Dockerfile.server -t ...`

**Verify architecture:**
```bash
# Check local image architecture
docker inspect distributed-ps-parameter-server:latest | grep Architecture

# Should show: "Architecture": "amd64"
# If it shows "arm64", rebuild with --platform linux/amd64
```

### Workers can't connect to Parameter Server

**Check security groups:**

```bash
terraform output security_group_parameter_server_id
terraform output security_group_worker_id
```

**Verify service discovery:**

```bash
aws servicediscovery list-services
```

**Check logs for connection errors:**

```bash
aws logs tail /ecs/distributed-ps-dev-worker --follow
```

### Task fails to start

**View task stopped reasons:**

```bash
aws ecs describe-tasks \
  --cluster distributed-ps-dev-cluster \
  --tasks <task-id>
```

**Common issues:**

- **Architecture mismatch** (ARM64 vs AMD64) - See above
- Missing Docker images in ECR
- Insufficient permissions for LabRole
- Resource limits (CPU/memory too low)

### ECR Authentication Issues

**Re-authenticate:**

```bash
aws ecr get-login-password --region us-west-2 | \
  docker login --username AWS --password-stdin <ecr-url>
```

## Next Steps

1. **Monitor Training Progress**: Check CloudWatch logs for training metrics
2. **Scale Workers**: Adjust worker count based on performance
3. **Export Model**: Add functionality to save trained models to S3
4. **Add Metrics**: Integrate CloudWatch metrics for training progress
5. **Implement Checkpointing**: Add periodic model checkpoints to S3

## IAM Permissions Required

The existing **LabRole** must have:

- `ecr:GetAuthorizationToken`
- `ecr:BatchGetImage`
- `ecr:GetDownloadUrlForLayer`
- `logs:CreateLogStream`
- `logs:PutLogEvents`
- `ecs:*` (for task execution)

## Security Considerations

- Security groups restrict gRPC traffic to internal VPC only
- No public load balancer (parameter server not exposed to internet)
- All container logs sent to CloudWatch
- ECR image scanning enabled

## Proto File Regeneration

If you modify `src/communication/ps.proto`, you need to regenerate the Python gRPC files:

```bash
# Using the provided script
./regenerate_proto.sh

# Or manually
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. src/communication/ps.proto
```

**Note**: Make sure to regenerate proto files before building Docker images if you've modified the proto definitions.

## Support

For issues or questions:

1. Check CloudWatch logs for error messages
2. Review Terraform outputs for configuration details
3. Verify all prerequisites are met
4. Check AWS service quotas (ECS task limits, etc.)
