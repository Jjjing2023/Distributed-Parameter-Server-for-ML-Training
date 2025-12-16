# Terraform Deployment Guide for Distributed Parameter Server

This directory contains Terraform configuration to deploy the distributed parameter server system to AWS using ECS Fargate.

## Architecture

The deployment creates:

- **VPC** with 2 public subnets across 2 availability zones
- **Security Groups** for parameter server and workers
- **ECR Repositories** for Docker images (parameter server and worker)
- **ECS Cluster** running on Fargate
- **Service Discovery** for internal DNS resolution
- **CloudWatch Log Groups** for container logs
- **2 ECS Services**:
  - Parameter Server (1 instance)
  - Workers (configurable count, default 2)

## Prerequisites

1. AWS CLI configured with credentials
2. Terraform >= 1.0 installed
3. Docker installed (for building images)
4. Existing IAM role named "LabRole" in your AWS account

## Deployment Steps

### Step 1: Initialize Terraform

```bash
cd terraform
terraform init
```

### Step 2: Review and Customize Variables

Edit `variables.tf` or create a `terraform.tfvars` file:

```hcl
aws_region   = "us-west-2"
project_name = "distributed-ps"
environment  = "dev"
worker_count = 4  # Adjust based on your needs
```

### Step 3: Plan the Deployment

```bash
terraform plan
```

Review the resources that will be created.

### Step 4: Build and Push Docker Images

Before applying Terraform, you need to build and push Docker images to ECR.

First, apply Terraform to create ECR repositories only:

```bash
# Create ECR repositories first
terraform apply -target=aws_ecr_repository.parameter_server -target=aws_ecr_repository.worker
```

Then build and push images:

```bash
# Get the ECR repository URLs from Terraform output
PS_REPO=$(terraform output -raw parameter_server_ecr_repository_url)
WORKER_REPO=$(terraform output -raw worker_ecr_repository_url)

# Login to ECR
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin $PS_REPO

# Build and push parameter server image
cd ..
docker build -f Dockerfile.server -t $PS_REPO:latest .
docker push $PS_REPO:latest

# Build and push worker image
docker build -f Dockerfile.worker -t $WORKER_REPO:latest .
docker push $WORKER_REPO:latest
```

### Step 5: Deploy Complete Infrastructure

```bash
cd terraform
terraform apply
```

Type `yes` to confirm the deployment.

### Step 6: Update Worker Environment Variable

After the parameter server is deployed, you need to update the worker task definition with the correct parameter server address:

```bash
# Get the parameter server DNS name
PS_DNS=$(terraform output -raw parameter_server_dns_name)

echo "Parameter Server DNS: $PS_DNS"
```

Then update the worker task definition in `main.tf` at line ~290 to set the `PARAMETER_SERVER_ADDRESS` environment variable to the DNS name shown above, or use the AWS console to update the task definition.

Alternatively, you can update via AWS CLI:

```bash
# This requires manual intervention - see terraform output for the DNS name
# and update the worker service to use the new task definition
```

### Step 7: Monitor Deployment

```bash
# Check ECS cluster status
aws ecs describe-clusters --clusters $(terraform output -raw ecs_cluster_name)

# Check services
aws ecs describe-services --cluster $(terraform output -raw ecs_cluster_name) \
  --services $(terraform output -raw parameter_server_service_name)

# View logs
aws logs tail $(terraform output -raw parameter_server_log_group) --follow
aws logs tail $(terraform output -raw worker_log_group) --follow
```

## Scaling Workers

To change the number of workers:

1. Update `worker_count` in `variables.tf` or `terraform.tfvars`
2. Run `terraform apply`

```bash
# Example: Scale to 8 workers
terraform apply -var="worker_count=8"
```

## Resource Configuration

### Compute Resources

The default configuration uses:

- **Parameter Server**: 2 vCPUs, 4GB RAM
- **Workers**: 2 vCPUs, 4GB RAM each

Adjust these in `variables.tf`:

- `parameter_server_cpu` / `parameter_server_memory`
- `worker_cpu` / `worker_memory`

### Cost Optimization

For development/testing, you can reduce resources:

```hcl
parameter_server_cpu    = 1024  # 1 vCPU
parameter_server_memory = 2048  # 2GB
worker_cpu              = 1024
worker_memory           = 2048
worker_count            = 2
```

## Troubleshooting

### Workers can't connect to parameter server

1. Check security group rules allow traffic on port 8000
2. Verify service discovery is working:

   ```bash
   aws servicediscovery list-services --filters Name=NAMESPACE_ID,Values=$(terraform output -raw service_discovery_namespace_id)
   ```

3. Check CloudWatch logs for errors:
   ```bash
   aws logs tail $(terraform output -raw worker_log_group) --follow
   ```

### Task fails to start

1. Check CloudWatch logs for errors
2. Verify Docker images were pushed successfully:

   ```bash
   aws ecr describe-images --repository-name distributed-ps-dev-parameter-server
   aws ecr describe-images --repository-name distributed-ps-dev-worker
   ```

3. Verify LabRole has necessary permissions:
   - ECR pull permissions
   - CloudWatch Logs write permissions
   - ECS task execution permissions

## Cleanup

To destroy all resources:

```bash
terraform destroy
```

**Note**: This will delete all resources including logs and ECR images.

## Terraform Outputs

After deployment, use these commands to get important information:

```bash
terraform output vpc_id
terraform output ecs_cluster_name
terraform output parameter_server_ecr_repository_url
terraform output worker_ecr_repository_url
terraform output parameter_server_dns_name
```

## Next Steps

1. Monitor training progress via CloudWatch Logs
2. Adjust worker count based on performance metrics
3. Export trained models from parameter server
4. Set up CloudWatch alarms for failures
