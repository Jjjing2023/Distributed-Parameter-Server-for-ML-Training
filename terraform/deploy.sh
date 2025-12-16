#!/bin/bash

# Deployment script for Distributed Parameter Server on AWS ECS
# This script automates the build, push, and deployment process

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
AWS_REGION=${AWS_REGION:-"us-west-2"}
PROJECT_ROOT="../"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Distributed Parameter Server Deployment${NC}"
echo -e "${GREEN}========================================${NC}"

# Step 1: Initialize Terraform
echo -e "\n${YELLOW}Step 1: Initializing Terraform...${NC}"
terraform init

# Step 2: Create ECR repositories first
echo -e "\n${YELLOW}Step 2: Creating ECR repositories...${NC}"
terraform apply -auto-approve \
  -target=aws_ecr_repository.parameter_server \
  -target=aws_ecr_repository.worker

# Get ECR repository URLs
echo -e "\n${YELLOW}Step 3: Getting ECR repository URLs...${NC}"
PS_REPO=$(terraform output -raw parameter_server_ecr_repository_url)
WORKER_REPO=$(terraform output -raw worker_ecr_repository_url)

echo -e "${GREEN}Parameter Server Repository: ${PS_REPO}${NC}"
echo -e "${GREEN}Worker Repository: ${WORKER_REPO}${NC}"

# Step 4: Login to ECR
echo -e "\n${YELLOW}Step 4: Logging in to ECR...${NC}"
aws ecr get-login-password --region ${AWS_REGION} | \
  docker login --username AWS --password-stdin ${PS_REPO}

# Step 5: Build and push Parameter Server image
echo -e "\n${YELLOW}Step 5: Building Parameter Server Docker image for linux/amd64...${NC}"
cd ${PROJECT_ROOT}
# docker build -f Dockerfile.server -t ${PS_REPO}:latest .
docker build --platform linux/amd64 -f Dockerfile.server -t ${PS_REPO}:latest .

echo -e "\n${YELLOW}Pushing Parameter Server image to ECR...${NC}"
docker push ${PS_REPO}:latest

# Step 6: Build and push Worker image
echo -e "\n${YELLOW}Step 6: Building Worker Docker image for linux/amd64...${NC}"
# docker build -f Dockerfile.worker -t ${WORKER_REPO}:latest .
docker build --platform linux/amd64 -f Dockerfile.worker -t ${WORKER_REPO}:latest .

echo -e "\n${YELLOW}Pushing Worker image to ECR...${NC}"
docker push ${WORKER_REPO}:latest

# Step 7: Deploy complete infrastructure
cd terraform
echo -e "\n${YELLOW}Step 7: Deploying complete infrastructure...${NC}"
terraform apply -auto-approve

# Step 8: Display outputs
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Deployment Complete!${NC}"
echo -e "${GREEN}========================================${NC}"

echo -e "\n${YELLOW}Important Information:${NC}"
echo -e "VPC ID: $(terraform output -raw vpc_id)"
echo -e "ECS Cluster: $(terraform output -raw ecs_cluster_name)"
echo -e "Parameter Server Service: $(terraform output -raw parameter_server_service_name)"
echo -e "Worker Service: $(terraform output -raw worker_service_name)"
echo -e "Parameter Server DNS: $(terraform output -raw parameter_server_dns_name)"

echo -e "\n${YELLOW}Monitoring Commands:${NC}"
echo -e "View Parameter Server logs:"
echo -e "  aws logs tail $(terraform output -raw parameter_server_log_group) --follow"
echo -e "\nView Worker logs:"
echo -e "  aws logs tail $(terraform output -raw worker_log_group) --follow"

echo -e "\n${GREEN}Deployment successful!${NC}"
echo -e "${YELLOW}Note: It may take a few minutes for services to become healthy.${NC}"
