#!/bin/bash

# Cleanup script for Distributed Parameter Server on AWS ECS
# This script destroys all AWS resources created by Terraform

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${RED}========================================${NC}"
echo -e "${RED}Distributed Parameter Server Cleanup${NC}"
echo -e "${RED}========================================${NC}"

echo -e "\n${YELLOW}WARNING: This will destroy all AWS resources!${NC}"
echo -e "${YELLOW}This includes:${NC}"
echo -e "  - ECS Cluster and Services"
echo -e "  - ECR Repositories (and all images)"
echo -e "  - VPC and Networking resources"
echo -e "  - CloudWatch Log Groups (and all logs)"
echo -e "  - Service Discovery resources"

read -p "Are you sure you want to continue? (yes/no): " -r
echo
if [[ ! $REPLY =~ ^[Yy]es$ ]]
then
    echo -e "${GREEN}Cleanup cancelled.${NC}"
    exit 1
fi

echo -e "\n${YELLOW}Starting cleanup...${NC}"

# Destroy all resources
terraform destroy -auto-approve

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Cleanup Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}All AWS resources have been destroyed.${NC}"
