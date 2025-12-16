variable "aws_region" {
  description = "AWS region to deploy resources"
  type        = string
  default     = "us-west-2"
}

variable "project_name" {
  description = "Project name for resource naming"
  type        = string
  default     = "distributed-ps"
}

variable "environment" {
  description = "Environment name (e.g., dev, prod)"
  type        = string
  default     = "dev"
}

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "public_subnet_cidrs" {
  description = "CIDR blocks for public subnets"
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24"]
}

variable "availability_zones" {
  description = "Availability zones for subnets"
  type        = list(string)
  default     = ["us-west-2a", "us-west-2b"]
}

variable "parameter_server_port" {
  description = "Port for parameter server gRPC communication"
  type        = number
  default     = 8000
}

variable "parameter_server_cpu" {
  description = "CPU units for parameter server task (1024 = 1 vCPU)"
  type        = number
  default     = 2048
}

variable "parameter_server_memory" {
  description = "Memory for parameter server task in MB"
  type        = number
  default     = 4096
}

variable "worker_cpu" {
  description = "CPU units for worker task (1024 = 1 vCPU)"
  type        = number
  default     = 2048
}

variable "worker_memory" {
  description = "Memory for worker task in MB"
  type        = number
  default     = 4096
}

variable "worker_count" {
  description = "Number of worker instances to run"
  type        = number
  default     = 4
}

variable "log_retention_days" {
  description = "CloudWatch log retention in days"
  type        = number
  default     = 7
}

variable "server_mode" {
  description = "Parameter server mode: 'sync' or 'async'"
  type        = string
  default     = "sync"

  validation {
    condition     = contains(["sync", "async"], var.server_mode)
    error_message = "Server mode must be either 'sync' or 'async'."
  }
}
