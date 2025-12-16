terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  required_version = ">= 1.0"
}

provider "aws" {
  region = var.aws_region
}

# ===== VPC and Networking =====

resource "aws_vpc" "main" {
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name        = "${var.project_name}-${var.environment}-vpc"
    Environment = var.environment
    Project     = var.project_name
  }
}

# Internet Gateway
resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id

  tags = {
    Name        = "${var.project_name}-${var.environment}-igw"
    Environment = var.environment
    Project     = var.project_name
  }
}

# Public Subnets
resource "aws_subnet" "public" {
  count                   = length(var.public_subnet_cidrs)
  vpc_id                  = aws_vpc.main.id
  cidr_block              = var.public_subnet_cidrs[count.index]
  availability_zone       = var.availability_zones[count.index]
  map_public_ip_on_launch = true

  tags = {
    Name        = "${var.project_name}-${var.environment}-public-subnet-${count.index + 1}"
    Environment = var.environment
    Project     = var.project_name
  }
}

# Route Table for Public Subnets
resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }

  tags = {
    Name        = "${var.project_name}-${var.environment}-public-rt"
    Environment = var.environment
    Project     = var.project_name
  }
}

# Route Table Associations
resource "aws_route_table_association" "public" {
  count          = length(aws_subnet.public)
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

# ===== Security Groups =====

# Security Group for Parameter Server
resource "aws_security_group" "parameter_server" {
  name        = "${var.project_name}-${var.environment}-ps-sg"
  description = "Security group for parameter server"
  vpc_id      = aws_vpc.main.id

  # Allow gRPC traffic from workers
  ingress {
    description     = "gRPC from workers"
    from_port       = var.parameter_server_port
    to_port         = var.parameter_server_port
    protocol        = "tcp"
    security_groups = [aws_security_group.worker.id]
  }

  # Allow health check traffic from NLB (within VPC)
  ingress {
    description = "Health checks from NLB"
    from_port   = var.parameter_server_port
    to_port     = var.parameter_server_port
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
  }

  # Allow all outbound traffic
  egress {
    description = "All outbound traffic"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name        = "${var.project_name}-${var.environment}-ps-sg"
    Environment = var.environment
    Project     = var.project_name
  }
}

# Security Group for Workers
resource "aws_security_group" "worker" {
  name        = "${var.project_name}-${var.environment}-worker-sg"
  description = "Security group for workers"
  vpc_id      = aws_vpc.main.id

  # Allow all outbound traffic (to reach parameter server and internet)
  egress {
    description = "All outbound traffic"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name        = "${var.project_name}-${var.environment}-worker-sg"
    Environment = var.environment
    Project     = var.project_name
  }
}

# ===== IAM Role =====

# Use existing LabRole
data "aws_iam_role" "lab_role" {
  name = "LabRole"
}

# ===== ECR Repositories =====

# ECR Repository for Parameter Server
resource "aws_ecr_repository" "parameter_server" {
  name                 = "${var.project_name}-${var.environment}-parameter-server"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = {
    Name        = "${var.project_name}-${var.environment}-parameter-server"
    Environment = var.environment
    Project     = var.project_name
  }
}

# ECR Repository for Workers
resource "aws_ecr_repository" "worker" {
  name                 = "${var.project_name}-${var.environment}-worker"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = {
    Name        = "${var.project_name}-${var.environment}-worker"
    Environment = var.environment
    Project     = var.project_name
  }
}

# ===== CloudWatch Log Groups =====

resource "aws_cloudwatch_log_group" "parameter_server" {
  name              = "/ecs/${var.project_name}-${var.environment}-parameter-server"
  retention_in_days = var.log_retention_days

  tags = {
    Name        = "${var.project_name}-${var.environment}-parameter-server-logs"
    Environment = var.environment
    Project     = var.project_name
  }
}

resource "aws_cloudwatch_log_group" "worker" {
  name              = "/ecs/${var.project_name}-${var.environment}-worker"
  retention_in_days = var.log_retention_days

  tags = {
    Name        = "${var.project_name}-${var.environment}-worker-logs"
    Environment = var.environment
    Project     = var.project_name
  }
}

# ===== ECS Cluster =====

resource "aws_ecs_cluster" "main" {
  name = "${var.project_name}-${var.environment}-cluster"

  setting {
    name  = "containerInsights"
    value = "enabled"
  }

  tags = {
    Name        = "${var.project_name}-${var.environment}-cluster"
    Environment = var.environment
    Project     = var.project_name
  }
}

# ===== ECS Task Definitions =====

# Parameter Server Task Definition
resource "aws_ecs_task_definition" "parameter_server" {
  family                   = "${var.project_name}-${var.environment}-parameter-server"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.parameter_server_cpu
  memory                   = var.parameter_server_memory
  execution_role_arn       = data.aws_iam_role.lab_role.arn
  task_role_arn            = data.aws_iam_role.lab_role.arn

  container_definitions = jsonencode([
    {
      name      = "parameter-server"
      image     = "${aws_ecr_repository.parameter_server.repository_url}:latest"
      essential = true

      portMappings = [
        {
          containerPort = var.parameter_server_port
          hostPort      = var.parameter_server_port
          protocol      = "tcp"
        }
      ]

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.parameter_server.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "ecs"
        }
      }

      environment = [
        {
          name  = "SERVER_PORT"
          value = tostring(var.parameter_server_port)
        },
        {
          name  = "TOTAL_WORKERS_EXPECTED"
          value = tostring(var.worker_count)
        },
        {
          name  = "SERVER_MODE"
          value = var.server_mode
        }
      ]
    }
  ])

  tags = {
    Name        = "${var.project_name}-${var.environment}-parameter-server-task"
    Environment = var.environment
    Project     = var.project_name
  }
}

# Worker Task Definition
resource "aws_ecs_task_definition" "worker" {
  family                   = "${var.project_name}-${var.environment}-worker"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.worker_cpu
  memory                   = var.worker_memory
  execution_role_arn       = data.aws_iam_role.lab_role.arn
  task_role_arn            = data.aws_iam_role.lab_role.arn

  container_definitions = jsonencode([
    {
      name      = "worker"
      image     = "${aws_ecr_repository.worker.repository_url}:latest"
      essential = true

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.worker.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "ecs"
        }
      }

      environment = [
        {
          name  = "PARAMETER_SERVER_ADDRESS"
          value = "${aws_lb.parameter_server.dns_name}:${var.parameter_server_port}"
        }
      ]
    }
  ])

  tags = {
    Name        = "${var.project_name}-${var.environment}-worker-task"
    Environment = var.environment
    Project     = var.project_name
  }
}

# ===== Network Load Balancer for Parameter Server =====

# NLB for Parameter Server
resource "aws_lb" "parameter_server" {
  name               = "${var.project_name}-${var.environment}-ps-nlb"
  internal           = true
  load_balancer_type = "network"
  subnets            = aws_subnet.public[*].id

  enable_deletion_protection = false

  tags = {
    Name        = "${var.project_name}-${var.environment}-ps-nlb"
    Environment = var.environment
    Project     = var.project_name
  }
}

# Target Group for Parameter Server
resource "aws_lb_target_group" "parameter_server" {
  name        = "${var.project_name}-${var.environment}-ps-tg"
  port        = var.parameter_server_port
  protocol    = "TCP"
  vpc_id      = aws_vpc.main.id
  target_type = "ip"

  health_check {
    enabled             = true
    protocol            = "TCP"
    port                = "traffic-port"
    healthy_threshold   = 2
    unhealthy_threshold = 2
    interval            = 30
  }

  tags = {
    Name        = "${var.project_name}-${var.environment}-ps-tg"
    Environment = var.environment
    Project     = var.project_name
  }
}

# NLB Listener
resource "aws_lb_listener" "parameter_server" {
  load_balancer_arn = aws_lb.parameter_server.arn
  port              = var.parameter_server_port
  protocol          = "TCP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.parameter_server.arn
  }

  tags = {
    Name        = "${var.project_name}-${var.environment}-ps-listener"
    Environment = var.environment
    Project     = var.project_name
  }
}

# ===== ECS Services =====

# Parameter Server Service
resource "aws_ecs_service" "parameter_server" {
  name            = "${var.project_name}-${var.environment}-parameter-server"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.parameter_server.arn
  desired_count   = 1
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = aws_subnet.public[*].id
    security_groups  = [aws_security_group.parameter_server.id]
    assign_public_ip = true
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.parameter_server.arn
    container_name   = "parameter-server"
    container_port   = var.parameter_server_port
  }

  tags = {
    Name        = "${var.project_name}-${var.environment}-parameter-server-service"
    Environment = var.environment
    Project     = var.project_name
  }
}

# Worker Service
resource "aws_ecs_service" "worker" {
  name            = "${var.project_name}-${var.environment}-worker"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.worker.arn
  desired_count   = var.worker_count
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = aws_subnet.public[*].id
    security_groups  = [aws_security_group.worker.id]
    assign_public_ip = true
  }

  # Ensure parameter server is running before starting workers
  depends_on = [aws_ecs_service.parameter_server]

  tags = {
    Name        = "${var.project_name}-${var.environment}-worker-service"
    Environment = var.environment
    Project     = var.project_name
  }
}
