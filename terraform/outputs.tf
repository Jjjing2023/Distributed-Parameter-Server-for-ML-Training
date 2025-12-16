output "vpc_id" {
  description = "ID of the VPC"
  value       = aws_vpc.main.id
}

output "public_subnet_ids" {
  description = "IDs of public subnets"
  value       = aws_subnet.public[*].id
}

output "ecs_cluster_name" {
  description = "Name of the ECS cluster"
  value       = aws_ecs_cluster.main.name
}

output "ecs_cluster_arn" {
  description = "ARN of the ECS cluster"
  value       = aws_ecs_cluster.main.arn
}

output "parameter_server_ecr_repository_url" {
  description = "URL of the parameter server ECR repository"
  value       = aws_ecr_repository.parameter_server.repository_url
}

output "worker_ecr_repository_url" {
  description = "URL of the worker ECR repository"
  value       = aws_ecr_repository.worker.repository_url
}

output "parameter_server_service_name" {
  description = "Name of the parameter server ECS service"
  value       = aws_ecs_service.parameter_server.name
}

output "worker_service_name" {
  description = "Name of the worker ECS service"
  value       = aws_ecs_service.worker.name
}

output "parameter_server_log_group" {
  description = "CloudWatch log group for parameter server"
  value       = aws_cloudwatch_log_group.parameter_server.name
}

output "worker_log_group" {
  description = "CloudWatch log group for workers"
  value       = aws_cloudwatch_log_group.worker.name
}

output "parameter_server_nlb_dns_name" {
  description = "NLB DNS name for parameter server"
  value       = aws_lb.parameter_server.dns_name
}

output "parameter_server_dns_name" {
  description = "DNS name for parameter server (use this for worker connection)"
  value       = "${aws_lb.parameter_server.dns_name}:${var.parameter_server_port}"
}

output "security_group_parameter_server_id" {
  description = "ID of the parameter server security group"
  value       = aws_security_group.parameter_server.id
}

output "security_group_worker_id" {
  description = "ID of the worker security group"
  value       = aws_security_group.worker.id
}
