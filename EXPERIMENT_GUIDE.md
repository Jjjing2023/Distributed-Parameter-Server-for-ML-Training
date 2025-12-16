# Experiment Guide: Distributed Parameter Server Performance Analysis

This guide walks you through running experiments to compare sync vs async modes and different worker configurations.

## Overview

The experiment framework allows you to:
- Compare **sync vs async** parameter server modes
- Test different **worker counts** (1, 2, 4, 8, 16, 32)
- Collect metrics: **epoch duration**, **test accuracy**, **throughput**
- Visualize results with comparison plots

## Quick Start: 4 Workers Sync vs Async

For a quick comparison of sync vs async with 4 workers:

### 1. Deploy Sync Mode
```bash
cd terraform

# Build and push updated images first (if you haven't already)
cd ..
./terraform/deploy.sh  # This builds and pushes images

cd terraform

# Deploy with sync mode
terraform apply \
  -var 'server_mode=sync' \
  -var 'worker_count=4' \
  -auto-approve
```

### 2. Monitor Training
```bash
# Watch server logs
aws logs tail $(terraform output -raw parameter_server_log_group) --follow

# In another terminal, watch worker logs
aws logs tail $(terraform output -raw worker_log_group) --follow
```

Wait for training to complete (watch for "METRICS_JSON" in logs).

### 3. Collect Sync Results
```bash
cd ..
mkdir -p experiment_results

python scripts/parse_cloudwatch_logs.py \
  --experiment-name sync_4workers \
  --use-terraform \
  --terraform-dir ./terraform \
  --output experiment_results/sync_4workers.json \
  --hours-ago 2
```

### 4. Destroy and Deploy Async Mode
```bash
cd terraform
terraform destroy -auto-approve

# Deploy with async mode
terraform apply \
  -var 'server_mode=async' \
  -var 'worker_count=4' \
  -auto-approve
```

### 5. Collect Async Results
Wait for training to complete, then:
```bash
cd ..
python scripts/parse_cloudwatch_logs.py \
  --experiment-name async_4workers \
  --use-terraform \
  --terraform-dir ./terraform \
  --output experiment_results/async_4workers.json \
  --hours-ago 2
```

### 6. Visualize Results
```bash
python scripts/visualize_results.py \
  --results-dir ./experiment_results \
  --output-dir ./plots
```

Check the `./plots` directory for comparison graphs!

## Full Experiment Suite

To run a comprehensive comparison across different worker counts:

### Recommended Experiment Matrix

| Experiment | Mode | Workers | Priority |
|------------|------|---------|----------|
| sync_1workers | sync | 1 | Optional (baseline) |
| async_1workers | async | 1 | Optional (baseline) |
| sync_2workers | sync | 2 | Medium |
| async_2workers | async | 2 | Medium |
| **sync_4workers** | **sync** | **4** | **High** |
| **async_4workers** | **async** | **4** | **High** |
| sync_8workers | sync | 8 | Medium |
| async_8workers | async | 8 | Medium |
| sync_16workers | sync | 16 | Optional |
| async_16workers | async | 16 | Optional |
| sync_32workers | sync | 32 | Optional |
| async_32workers | async | 32 | Optional |

### Running Each Experiment

For each experiment in the table:

```bash
cd terraform

# Deploy
terraform apply \
  -var 'server_mode=<MODE>' \
  -var 'worker_count=<WORKERS>' \
  -auto-approve

# Wait for completion (monitor logs)
aws logs tail $(terraform output -raw parameter_server_log_group) --follow

# Collect results
cd ..
python scripts/parse_cloudwatch_logs.py \
  --experiment-name <EXPERIMENT_NAME> \
  --use-terraform \
  --output experiment_results/<EXPERIMENT_NAME>.json \
  --hours-ago 2

# Clean up
cd terraform
terraform destroy -auto-approve

# Wait a few minutes before next deployment
```

## Understanding the Metrics

### Server Metrics
The parameter server tracks:
- `mode`: sync or async
- `total_workers`: number of workers
- `total_training_time_seconds`: end-to-end training time
- `global_steps_completed`: total parameter updates
- `gradients_processed`: total gradients received
- `updates_per_second`: throughput
- (async only) `average_gradient_staleness`: avg staleness of gradients

### Worker Metrics (Aggregated)
The workers track:
- `average_epoch_time_seconds`: avg time per epoch across workers
- `final_test_accuracy_percent`: model accuracy on test set
- `epoch_times_by_epoch`: time for each epoch (max/avg/min across workers)
- `accuracy_by_epoch`: accuracy progression over epochs

### Example Metrics Output

After parsing, you'll see results like:
```json
{
  "experiment_name": "sync_4workers",
  "server_metrics": {
    "mode": "sync",
    "total_workers": 4,
    "total_training_time_seconds": 245.67,
    "global_steps_completed": 150,
    "updates_per_second": 0.61
  },
  "worker_metrics_aggregated": {
    "num_workers": 4,
    "average_epoch_time_seconds": 78.5,
    "final_test_accuracy_percent": 45.23,
    "epoch_times_by_epoch": [
      {"epoch": 1, "max_time": 82.1, "avg_time": 80.3},
      {"epoch": 2, "max_time": 79.8, "avg_time": 78.1},
      {"epoch": 3, "max_time": 76.2, "avg_time": 74.5}
    ]
  }
}
```

## Visualization Outputs

The visualization script creates:

### 1. Sync vs Async Comparison (per worker count)
- `sync_vs_async_4workers.png`
- Shows 4 plots:
  - Total training time
  - Average epoch time
  - Epoch duration over time
  - Final test accuracy

### 2. Scaling Analysis
- `scaling_analysis.png`
- Shows 4 plots:
  - Training time vs worker count
  - Epoch time vs worker count
  - Accuracy vs worker count
  - Speedup vs worker count (with ideal linear speedup line)

### 3. Summary Table
Printed to console with all experiments:
```
Experiment                Mode     Workers  Time(s)    Epoch(s)   Accuracy(%)
--------------------------------------------------------------------------------
sync_4workers            sync     4        245.7      78.5       45.23
async_4workers           async    4        189.2      62.3       43.87
```

## Deployment Configuration

### Modifying Terraform Variables

You can adjust these in `terraform/variables.tf` or via `-var` flags:

```bash
terraform apply \
  -var 'server_mode=async' \           # sync or async
  -var 'worker_count=8' \              # 1-32 workers
  -var 'parameter_server_cpu=4096' \   # CPU units (2048 = 2 vCPU)
  -var 'parameter_server_memory=8192' \ # Memory in MB
  -var 'worker_cpu=4096' \
  -var 'worker_memory=8192' \
  -auto-approve
```

### Environment Variables Passed to Containers

The terraform configuration automatically sets:
- `SERVER_MODE`: sync or async
- `TOTAL_WORKERS_EXPECTED`: number of workers
- `SERVER_PORT`: gRPC port (default 8000)
- `PARAMETER_SERVER_ADDRESS`: load balancer DNS (for workers)

## Troubleshooting

### No metrics in logs
- Wait longer - training takes time
- Check ECS tasks are running: `aws ecs list-tasks --cluster <cluster-name>`
- Check task logs for errors

### Parse script finds no data
- Increase `--hours-ago` value
- Verify log group names: `terraform output`
- Check CloudWatch has logs: AWS Console → CloudWatch → Log Groups

### Workers not connecting
- Check security groups allow traffic between workers and server
- Verify NLB DNS name is correct
- Check server logs for connection attempts

### High costs
- Destroy resources when not in use: `terraform destroy`
- Use smaller instance types for testing
- Reduce worker count for initial tests

## Cost Optimization Tips

1. **Start small**: Run 4-worker experiments first
2. **Batch experiments**: Minimize deployments/teardowns
3. **Use spot instances**: Modify terraform to use Fargate Spot
4. **Monitor costs**: Set up AWS billing alerts

## Example Workflow

Full workflow for comparing sync vs async with 4, 8, and 16 workers:

```bash
# Create results directory
mkdir -p experiment_results plots

# Run experiments
for workers in 4 8 16; do
  for mode in sync async; do
    echo "Running: ${mode}_${workers}workers"

    # Deploy
    cd terraform
    terraform apply -var "server_mode=${mode}" -var "worker_count=${workers}" -auto-approve

    # Monitor (wait for "METRICS_JSON" in logs)
    aws logs tail $(terraform output -raw parameter_server_log_group) --follow

    # Collect results
    cd ..
    python scripts/parse_cloudwatch_logs.py \
      --experiment-name "${mode}_${workers}workers" \
      --use-terraform \
      --output "experiment_results/${mode}_${workers}workers.json" \
      --hours-ago 2

    # Clean up
    cd terraform
    terraform destroy -auto-approve

    # Wait before next deployment
    sleep 300
  done
done

# Visualize all results
python scripts/visualize_results.py \
  --results-dir ./experiment_results \
  --output-dir ./plots
```

## Next Steps

After collecting your results:

1. **Analyze plots** in `./plots` directory
2. **Compare performance**: Which mode is faster? More accurate?
3. **Scaling efficiency**: How does performance scale with workers?
4. **Write conclusions**: Document findings for your project report

Good luck with your experiments!
