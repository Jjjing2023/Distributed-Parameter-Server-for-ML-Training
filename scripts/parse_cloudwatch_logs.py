#!/usr/bin/env python3
"""
CloudWatch Log Parser for Distributed Parameter Server Experiments

This script downloads and parses CloudWatch logs to extract structured metrics
from both parameter server and worker logs.

Usage:
    # Parse logs for a specific experiment
    python parse_cloudwatch_logs.py \
        --experiment-name sync_4workers \
        --server-log-group /ecs/distributed-ps-dev-parameter-server \
        --worker-log-group /ecs/distributed-ps-dev-worker \
        --output results/sync_4workers.json

    # Using terraform outputs
    python parse_cloudwatch_logs.py \
        --experiment-name sync_4workers \
        --use-terraform \
        --output results/sync_4workers.json
"""

import json
import argparse
import re
import subprocess
from pathlib import Path
from datetime import datetime, timedelta

class CloudWatchLogParser:
    def __init__(self, region='us-west-2'):
        self.region = region

    def get_terraform_outputs(self, terraform_dir='./terraform'):
        """Get log group names from terraform outputs"""
        try:
            result = subprocess.run(
                ['terraform', 'output', '-json'],
                cwd=terraform_dir,
                capture_output=True,
                text=True,
                check=True
            )
            outputs = json.loads(result.stdout)

            server_log_group = outputs['parameter_server_log_group']['value']
            worker_log_group = outputs['worker_log_group']['value']

            print(f"Server log group: {server_log_group}")
            print(f"Worker log group: {worker_log_group}")

            return server_log_group, worker_log_group
        except Exception as e:
            print(f"Error getting terraform outputs: {e}")
            return None, None

    def download_logs(self, log_group, start_time=None, end_time=None, filter_pattern=None):
        """Download logs from CloudWatch"""
        print(f"Downloading logs from {log_group}...")

        cmd = [
            'aws', 'logs', 'filter-log-events',
            '--log-group-name', log_group,
            '--region', self.region,
            '--output', 'json'
        ]

        if start_time:
            # Convert to milliseconds since epoch
            start_ms = int(start_time.timestamp() * 1000)
            cmd.extend(['--start-time', str(start_ms)])

        if end_time:
            end_ms = int(end_time.timestamp() * 1000)
            cmd.extend(['--end-time', str(end_ms)])

        if filter_pattern:
            cmd.extend(['--filter-pattern', filter_pattern])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)
            print(f"  Downloaded {len(data.get('events', []))} log events")
            return data.get('events', [])
        except subprocess.CalledProcessError as e:
            print(f"Error downloading logs: {e.stderr}")
            return []
        except Exception as e:
            print(f"Error: {e}")
            return []

    def parse_metrics_from_logs(self, events):
        """Extract METRICS_JSON from log events"""
        metrics = {
            'server_metrics': None,
            'worker_metrics': []
        }

        # Pattern to match METRICS_JSON: {...}
        metrics_pattern = re.compile(r'METRICS_JSON:\s*(\{.*\})')

        for event in events:
            message = event.get('message', '')

            # Look for METRICS_JSON lines
            match = metrics_pattern.search(message)
            if match:
                try:
                    metric_data = json.loads(match.group(1))

                    if metric_data.get('type') == 'SERVER_FINAL_METRICS':
                        metrics['server_metrics'] = metric_data
                        print(f"  Found server metrics: {metric_data['mode']} mode, {metric_data['total_workers']} workers")

                    elif metric_data.get('type') == 'WORKER_FINAL_METRICS':
                        metrics['worker_metrics'].append(metric_data)
                        print(f"  Found worker {metric_data['worker_id']} metrics")

                except json.JSONDecodeError as e:
                    print(f"  Warning: Failed to parse JSON: {e}")
                    continue

        return metrics

    def aggregate_worker_metrics(self, worker_metrics):
        """Aggregate metrics across all workers"""
        if not worker_metrics:
            return None

        aggregated = {
            'num_workers': len(worker_metrics),
            'total_training_time_seconds': max(w['total_training_time_seconds'] for w in worker_metrics),
            'average_epoch_time_seconds': sum(w['average_epoch_time_seconds'] for w in worker_metrics) / len(worker_metrics),
            'final_test_accuracy_percent': sum(w['final_test_accuracy_percent'] for w in worker_metrics) / len(worker_metrics),
            'total_local_steps': sum(w['local_steps_completed'] for w in worker_metrics),
            'per_worker_metrics': worker_metrics
        }

        # Aggregate epoch times across workers
        max_epochs = max(len(w['epoch_times_seconds']) for w in worker_metrics)
        epoch_times_by_epoch = []

        for epoch_idx in range(max_epochs):
            epoch_times = [
                w['epoch_times_seconds'][epoch_idx]
                for w in worker_metrics
                if epoch_idx < len(w['epoch_times_seconds'])
            ]
            if epoch_times:
                epoch_times_by_epoch.append({
                    'epoch': epoch_idx + 1,
                    'max_time': max(epoch_times),
                    'avg_time': sum(epoch_times) / len(epoch_times),
                    'min_time': min(epoch_times)
                })

        aggregated['epoch_times_by_epoch'] = epoch_times_by_epoch

        # Aggregate accuracies
        accuracy_by_epoch = []
        for epoch_idx in range(max_epochs):
            accuracies = [
                w['all_accuracies_percent'][epoch_idx]
                for w in worker_metrics
                if epoch_idx < len(w['all_accuracies_percent'])
            ]
            if accuracies:
                accuracy_by_epoch.append({
                    'epoch': epoch_idx + 1,
                    'avg_accuracy': sum(accuracies) / len(accuracies),
                    'max_accuracy': max(accuracies),
                    'min_accuracy': min(accuracies)
                })

        aggregated['accuracy_by_epoch'] = accuracy_by_epoch

        return aggregated

    def parse_experiment(self, experiment_name, server_log_group, worker_log_group,
                        start_time=None, end_time=None):
        """Parse logs for a complete experiment"""
        print(f"\nParsing experiment: {experiment_name}")
        print("="*60)

        # Download logs
        server_events = self.download_logs(server_log_group, start_time, end_time)
        worker_events = self.download_logs(worker_log_group, start_time, end_time)

        # Parse metrics
        server_metrics = self.parse_metrics_from_logs(server_events)
        worker_metrics = self.parse_metrics_from_logs(worker_events)

        # Aggregate worker metrics
        aggregated_workers = self.aggregate_worker_metrics(worker_metrics['worker_metrics'])

        # Combine results
        results = {
            'experiment_name': experiment_name,
            'timestamp': datetime.now().isoformat(),
            'server_metrics': server_metrics['server_metrics'],
            'worker_metrics_aggregated': aggregated_workers,
            'raw_worker_metrics': worker_metrics['worker_metrics']
        }

        # Print summary
        print(f"\n{'='*60}")
        print("Summary:")
        if results['server_metrics']:
            print(f"  Mode: {results['server_metrics']['mode']}")
            print(f"  Workers: {results['server_metrics']['total_workers']}")
            print(f"  Training time: {results['server_metrics']['total_training_time_seconds']:.1f}s")
        if aggregated_workers:
            print(f"  Avg epoch time: {aggregated_workers['average_epoch_time_seconds']:.1f}s")
            print(f"  Final accuracy: {aggregated_workers['final_test_accuracy_percent']:.2f}%")
        print(f"{'='*60}\n")

        return results

def main():
    parser = argparse.ArgumentParser(description='Parse CloudWatch logs for experiment metrics')
    parser.add_argument('--experiment-name', required=True,
                       help='Name of the experiment (e.g., sync_4workers)')
    parser.add_argument('--server-log-group',
                       help='CloudWatch log group for parameter server')
    parser.add_argument('--worker-log-group',
                       help='CloudWatch log group for workers')
    parser.add_argument('--use-terraform', action='store_true',
                       help='Get log group names from terraform outputs')
    parser.add_argument('--terraform-dir', default='./terraform',
                       help='Terraform directory (default: ./terraform)')
    parser.add_argument('--output', required=True,
                       help='Output JSON file for results')
    parser.add_argument('--region', default='us-west-2',
                       help='AWS region (default: us-west-2)')
    parser.add_argument('--hours-ago', type=float,
                       help='Only get logs from last N hours (can use decimals like 0.5 for 30 min)')

    args = parser.parse_args()

    # Create parser
    log_parser = CloudWatchLogParser(region=args.region)

    # Get log group names
    if args.use_terraform:
        server_log_group, worker_log_group = log_parser.get_terraform_outputs(args.terraform_dir)
        if not server_log_group or not worker_log_group:
            print("Error: Could not get log groups from terraform")
            return 1
    else:
        server_log_group = args.server_log_group
        worker_log_group = args.worker_log_group

    if not server_log_group or not worker_log_group:
        print("Error: Must provide log group names or use --use-terraform")
        return 1

    # Calculate time range
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=args.hours_ago) if args.hours_ago else None

    # Parse experiment
    results = log_parser.parse_experiment(
        args.experiment_name,
        server_log_group,
        worker_log_group,
        start_time,
        end_time
    )

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {output_path}")

    return 0

if __name__ == '__main__':
    exit(main())
