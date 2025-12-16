#!/usr/bin/env python3
"""
Visualization Script for Distributed Parameter Server Experiments

This script creates comparison plots from experiment results:
- Sync vs Async mode comparisons
- Scaling analysis (different worker counts)
- Epoch duration trends
- Test accuracy comparisons

Usage:
    # Visualize all experiments in a directory
    python visualize_results.py --results-dir ./results --output-dir ./plots

    # Compare specific experiments
    python visualize_results.py \
        --experiments sync_4workers.json async_4workers.json \
        --output-dir ./plots
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

class ExperimentVisualizer:
    def __init__(self, output_dir='./plots'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.experiments = []

    def load_experiment(self, filepath):
        """Load a single experiment result"""
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Extract key info
        experiment = {
            'name': data['experiment_name'],
            'filepath': filepath,
            'data': data
        }

        if data.get('server_metrics'):
            experiment['mode'] = data['server_metrics']['mode']
            experiment['num_workers'] = data['server_metrics']['total_workers']
            experiment['training_time'] = data['server_metrics']['total_training_time_seconds']

        if data.get('worker_metrics_aggregated'):
            experiment['avg_epoch_time'] = data['worker_metrics_aggregated']['average_epoch_time_seconds']
            experiment['final_accuracy'] = data['worker_metrics_aggregated']['final_test_accuracy_percent']
            experiment['epoch_times'] = data['worker_metrics_aggregated']['epoch_times_by_epoch']
            experiment['accuracy_by_epoch'] = data['worker_metrics_aggregated']['accuracy_by_epoch']

        self.experiments.append(experiment)
        print(f"Loaded: {experiment['name']} ({experiment.get('mode', 'unknown')}, {experiment.get('num_workers', '?')} workers)")

        return experiment

    def load_experiments_from_directory(self, directory):
        """Load all experiment JSON files from a directory"""
        results_dir = Path(directory)
        json_files = list(results_dir.glob('*.json'))

        print(f"Found {len(json_files)} experiment files in {results_dir}")

        for filepath in json_files:
            try:
                self.load_experiment(filepath)
            except Exception as e:
                print(f"Warning: Could not load {filepath}: {e}")

        print(f"Successfully loaded {len(self.experiments)} experiments\n")

    def plot_sync_vs_async_comparison(self, worker_count=None):
        """Compare sync vs async modes for the same worker count"""
        # Group experiments by worker count
        by_workers = defaultdict(lambda: {'sync': None, 'async': None})

        for exp in self.experiments:
            if 'mode' in exp and 'num_workers' in exp:
                by_workers[exp['num_workers']][exp['mode']] = exp

        # If specific worker count requested, filter
        if worker_count:
            by_workers = {worker_count: by_workers.get(worker_count, {})}

        # Create comparison plots
        for num_workers, modes in sorted(by_workers.items()):
            if not modes['sync'] or not modes['async']:
                print(f"Warning: Missing sync or async data for {num_workers} workers, skipping...")
                continue

            sync_exp = modes['sync']
            async_exp = modes['async']

            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle(f'Sync vs Async Comparison ({num_workers} Workers)', fontsize=16, fontweight='bold')

            # 1. Training Time Comparison
            ax = axes[0, 0]
            modes_list = ['Sync', 'Async']
            times = [sync_exp['training_time'], async_exp['training_time']]
            colors = ['#3498db', '#e74c3c']
            bars = ax.bar(modes_list, times, color=colors, alpha=0.7, edgecolor='black')
            ax.set_ylabel('Training Time (seconds)', fontweight='bold')
            ax.set_title('Total Training Time', fontweight='bold')
            ax.grid(axis='y', alpha=0.3)

            # Add value labels on bars
            for bar, time in zip(bars, times):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{time:.1f}s',
                       ha='center', va='bottom', fontweight='bold')

            # 2. Average Epoch Time
            ax = axes[0, 1]
            epoch_times = [sync_exp['avg_epoch_time'], async_exp['avg_epoch_time']]
            bars = ax.bar(modes_list, epoch_times, color=colors, alpha=0.7, edgecolor='black')
            ax.set_ylabel('Epoch Time (seconds)', fontweight='bold')
            ax.set_title('Average Epoch Time', fontweight='bold')
            ax.grid(axis='y', alpha=0.3)

            for bar, time in zip(bars, epoch_times):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{time:.1f}s',
                       ha='center', va='bottom', fontweight='bold')

            # 3. Epoch Time by Epoch
            ax = axes[1, 0]
            if sync_exp.get('epoch_times') and async_exp.get('epoch_times'):
                sync_epochs = [e['epoch'] for e in sync_exp['epoch_times']]
                sync_max_times = [e['max_time'] for e in sync_exp['epoch_times']]
                async_epochs = [e['epoch'] for e in async_exp['epoch_times']]
                async_max_times = [e['max_time'] for e in async_exp['epoch_times']]

                ax.plot(sync_epochs, sync_max_times, 'o-', label='Sync', color=colors[0], linewidth=2, markersize=8)
                ax.plot(async_epochs, async_max_times, 's-', label='Async', color=colors[1], linewidth=2, markersize=8)
                ax.set_xlabel('Epoch', fontweight='bold')
                ax.set_ylabel('Max Epoch Time (seconds)', fontweight='bold')
                ax.set_title('Epoch Duration Over Time', fontweight='bold')
                ax.legend()
                ax.grid(alpha=0.3)

            # 4. Test Accuracy
            ax = axes[1, 1]
            accuracies = [sync_exp['final_accuracy'], async_exp['final_accuracy']]
            bars = ax.bar(modes_list, accuracies, color=colors, alpha=0.7, edgecolor='black')
            ax.set_ylabel('Test Accuracy (%)', fontweight='bold')
            ax.set_title('Final Test Accuracy', fontweight='bold')
            ax.set_ylim([0, 100])
            ax.grid(axis='y', alpha=0.3)

            for bar, acc in zip(bars, accuracies):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{acc:.2f}%',
                       ha='center', va='bottom', fontweight='bold')

            plt.tight_layout()

            # Save plot
            output_file = self.output_dir / f'sync_vs_async_{num_workers}workers.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Saved: {output_file}")
            plt.close()

    def plot_scaling_analysis(self):
        """Plot performance vs number of workers"""
        # Group by mode
        sync_exps = [e for e in self.experiments if e.get('mode') == 'sync']
        async_exps = [e for e in self.experiments if e.get('mode') == 'async']

        if not sync_exps and not async_exps:
            print("Warning: No experiments with mode information found")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Scaling Analysis: Performance vs Worker Count', fontsize=16, fontweight='bold')

        # 1. Training Time vs Workers
        ax = axes[0, 0]
        if sync_exps:
            sync_workers = sorted([e['num_workers'] for e in sync_exps])
            sync_times = [next(e['training_time'] for e in sync_exps if e['num_workers'] == w) for w in sync_workers]
            ax.plot(sync_workers, sync_times, 'o-', label='Sync', color='#3498db', linewidth=2, markersize=8)

        if async_exps:
            async_workers = sorted([e['num_workers'] for e in async_exps])
            async_times = [next(e['training_time'] for e in async_exps if e['num_workers'] == w) for w in async_workers]
            ax.plot(async_workers, async_times, 's-', label='Async', color='#e74c3c', linewidth=2, markersize=8)

        ax.set_xlabel('Number of Workers', fontweight='bold')
        ax.set_ylabel('Training Time (seconds)', fontweight='bold')
        ax.set_title('Total Training Time vs Workers', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_xscale('log', base=2)

        # 2. Epoch Time vs Workers
        ax = axes[0, 1]
        if sync_exps:
            sync_workers = sorted([e['num_workers'] for e in sync_exps])
            sync_epoch_times = [next(e['avg_epoch_time'] for e in sync_exps if e['num_workers'] == w) for w in sync_workers]
            ax.plot(sync_workers, sync_epoch_times, 'o-', label='Sync', color='#3498db', linewidth=2, markersize=8)

        if async_exps:
            async_workers = sorted([e['num_workers'] for e in async_exps])
            async_epoch_times = [next(e['avg_epoch_time'] for e in async_exps if e['num_workers'] == w) for w in async_workers]
            ax.plot(async_workers, async_epoch_times, 's-', label='Async', color='#e74c3c', linewidth=2, markersize=8)

        ax.set_xlabel('Number of Workers', fontweight='bold')
        ax.set_ylabel('Average Epoch Time (seconds)', fontweight='bold')
        ax.set_title('Epoch Time vs Workers', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_xscale('log', base=2)

        # 3. Accuracy vs Workers
        ax = axes[1, 0]
        if sync_exps:
            sync_workers = sorted([e['num_workers'] for e in sync_exps])
            sync_acc = [next(e['final_accuracy'] for e in sync_exps if e['num_workers'] == w) for w in sync_workers]
            ax.plot(sync_workers, sync_acc, 'o-', label='Sync', color='#3498db', linewidth=2, markersize=8)

        if async_exps:
            async_workers = sorted([e['num_workers'] for e in async_exps])
            async_acc = [next(e['final_accuracy'] for e in async_exps if e['num_workers'] == w) for w in async_workers]
            ax.plot(async_workers, async_acc, 's-', label='Async', color='#e74c3c', linewidth=2, markersize=8)

        ax.set_xlabel('Number of Workers', fontweight='bold')
        ax.set_ylabel('Test Accuracy (%)', fontweight='bold')
        ax.set_title('Final Accuracy vs Workers', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_xscale('log', base=2)

        # 4. Speedup vs Workers (using 1 worker as baseline)
        ax = axes[1, 1]
        if sync_exps:
            sync_workers = sorted([e['num_workers'] for e in sync_exps])
            if 1 in sync_workers:
                baseline_time = next(e['training_time'] for e in sync_exps if e['num_workers'] == 1)
                sync_speedup = [baseline_time / next(e['training_time'] for e in sync_exps if e['num_workers'] == w) for w in sync_workers]
                ax.plot(sync_workers, sync_speedup, 'o-', label='Sync', color='#3498db', linewidth=2, markersize=8)

        if async_exps:
            async_workers = sorted([e['num_workers'] for e in async_exps])
            if 1 in async_workers:
                baseline_time = next(e['training_time'] for e in async_exps if e['num_workers'] == 1)
                async_speedup = [baseline_time / next(e['training_time'] for e in async_exps if e['num_workers'] == w) for w in async_workers]
                ax.plot(async_workers, async_speedup, 's-', label='Async', color='#e74c3c', linewidth=2, markersize=8)

        # Add ideal linear speedup line
        max_workers = max([e['num_workers'] for e in self.experiments])
        ax.plot([1, max_workers], [1, max_workers], '--', color='gray', label='Linear Speedup', alpha=0.5)

        ax.set_xlabel('Number of Workers', fontweight='bold')
        ax.set_ylabel('Speedup', fontweight='bold')
        ax.set_title('Speedup vs Workers (vs 1 worker)', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_xscale('log', base=2)
        ax.set_yscale('log', base=2)

        plt.tight_layout()

        # Save plot
        output_file = self.output_dir / 'scaling_analysis.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()

    def create_summary_table(self):
        """Create a summary table of all experiments"""
        print("\n" + "="*80)
        print("EXPERIMENT SUMMARY TABLE")
        print("="*80)
        print(f"{'Experiment':<25} {'Mode':<8} {'Workers':<8} {'Time(s)':<10} {'Epoch(s)':<10} {'Accuracy(%)':<12}")
        print("-"*80)

        for exp in sorted(self.experiments, key=lambda x: (x.get('mode', ''), x.get('num_workers', 0))):
            name = exp['name'][:24]
            mode = exp.get('mode', 'N/A')
            workers = exp.get('num_workers', 'N/A')
            time = f"{exp.get('training_time', 0):.1f}" if 'training_time' in exp else 'N/A'
            epoch = f"{exp.get('avg_epoch_time', 0):.1f}" if 'avg_epoch_time' in exp else 'N/A'
            acc = f"{exp.get('final_accuracy', 0):.2f}" if 'final_accuracy' in exp else 'N/A'

            print(f"{name:<25} {mode:<8} {workers:<8} {time:<10} {epoch:<10} {acc:<12}")

        print("="*80 + "\n")

def main():
    parser = argparse.ArgumentParser(description='Visualize distributed parameter server experiment results')
    parser.add_argument('--results-dir',
                       help='Directory containing experiment JSON files')
    parser.add_argument('--experiments', nargs='+',
                       help='Specific experiment JSON files to visualize')
    parser.add_argument('--output-dir', default='./plots',
                       help='Directory to save plots (default: ./plots)')
    parser.add_argument('--compare-workers', type=int,
                       help='Create sync vs async comparison for specific worker count')

    args = parser.parse_args()

    # Create visualizer
    viz = ExperimentVisualizer(output_dir=args.output_dir)

    # Load experiments
    if args.results_dir:
        viz.load_experiments_from_directory(args.results_dir)
    elif args.experiments:
        for exp_file in args.experiments:
            viz.load_experiment(exp_file)
    else:
        print("Error: Must provide either --results-dir or --experiments")
        return 1

    if not viz.experiments:
        print("Error: No experiments loaded")
        return 1

    # Create visualizations
    print("\nGenerating visualizations...")
    print("-" * 60)

    # Summary table
    viz.create_summary_table()

    # Sync vs async comparisons
    viz.plot_sync_vs_async_comparison(worker_count=args.compare_workers)

    # Scaling analysis
    viz.plot_scaling_analysis()

    print(f"\nAll plots saved to: {viz.output_dir}")
    print("="*60)

    return 0

if __name__ == '__main__':
    exit(main())
