#!/usr/bin/env python -u
"""
Master script to run all CSSA ablation experiments.

This script runs 11 experiments:
- 7 stage configurations at threshold=0.5
- Top 2 performing stage configs at thresholds {0.3, 0.7}

Logs all progress and aggregates results into summary CSV.

Usage:
    python crossattentiondet/ablations/scripts/run_cssa_ablations.py \
        --data ../RGBX_Semantic_Segmentation/data/images \
        --labels ../RGBX_Semantic_Segmentation/data/labels \
        --output-base results/cssa_ablations \
        --epochs 25 \
        --backbone mit_b1
"""

import sys
import os
import argparse
import subprocess
from datetime import datetime
import time
import json
import csv

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


class AblationRunner:
    """Manages and runs all CSSA ablation experiments."""

    def __init__(self, args):
        self.args = args
        self.output_base = args.output_base
        self.master_log_path = os.path.join(self.output_base, 'master_log.txt')
        self.summary_csv_path = os.path.join(self.output_base, 'summary_all_experiments.csv')

        # Create output directory
        os.makedirs(self.output_base, exist_ok=True)

        # Open master log
        self.master_log = open(self.master_log_path, 'w', buffering=1)

        # Track experiments
        self.experiments = []
        self.results = []
        self.start_time = None

        # Define initial experiments (7 stage configs at threshold=0.5)
        self.initial_experiments = self._define_initial_experiments()

    def _define_initial_experiments(self):
        """Define the initial 7 stage configuration experiments."""
        stage_configs = [
            ([1], "s1"),
            ([2], "s2"),
            ([3], "s3"),
            ([4], "s4"),
            ([2, 3], "s23"),
            ([3, 4], "s34"),
            ([1, 2, 3, 4], "s1234"),
        ]

        experiments = []
        for stages, stage_label in stage_configs:
            exp_id = f"exp_{len(experiments)+1:03d}_{stage_label}_t0.5"
            experiments.append({
                'id': exp_id,
                'stages': stages,
                'stage_label': stage_label,
                'threshold': 0.5,
                'kernel': 3,
                'output_dir': os.path.join(self.output_base, exp_id)
            })

        return experiments

    def _add_threshold_variants(self, top_configs):
        """Add threshold variant experiments for top performing configs."""
        threshold_experiments = []
        for rank, (stage_label, stages, _) in enumerate(top_configs[:2], 1):
            for thresh in [0.3, 0.7]:
                exp_id = f"exp_{len(self.initial_experiments) + len(threshold_experiments)+1:03d}_{stage_label}_t{thresh}"
                threshold_experiments.append({
                    'id': exp_id,
                    'stages': stages,
                    'stage_label': stage_label,
                    'threshold': thresh,
                    'kernel': 3,
                    'output_dir': os.path.join(self.output_base, exp_id)
                })

        return threshold_experiments

    def log(self, message):
        """Log to master log and console."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_msg = f"[{timestamp}] {message}"
        self.master_log.write(log_msg + '\n')
        print(log_msg)

    def log_header(self, title, width=80):
        """Log a header."""
        self.log("=" * width)
        self.log(title.center(width))
        self.log("=" * width)

    def run_experiment(self, exp):
        """Run a single experiment."""
        exp_num = self.experiments.index(exp) + 1
        total_exps = len(self.experiments)

        self.log_header(f"EXPERIMENT {exp_num}/{total_exps}: {exp['id']}")
        self.log(f"  Stages: {exp['stages']}")
        self.log(f"  Threshold: {exp['threshold']}")
        self.log(f"  Output: {exp['output_dir']}")

        # Build command
        cmd = [
            'python', '-u',
            'crossattentiondet/ablations/scripts/train_cssa_ablation.py',
            '--data', self.args.data,
            '--labels', self.args.labels,
            '--output-dir', exp['output_dir'],
            '--backbone', self.args.backbone,
            '--epochs', str(self.args.epochs),
            '--batch-size', str(self.args.batch_size),
            '--lr', str(self.args.lr),
            '--cssa-stages', ','.join(map(str, exp['stages'])),
            '--cssa-thresh', str(exp['threshold']),
            '--cssa-kernel', str(exp['kernel'])
        ]

        self.log(f"  Command: {' '.join(cmd)}")

        # Run experiment
        exp_start = time.time()
        try:
            result = subprocess.run(cmd, capture_output=False, text=True, check=True)
            status = "SUCCESS"
            self.log(f"  Status: ✓ {status}")

        except subprocess.CalledProcessError as e:
            status = "FAILED"
            self.log(f"  Status: ✗ {status}")
            self.log(f"  Error: {str(e)}")
            return None

        exp_time = (time.time() - exp_start) / 3600  # hours

        # Load results
        results_path = os.path.join(exp['output_dir'], 'final_results.json')
        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                exp_results = json.load(f)

            # Extract key metrics
            result_summary = {
                'exp_id': exp['id'],
                'stages': ','.join(map(str, exp['stages'])),
                'stage_label': exp['stage_label'],
                'threshold': exp['threshold'],
                'kernel': exp['kernel'],
                'backbone': self.args.backbone,
                'epochs': self.args.epochs,
                'mAP': exp_results['results']['mAP'],
                'mAP_50': exp_results['results']['mAP_50'],
                'mAP_75': exp_results['results']['mAP_75'],
                'mAP_small': exp_results['results']['mAP_small'],
                'mAP_medium': exp_results['results']['mAP_medium'],
                'mAP_large': exp_results['results']['mAP_large'],
                'params_M': exp_results['model']['total_params_M'],
                'time_hr': exp_time,
                'status': status
            }

            self.log(f"  Results: mAP={result_summary['mAP']:.4f}, AP50={result_summary['mAP_50']:.4f}, AP75={result_summary['mAP_75']:.4f}")
            self.log(f"  Time: {exp_time:.2f} hours")

            return result_summary
        else:
            self.log(f"  Warning: Results file not found at {results_path}")
            return None

    def update_summary_csv(self, result):
        """Update summary CSV with new result."""
        if result is None:
            return

        # Create CSV if doesn't exist
        file_exists = os.path.exists(self.summary_csv_path)

        with open(self.summary_csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=result.keys())

            if not file_exists:
                writer.writeheader()

            writer.writerow(result)

    def get_top_stage_configs(self):
        """Analyze results and return top 2 stage configurations by mAP."""
        # Read summary CSV
        if not os.path.exists(self.summary_csv_path):
            self.log("Warning: Summary CSV not found, cannot determine top configs")
            return []

        results = []
        with open(self.summary_csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                results.append(row)

        # Sort by mAP (descending)
        results.sort(key=lambda x: float(x['mAP']), reverse=True)

        # Get top 2 unique stage configurations
        top_configs = []
        seen_stages = set()

        for r in results:
            stages_str = r['stages']
            if stages_str not in seen_stages:
                seen_stages.add(stages_str)
                # Convert back to list
                stages = [int(s.strip()) for s in stages_str.split(',')]
                top_configs.append((r['stage_label'], stages, float(r['mAP'])))

            if len(top_configs) >= 2:
                break

        self.log("\nTop 2 stage configurations by mAP:")
        for rank, (label, stages, mAP) in enumerate(top_configs, 1):
            self.log(f"  {rank}. {label} (stages {stages}): mAP = {mAP:.4f}")

        return top_configs

    def run_all_experiments(self):
        """Run all experiments: initial 7 + threshold variants for top 2."""
        self.start_time = time.time()

        self.log_header("CSSA ABLATION STUDY - MASTER RUNNER")
        self.log(f"Data: {self.args.data}")
        self.log(f"Labels: {self.args.labels}")
        self.log(f"Output: {self.output_base}")
        self.log(f"Backbone: {self.args.backbone}")
        self.log(f"Epochs: {self.args.epochs}")
        self.log(f"Batch size: {self.args.batch_size}")
        self.log("")

        # Phase 1: Run initial 7 experiments
        self.log_header("PHASE 1: STAGE CONFIGURATION EXPERIMENTS (7 experiments)")
        self.log(f"Testing stages: {[exp['stage_label'] for exp in self.initial_experiments]}")
        self.log(f"Threshold: 0.5 (default)")
        self.log("")

        self.experiments = self.initial_experiments.copy()

        for exp in self.experiments:
            result = self.run_experiment(exp)
            if result:
                self.results.append(result)
                self.update_summary_csv(result)

            # Log progress
            elapsed = (time.time() - self.start_time) / 3600
            completed = len(self.results)
            avg_time_per_exp = elapsed / completed if completed > 0 else 0
            remaining = len(self.experiments) - completed
            est_remaining = avg_time_per_exp * remaining

            self.log(f"\nProgress: {completed}/{len(self.experiments)} complete")
            self.log(f"Elapsed: {elapsed:.2f} hours | Est. remaining: {est_remaining:.2f} hours")
            self.log("")

        # Phase 2: Threshold variants for top 2 configs
        self.log_header("PHASE 2: THRESHOLD SENSITIVITY (4 experiments)")
        self.log("Analyzing Phase 1 results to determine top 2 stage configurations...")

        top_configs = self.get_top_stage_configs()

        if len(top_configs) >= 2:
            threshold_exps = self._add_threshold_variants(top_configs)
            self.experiments.extend(threshold_exps)

            self.log(f"\nTesting thresholds {{0.3, 0.7}} for top 2 configs")
            self.log("")

            for exp in threshold_exps:
                result = self.run_experiment(exp)
                if result:
                    self.results.append(result)
                    self.update_summary_csv(result)

                # Log progress
                elapsed = (time.time() - self.start_time) / 3600
                completed = len(self.results)
                self.log(f"\nProgress: {completed}/{len(self.experiments)} complete")
                self.log(f"Total elapsed: {elapsed:.2f} hours")
                self.log("")
        else:
            self.log("Warning: Could not determine top 2 configs, skipping Phase 2")

        # Final summary
        total_time = (time.time() - self.start_time) / 3600
        self.log_header("ABLATION STUDY COMPLETE")
        self.log(f"Total experiments: {len(self.results)}")
        self.log(f"Successful: {sum(1 for r in self.results if r['status'] == 'SUCCESS')}")
        self.log(f"Failed: {sum(1 for r in self.results if r['status'] != 'SUCCESS')}")
        self.log(f"Total time: {total_time:.2f} hours ({total_time/24:.2f} days)")
        self.log(f"\nResults saved to: {self.output_base}")
        self.log(f"Summary CSV: {self.summary_csv_path}")
        self.log(f"Master log: {self.master_log_path}")

    def close(self):
        """Close log files."""
        if hasattr(self, 'master_log'):
            self.master_log.close()


def main():
    parser = argparse.ArgumentParser(description='Run CSSA Ablation Experiments')

    # Dataset paths
    parser.add_argument('--data', type=str, required=True, help='Path to image data directory')
    parser.add_argument('--labels', type=str, required=True, help='Path to labels directory')
    parser.add_argument('--output-base', type=str, default='results/cssa_ablations',
                        help='Base directory for all experiment outputs')

    # Training parameters
    parser.add_argument('--backbone', type=str, default='mit_b1',
                        choices=['mit_b0', 'mit_b1', 'mit_b2', 'mit_b4', 'mit_b5'],
                        help='Backbone architecture')
    parser.add_argument('--epochs', type=int, default=25, help='Number of epochs per experiment')
    parser.add_argument('--batch-size', type=int, default=2, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate')

    args = parser.parse_args()

    # Create and run ablation runner
    runner = AblationRunner(args)

    try:
        runner.run_all_experiments()
    except KeyboardInterrupt:
        runner.log("\n\nAblation study interrupted by user")
        runner.log(f"Completed {len(runner.results)}/{len(runner.experiments)} experiments")
    except Exception as e:
        runner.log(f"\n\nERROR: {str(e)}")
        import traceback
        runner.log(traceback.format_exc())
        raise
    finally:
        runner.close()


if __name__ == '__main__':
    main()
