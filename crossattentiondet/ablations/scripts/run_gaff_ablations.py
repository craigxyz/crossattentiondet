#!/usr/bin/env python -u
"""
Master script to run all GAFF ablation experiments.

This script runs experiments in two phases:
Phase 1: Test all stage configurations with default hyperparameters (8 experiments)
Phase 2: Test hyperparameter variants on top 3 stage configs (24 experiments)

Total: 32 experiments

Logs all progress and aggregates results into summary CSV.

Usage:
    python crossattentiondet/ablations/scripts/run_gaff_ablations.py \
        --data ../RGBX_Semantic_Segmentation/data/images \
        --labels ../RGBX_Semantic_Segmentation/data/labels \
        --output-base results/gaff_ablations \
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


class GAFFAblationRunner:
    """Manages and runs all GAFF ablation experiments."""

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

        # Define phase 1 experiments (stage selection with default hyperparams)
        self.phase1_experiments = self._define_phase1_experiments()

    def _define_phase1_experiments(self):
        """Define Phase 1: Stage configuration experiments with default hyperparameters."""
        stage_configs = [
            ([1], "s1"),
            ([2], "s2"),
            ([3], "s3"),
            ([4], "s4"),
            ([2, 3], "s23"),
            ([3, 4], "s34"),
            ([2, 3, 4], "s234"),
            ([1, 2, 3, 4], "s1234"),
        ]

        experiments = []
        for stages, stage_label in stage_configs:
            exp_id = f"exp_{len(experiments)+1:03d}_{stage_label}_r4_is0_mb0"
            experiments.append({
                'id': exp_id,
                'stages': stages,
                'stage_label': stage_label,
                'se_reduction': 4,
                'inter_shared': False,
                'merge_bottleneck': False,
                'phase': 1,
                'output_dir': os.path.join(self.output_base, 'phase1_stage_selection', exp_id)
            })

        return experiments

    def _define_phase2_experiments(self, top_stage_configs):
        """Define Phase 2: Hyperparameter tuning on top stage configs."""
        # Hyperparameter grid
        se_reductions = [4, 8]
        inter_shared_opts = [False, True]
        merge_bottleneck_opts = [False, True]

        experiments = []
        exp_num = len(self.phase1_experiments) + 1

        for stage_label, stages, _ in top_stage_configs[:3]:  # Top 3 configs
            for se_r in se_reductions:
                for inter_s in inter_shared_opts:
                    for merge_b in merge_bottleneck_opts:
                        # Skip the default config (already tested in phase 1)
                        if se_r == 4 and not inter_s and not merge_b:
                            continue

                        exp_id = (f"exp_{exp_num:03d}_{stage_label}_"
                                  f"r{se_r}_is{int(inter_s)}_mb{int(merge_b)}")

                        experiments.append({
                            'id': exp_id,
                            'stages': stages,
                            'stage_label': stage_label,
                            'se_reduction': se_r,
                            'inter_shared': inter_s,
                            'merge_bottleneck': merge_b,
                            'phase': 2,
                            'output_dir': os.path.join(self.output_base, 'phase2_hyperparameter_tuning', exp_id)
                        })
                        exp_num += 1

        return experiments

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
        self.log(f"  Phase: {exp['phase']}")
        self.log(f"  Stages: {exp['stages']}")
        self.log(f"  SE reduction: {exp['se_reduction']}")
        self.log(f"  Inter-modality shared: {exp['inter_shared']}")
        self.log(f"  Merge bottleneck: {exp['merge_bottleneck']}")
        self.log(f"  Output: {exp['output_dir']}")

        # Build command
        cmd = [
            'python', '-u',
            'crossattentiondet/ablations/scripts/train_gaff_ablation.py',
            '--data', self.args.data,
            '--labels', self.args.labels,
            '--output-dir', exp['output_dir'],
            '--backbone', self.args.backbone,
            '--epochs', str(self.args.epochs),
            '--batch-size', str(self.args.batch_size),
            '--lr', str(self.args.lr),
            '--gaff-stages', ','.join(map(str, exp['stages'])),
            '--gaff-se-reduction', str(exp['se_reduction']),
            '--gaff-inter-shared', 'true' if exp['inter_shared'] else 'false',
            '--gaff-merge-bottleneck', 'true' if exp['merge_bottleneck'] else 'false'
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
            self.log(f"  Error: {e}")

        except Exception as e:
            status = "ERROR"
            self.log(f"  Status: ✗ {status}")
            self.log(f"  Error: {e}")

        exp_time = (time.time() - exp_start) / 3600  # hours
        self.log(f"  Time: {exp_time:.2f} hours")

        # Collect results
        result_data = self._collect_results(exp, status, exp_time)
        self.results.append(result_data)

        return status, exp_time

    def _collect_results(self, exp, status, exp_time):
        """Collect results from experiment output directory."""
        result_data = {
            'experiment_id': exp['id'],
            'phase': exp['phase'],
            'stages': '-'.join(map(str, exp['stages'])),
            'se_reduction': exp['se_reduction'],
            'inter_shared': exp['inter_shared'],
            'merge_bottleneck': exp['merge_bottleneck'],
            'status': status,
            'time_hours': exp_time
        }

        # Try to load final results
        results_json_path = os.path.join(exp['output_dir'], 'final_results.json')
        if os.path.exists(results_json_path):
            try:
                with open(results_json_path, 'r') as f:
                    results_json = json.load(f)

                result_data.update({
                    'mAP': results_json.get('results', {}).get('mAP', None),
                    'mAP_50': results_json.get('results', {}).get('mAP_50', None),
                    'mAP_75': results_json.get('results', {}).get('mAP_75', None),
                    'final_train_loss': results_json.get('training', {}).get('final_train_loss', None),
                    'total_params_M': results_json.get('model', {}).get('total_params_M', None)
                })
            except Exception as e:
                self.log(f"  Warning: Could not load results JSON: {e}")

        return result_data

    def save_summary_csv(self):
        """Save summary CSV of all experiments."""
        if not self.results:
            return

        fieldnames = [
            'experiment_id', 'phase', 'stages', 'se_reduction', 'inter_shared', 'merge_bottleneck',
            'status', 'time_hours', 'mAP', 'mAP_50', 'mAP_75', 'final_train_loss', 'total_params_M'
        ]

        with open(self.summary_csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.results)

        self.log(f"Summary CSV saved to: {self.summary_csv_path}")

    def get_top_stage_configs(self, k=3):
        """Get top k stage configurations by mAP from Phase 1."""
        phase1_results = [r for r in self.results if r['phase'] == 1 and r['status'] == 'SUCCESS']

        # Sort by mAP (descending)
        phase1_results.sort(key=lambda x: x.get('mAP', 0), reverse=True)

        # Extract stage configurations
        top_configs = []
        for r in phase1_results[:k]:
            # Find matching experiment
            for exp in self.phase1_experiments:
                if exp['id'] == r['experiment_id']:
                    top_configs.append((
                        exp['stage_label'],
                        exp['stages'],
                        r.get('mAP', 0)
                    ))
                    break

        return top_configs

    def run_all(self):
        """Run all ablation experiments."""
        self.start_time = time.time()
        self.log_header("GAFF ABLATION STUDY START")
        self.log(f"Output base directory: {self.output_base}")
        self.log(f"Backbone: {self.args.backbone}")
        self.log(f"Epochs per experiment: {self.args.epochs}")
        self.log(f"Batch size: {self.args.batch_size}")
        self.log("")

        # Phase 1: Stage selection
        self.log_header("PHASE 1: STAGE SELECTION")
        self.log(f"Running {len(self.phase1_experiments)} experiments with default hyperparameters")
        self.log("Default config: SE_reduction=4, inter_shared=False, merge_bottleneck=False")
        self.log("")

        self.experiments = self.phase1_experiments.copy()
        for exp in self.phase1_experiments:
            self.run_experiment(exp)
            self.save_summary_csv()  # Save after each experiment

        # Analyze Phase 1 results
        self.log_header("PHASE 1 COMPLETE - ANALYZING RESULTS")
        top_stage_configs = self.get_top_stage_configs(k=3)

        if not top_stage_configs:
            self.log("WARNING: No successful experiments in Phase 1. Cannot proceed to Phase 2.")
            self.log_header("ABLATION STUDY INCOMPLETE")
            return

        self.log("Top 3 stage configurations:")
        for rank, (stage_label, stages, mAP) in enumerate(top_stage_configs, 1):
            self.log(f"  {rank}. {stage_label} (stages {stages}): mAP={mAP:.4f}")
        self.log("")

        # Phase 2: Hyperparameter tuning
        self.log_header("PHASE 2: HYPERPARAMETER TUNING")
        phase2_experiments = self._define_phase2_experiments(top_stage_configs)
        self.log(f"Running {len(phase2_experiments)} hyperparameter variants")
        self.log(f"Testing on top 3 stage configs: {[label for label, _, _ in top_stage_configs]}")
        self.log("Hyperparameter grid: SE_reduction∈{4,8}, inter_shared∈{T,F}, merge_bottleneck∈{T,F}")
        self.log("")

        self.experiments.extend(phase2_experiments)
        for exp in phase2_experiments:
            self.run_experiment(exp)
            self.save_summary_csv()  # Save after each experiment

        # Final summary
        total_time = (time.time() - self.start_time) / 3600
        self.log_header("ABLATION STUDY COMPLETE")
        self.log(f"Total experiments: {len(self.experiments)}")
        self.log(f"Phase 1: {len(self.phase1_experiments)}")
        self.log(f"Phase 2: {len(phase2_experiments)}")
        self.log(f"Total time: {total_time:.2f} hours ({total_time/24:.2f} days)")

        successful = sum(1 for r in self.results if r['status'] == 'SUCCESS')
        failed = len(self.results) - successful
        self.log(f"Success rate: {successful}/{len(self.results)} ({successful/len(self.results)*100:.1f}%)")

        self.log("")
        self.log(f"Summary CSV: {self.summary_csv_path}")
        self.log(f"Master log: {self.master_log_path}")
        self.log("")

        # Find best overall config
        successful_results = [r for r in self.results if r['status'] == 'SUCCESS' and r.get('mAP')]
        if successful_results:
            best = max(successful_results, key=lambda x: x.get('mAP', 0))
            self.log("Best configuration:")
            self.log(f"  Experiment: {best['experiment_id']}")
            self.log(f"  Stages: {best['stages']}")
            self.log(f"  SE reduction: {best['se_reduction']}")
            self.log(f"  Inter-modality shared: {best['inter_shared']}")
            self.log(f"  Merge bottleneck: {best['merge_bottleneck']}")
            self.log(f"  mAP: {best.get('mAP', 'N/A')}")
            self.log(f"  mAP@50: {best.get('mAP_50', 'N/A')}")

    def close(self):
        """Close log files."""
        if hasattr(self, 'master_log'):
            self.master_log.close()


def main():
    parser = argparse.ArgumentParser(description='Run GAFF Ablation Study')

    # Dataset paths
    parser.add_argument('--data', type=str, required=True,
                        help='Path to image data directory')
    parser.add_argument('--labels', type=str, required=True,
                        help='Path to labels directory')

    # Output
    parser.add_argument('--output-base', type=str, required=True,
                        help='Base directory for all experiment outputs')

    # Training config
    parser.add_argument('--backbone', type=str, default='mit_b1',
                        choices=['mit_b0', 'mit_b1', 'mit_b2', 'mit_b3', 'mit_b4', 'mit_b5'],
                        help='Backbone architecture')
    parser.add_argument('--epochs', type=int, default=25,
                        help='Number of training epochs per experiment')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate')

    args = parser.parse_args()

    # Run ablation study
    runner = GAFFAblationRunner(args)

    try:
        runner.run_all()
    finally:
        runner.close()


if __name__ == '__main__':
    main()
