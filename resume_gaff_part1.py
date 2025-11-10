#!/usr/bin/env python -u
"""
Resume script for GAFF ablation experiments - PART 1 (Computer 1)

Runs experiments 010-019 (10 experiments total):
- exp_010: Resume from checkpoint (s4, inter_shared=True)
- exp_011-015: s4 stage config variants
- exp_016-019: s3 stage config variants (first 4 of 7)

Usage:
    python resume_gaff_part1.py
"""

import sys
import os
import subprocess
import json
import csv
import time
from datetime import datetime

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


class GAFFAblationResumer:
    """Resumes GAFF ablation study - Part 1."""

    def __init__(self):
        self.output_base = 'results/gaff_ablations_full'
        self.master_log_path = os.path.join(self.output_base, 'master_log_part1.txt')
        self.summary_csv_path = os.path.join(self.output_base, 'summary_all_experiments.csv')

        # Create master log for this part
        self.master_log = open(self.master_log_path, 'w', buffering=1)

        # Configuration from original run
        self.data_dir = '../RGBX_Semantic_Segmentation/data/images'
        self.labels_dir = '../RGBX_Semantic_Segmentation/data/labels'
        self.backbone = 'mit_b1'
        self.epochs = 15
        self.batch_size = 16
        self.lr = 0.02

        # Define experiments for Part 1: exp_010 through exp_019
        self.experiments = self._define_experiments()

        # Results tracking
        self.results = []
        self.load_existing_results()

    def _define_experiments(self):
        """Define experiments 010-019 for Part 1."""
        experiments = []

        # s4 stage config (experiments 010-015)
        stage_label, stages = 's4', [4]
        configs_s4 = [
            (10, 4, True, False),   # exp_010 - RESUME
            (11, 4, True, True),    # exp_011
            (12, 8, False, False),  # exp_012
            (13, 8, False, True),   # exp_013
            (14, 8, True, False),   # exp_014
            (15, 8, True, True),    # exp_015
        ]

        for exp_num, se_r, inter_s, merge_b in configs_s4:
            exp_id = f"exp_{exp_num:03d}_{stage_label}_r{se_r}_is{int(inter_s)}_mb{int(merge_b)}"
            experiments.append({
                'id': exp_id,
                'number': exp_num,
                'stages': stages,
                'stage_label': stage_label,
                'se_reduction': se_r,
                'inter_shared': inter_s,
                'merge_bottleneck': merge_b,
                'phase': 2,
                'resume': (exp_num == 10),  # Only resume exp_010
                'output_dir': os.path.join(
                    self.output_base,
                    'phase2_hyperparameter_tuning',
                    exp_id
                )
            })

        # s3 stage config (experiments 016-019, first 4 of 7)
        stage_label, stages = 's3', [3]
        configs_s3_part1 = [
            (16, 4, False, True),   # exp_016
            (17, 4, True, False),   # exp_017
            (18, 4, True, True),    # exp_018
            (19, 8, False, False),  # exp_019
        ]

        for exp_num, se_r, inter_s, merge_b in configs_s3_part1:
            exp_id = f"exp_{exp_num:03d}_{stage_label}_r{se_r}_is{int(inter_s)}_mb{int(merge_b)}"
            experiments.append({
                'id': exp_id,
                'number': exp_num,
                'stages': stages,
                'stage_label': stage_label,
                'se_reduction': se_r,
                'inter_shared': inter_s,
                'merge_bottleneck': merge_b,
                'phase': 2,
                'resume': False,
                'output_dir': os.path.join(
                    self.output_base,
                    'phase2_hyperparameter_tuning',
                    exp_id
                )
            })

        return experiments

    def load_existing_results(self):
        """Load results from existing summary CSV."""
        if os.path.exists(self.summary_csv_path):
            with open(self.summary_csv_path, 'r') as f:
                reader = csv.DictReader(f)
                self.results = list(reader)
            self.log(f"Loaded {len(self.results)} existing results")

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

    def run_experiment(self, exp, total_exp_count=29):
        """Run or resume a single experiment."""
        self.log_header(f"EXPERIMENT {exp['number']}/{total_exp_count}: {exp['id']}")
        self.log(f"  Phase: {exp['phase']}")
        self.log(f"  Stages: {exp['stages']}")
        self.log(f"  SE reduction: {exp['se_reduction']}")
        self.log(f"  Inter-modality shared: {exp['inter_shared']}")
        self.log(f"  Merge bottleneck: {exp['merge_bottleneck']}")
        self.log(f"  Output: {exp['output_dir']}")

        if exp['resume']:
            self.log(f"  Mode: RESUME FROM CHECKPOINT")

        # Build command
        cmd = [
            'python', '-u',
            'train_gaff_ablation_resume.py',
            '--data', self.data_dir,
            '--labels', self.labels_dir,
            '--output-dir', exp['output_dir'],
            '--backbone', self.backbone,
            '--epochs', str(self.epochs),
            '--batch-size', str(self.batch_size),
            '--lr', str(self.lr),
            '--gaff-stages', ','.join(map(str, exp['stages'])),
            '--gaff-se-reduction', str(exp['se_reduction']),
            '--gaff-inter-shared', 'true' if exp['inter_shared'] else 'false',
            '--gaff-merge-bottleneck', 'true' if exp['merge_bottleneck'] else 'false'
        ]

        if exp['resume']:
            cmd.extend(['--resume-from-checkpoint', 'true'])

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

        # Collect and save results
        self._update_results(exp, status, exp_time)

        return status

    def _update_results(self, exp, status, exp_time):
        """Update results CSV with experiment data."""
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
                    'mAP': results_json.get('results', {}).get('mAP', ''),
                    'mAP_50': results_json.get('results', {}).get('mAP_50', ''),
                    'mAP_75': results_json.get('results', {}).get('mAP_75', ''),
                    'final_train_loss': results_json.get('training', {}).get('final_train_loss', ''),
                    'total_params_M': results_json.get('model', {}).get('total_params_M', '')
                })
            except Exception as e:
                self.log(f"  Warning: Could not load results JSON: {e}")
                result_data.update({
                    'mAP': '', 'mAP_50': '', 'mAP_75': '',
                    'final_train_loss': '', 'total_params_M': ''
                })
        else:
            result_data.update({
                'mAP': '', 'mAP_50': '', 'mAP_75': '',
                'final_train_loss': '', 'total_params_M': ''
            })

        # Update or append to results
        updated = False
        for i, r in enumerate(self.results):
            if r['experiment_id'] == exp['id']:
                self.results[i] = result_data
                updated = True
                break

        if not updated:
            self.results.append(result_data)

        # Save to CSV
        self._save_summary_csv()

    def _save_summary_csv(self):
        """Save summary CSV of all experiments."""
        fieldnames = [
            'experiment_id', 'phase', 'stages', 'se_reduction', 'inter_shared', 'merge_bottleneck',
            'status', 'time_hours', 'mAP', 'mAP_50', 'mAP_75', 'final_train_loss', 'total_params_M'
        ]

        with open(self.summary_csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.results)

        self.log(f"Summary CSV updated: {self.summary_csv_path}")

    def run_all(self):
        """Run all Part 1 experiments."""
        start_time = time.time()

        self.log_header("GAFF ABLATION STUDY - PART 1 (Computer 1)")
        self.log(f"Running experiments 010-019 (10 experiments)")
        self.log(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log("")

        # Run all experiments
        for exp in self.experiments:
            self.run_experiment(exp)

        # Final summary
        total_time = (time.time() - start_time) / 3600
        self.log_header("PART 1 COMPLETE")
        self.log(f"Completed {len(self.experiments)} experiments")
        self.log(f"Total time: {total_time:.2f} hours ({total_time/24:.2f} days)")
        self.log(f"Part 1 log: {self.master_log_path}")

    def close(self):
        """Close log files."""
        if hasattr(self, 'master_log'):
            self.master_log.close()


def main():
    resumer = GAFFAblationResumer()

    try:
        resumer.run_all()
    finally:
        resumer.close()


if __name__ == '__main__':
    main()
