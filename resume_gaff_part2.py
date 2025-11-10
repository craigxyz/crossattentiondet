#!/usr/bin/env python -u
"""
Resume script for GAFF ablation experiments - PART 2 (Computer 2)

Runs experiments 020-029 (10 experiments total):
- exp_020-022: s3 stage config variants (last 3 of 7)
- exp_023-029: s234 stage config variants (all 7)

Usage:
    python resume_gaff_part2.py
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
    """Resumes GAFF ablation study - Part 2."""

    def __init__(self):
        self.output_base = 'results/gaff_ablations_full'
        self.master_log_path = os.path.join(self.output_base, 'master_log_part2.txt')
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

        # Define experiments for Part 2: exp_020 through exp_029
        self.experiments = self._define_experiments()

        # Results tracking
        self.results = []
        self.load_existing_results()

    def _define_experiments(self):
        """Define experiments 020-029 for Part 2."""
        experiments = []

        # s3 stage config (experiments 020-022, last 3 of 7)
        stage_label, stages = 's3', [3]
        configs_s3_part2 = [
            (20, 8, False, True),   # exp_020
            (21, 8, True, False),   # exp_021
            (22, 8, True, True),    # exp_022
        ]

        for exp_num, se_r, inter_s, merge_b in configs_s3_part2:
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

        # s234 stage config (experiments 023-029, all 7)
        stage_label, stages = 's234', [2, 3, 4]
        configs_s234 = [
            (23, 4, False, True),   # exp_023
            (24, 4, True, False),   # exp_024
            (25, 4, True, True),    # exp_025
            (26, 8, False, False),  # exp_026
            (27, 8, False, True),   # exp_027
            (28, 8, True, False),   # exp_028
            (29, 8, True, True),    # exp_029
        ]

        for exp_num, se_r, inter_s, merge_b in configs_s234:
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
        """Run a single experiment."""
        self.log_header(f"EXPERIMENT {exp['number']}/{total_exp_count}: {exp['id']}")
        self.log(f"  Phase: {exp['phase']}")
        self.log(f"  Stages: {exp['stages']}")
        self.log(f"  SE reduction: {exp['se_reduction']}")
        self.log(f"  Inter-modality shared: {exp['inter_shared']}")
        self.log(f"  Merge bottleneck: {exp['merge_bottleneck']}")
        self.log(f"  Output: {exp['output_dir']}")

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
        """Run all Part 2 experiments."""
        start_time = time.time()

        self.log_header("GAFF ABLATION STUDY - PART 2 (Computer 2)")
        self.log(f"Running experiments 020-029 (10 experiments)")
        self.log(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log("")

        # Run all experiments
        for exp in self.experiments:
            self.run_experiment(exp)

        # Final summary
        total_time = (time.time() - start_time) / 3600
        self.log_header("PART 2 COMPLETE")
        self.log(f"Completed {len(self.experiments)} experiments")
        self.log(f"Total time: {total_time:.2f} hours ({total_time/24:.2f} days)")
        self.log(f"Part 2 log: {self.master_log_path}")

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
