#!/usr/bin/env python -u
"""
Resume script for GAFF ablation experiments.

Resumes from where the ablation study was interrupted, including:
- Resuming exp_010 from its checkpoint (epoch 11)
- Running all remaining experiments (exp_011 through exp_029)

Usage:
    python resume_gaff_ablations.py
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
    """Resumes GAFF ablation study from interruption point."""

    def __init__(self):
        self.output_base = 'results/gaff_ablations_full'
        self.master_log_path = os.path.join(self.output_base, 'master_log.txt')
        self.summary_csv_path = os.path.join(self.output_base, 'summary_all_experiments.csv')

        # Open master log in append mode
        self.master_log = open(self.master_log_path, 'a', buffering=1)

        # Configuration from original run
        self.data_dir = '../RGBX_Semantic_Segmentation/data/images'
        self.labels_dir = '../RGBX_Semantic_Segmentation/data/labels'
        self.backbone = 'mit_b1'
        self.epochs = 15
        self.batch_size = 16
        self.lr = 0.02

        # Top 3 stage configs from phase 1
        self.top_stage_configs = [
            ('s4', [4]),
            ('s3', [3]),
            ('s234', [2, 3, 4])
        ]

        # Results tracking
        self.results = []
        self.load_existing_results()

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

    def get_phase2_experiments(self):
        """Generate all phase 2 experiment configurations."""
        se_reductions = [4, 8]
        inter_shared_opts = [False, True]
        merge_bottleneck_opts = [False, True]

        experiments = []
        exp_num = 9  # Start from exp_009 (first phase 2 experiment)

        for stage_label, stages in self.top_stage_configs:
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
                            'number': exp_num,
                            'stages': stages,
                            'stage_label': stage_label,
                            'se_reduction': se_r,
                            'inter_shared': inter_s,
                            'merge_bottleneck': merge_b,
                            'phase': 2,
                            'output_dir': os.path.join(
                                self.output_base,
                                'phase2_hyperparameter_tuning',
                                exp_id
                            )
                        })
                        exp_num += 1

        return experiments

    def check_experiment_status(self, exp):
        """Check if experiment is complete, partial, or not started."""
        output_dir = exp['output_dir']

        # Check if final results exist
        results_json = os.path.join(output_dir, 'final_results.json')
        if os.path.exists(results_json):
            return 'complete', None

        # Check if checkpoint exists (partial run)
        checkpoint = os.path.join(output_dir, 'checkpoint.pth')
        metrics_csv = os.path.join(output_dir, 'metrics_per_epoch.csv')

        if os.path.exists(checkpoint) and os.path.exists(metrics_csv):
            # Count completed epochs
            with open(metrics_csv, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                completed_epochs = len(rows)

            if completed_epochs > 0:
                return 'partial', completed_epochs

        return 'not_started', None

    def run_experiment(self, exp, resume_epoch=None, total_exp_count=29):
        """Run or resume a single experiment."""
        self.log_header(f"EXPERIMENT {exp['number']}/{total_exp_count}: {exp['id']}")
        self.log(f"  Phase: {exp['phase']}")
        self.log(f"  Stages: {exp['stages']}")
        self.log(f"  SE reduction: {exp['se_reduction']}")
        self.log(f"  Inter-modality shared: {exp['inter_shared']}")
        self.log(f"  Merge bottleneck: {exp['merge_bottleneck']}")
        self.log(f"  Output: {exp['output_dir']}")

        if resume_epoch is not None:
            self.log(f"  Resuming from epoch {resume_epoch}")

        # Build command
        cmd = [
            'python', '-u',
            'train_gaff_ablation_resume.py',  # Use enhanced training script
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

        if resume_epoch is not None:
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

        self.log(f"Summary CSV saved to: {self.summary_csv_path}")

    def resume_all(self):
        """Resume all pending experiments."""
        start_time = time.time()

        self.log_header("GAFF ABLATION STUDY RESUME")
        self.log(f"Resuming from interruption at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log("")

        # Get all phase 2 experiments
        all_phase2_experiments = self.get_phase2_experiments()

        # Determine which experiments need to run
        pending_experiments = []

        for exp in all_phase2_experiments:
            status, completed_epochs = self.check_experiment_status(exp)

            if status == 'complete':
                self.log(f"{exp['id']}: Already complete, skipping")
            elif status == 'partial':
                self.log(f"{exp['id']}: Partial run detected ({completed_epochs}/{self.epochs} epochs), will resume")
                pending_experiments.append((exp, completed_epochs))
            else:  # not_started
                self.log(f"{exp['id']}: Not started, will run")
                pending_experiments.append((exp, None))

        self.log("")
        self.log(f"Found {len(pending_experiments)} experiments to run/resume")
        self.log("")

        # Run all pending experiments
        for exp, resume_epoch in pending_experiments:
            self.run_experiment(exp, resume_epoch=resume_epoch)

        # Final summary
        total_time = (time.time() - start_time) / 3600
        self.log_header("ABLATION STUDY RESUME COMPLETE")
        self.log(f"Completed {len(pending_experiments)} experiments")
        self.log(f"Total time: {total_time:.2f} hours ({total_time/24:.2f} days)")
        self.log(f"Summary CSV: {self.summary_csv_path}")
        self.log(f"Master log: {self.master_log_path}")

    def close(self):
        """Close log files."""
        if hasattr(self, 'master_log'):
            self.master_log.close()


def main():
    resumer = GAFFAblationResumer()

    try:
        resumer.resume_all()
    finally:
        resumer.close()


if __name__ == '__main__':
    main()
