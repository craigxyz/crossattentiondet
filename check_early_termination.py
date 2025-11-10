#!/usr/bin/env python3
"""Check which experiments ended early by examining training logs."""

import os
import re
from pathlib import Path

def check_experiment_status(log_path):
    """Check if an experiment completed or ended early."""
    try:
        with open(log_path, 'r') as f:
            content = f.read()

        # Find all epoch mentions (format: "EPOCH X/Y" in uppercase)
        epoch_pattern = r'EPOCH (\d+)/(\d+)'
        epochs = re.findall(epoch_pattern, content, re.IGNORECASE)

        if not epochs:
            return None, None, "No epoch information"

        # Get the expected total epochs
        total_epochs = int(epochs[0][1])

        # Find the last epoch that was started
        last_epoch_started = int(epochs[-1][0])

        # Check if experiment completed properly
        has_completion = "EXPERIMENT COMPLETE" in content or "TRAINING COMPLETE" in content
        has_final_results = "FINAL RESULTS" in content

        # Check if it completed all epochs
        completed = last_epoch_started == total_epochs and has_completion

        return last_epoch_started, total_epochs, completed

    except Exception as e:
        return None, None, f"Error: {str(e)}"

# Find all training logs
base_dir = Path('results/gaff_ablations_full')
log_paths = []

for phase_dir in ['phase1_stage_selection', 'phase2_hyperparameter_tuning']:
    phase_path = base_dir / phase_dir
    if phase_path.exists():
        for exp_dir in sorted(phase_path.iterdir()):
            log_file = exp_dir / 'training.log'
            if log_file.exists():
                log_paths.append((exp_dir.name, log_file))

print("=" * 80)
print("EXPERIMENT STATUS REPORT")
print("=" * 80)

early_terminations = []
completed = []
errors = []

for exp_name, log_path in log_paths:
    last_epoch, total_epochs, status = check_experiment_status(log_path)

    if last_epoch is None:
        errors.append((exp_name, status))
    elif status == True:
        completed.append((exp_name, last_epoch, total_epochs))
    else:
        early_terminations.append((exp_name, last_epoch, total_epochs))

print(f"\n✓ COMPLETED: {len(completed)} experiments")
print("-" * 80)
for exp_name, last, total in completed:
    print(f"  {exp_name}: {last}/{total} epochs")

print(f"\n✗ ENDED EARLY: {len(early_terminations)} experiments")
print("-" * 80)
for exp_name, last, total in early_terminations:
    missing = total - last if last and total else "?"
    print(f"  {exp_name}: Only {last}/{total} epochs (missing {missing})")

if errors:
    print(f"\n⚠ ERRORS: {len(errors)} experiments")
    print("-" * 80)
    for exp_name, error_msg in errors:
        print(f"  {exp_name}: {error_msg}")

print("\n" + "=" * 80)
print(f"SUMMARY: {len(completed)} completed, {len(early_terminations)} ended early, {len(errors)} errors")
print("=" * 80)
