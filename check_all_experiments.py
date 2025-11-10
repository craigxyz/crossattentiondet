#!/usr/bin/env python3
"""Check which experiments ended early across all ablation studies."""

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

def check_ablation_study(study_name, base_path):
    """Check all experiments in a study."""
    base_dir = Path(base_path)
    if not base_dir.exists():
        return None

    results = {
        'completed': [],
        'early': [],
        'errors': [],
        'missing': []
    }

    # Find all experiment directories
    exp_dirs = []
    for item in base_dir.rglob('exp_*'):
        if item.is_dir():
            exp_dirs.append(item)

    for exp_dir in sorted(exp_dirs):
        log_file = exp_dir / 'training.log'
        exp_name = exp_dir.name

        if not log_file.exists():
            results['missing'].append(exp_name)
            continue

        last_epoch, total_epochs, status = check_experiment_status(log_file)

        if last_epoch is None:
            results['errors'].append((exp_name, status))
        elif status == True:
            results['completed'].append((exp_name, last_epoch, total_epochs))
        else:
            results['early'].append((exp_name, last_epoch, total_epochs))

    return results

# Check all ablation studies
studies = {
    'GAFF Ablations': 'results/gaff_ablations_full',
    'CSSA Ablations Phase 2': 'results/cssa_ablations_phase2',
    'Modality Ablations': 'results/modality_ablations'
}

print("=" * 80)
print("COMPREHENSIVE EXPERIMENT STATUS REPORT")
print("=" * 80)

total_completed = 0
total_early = 0
total_errors = 0
total_missing = 0

for study_name, study_path in studies.items():
    print(f"\n{'='*80}")
    print(f"{study_name.upper()}")
    print(f"{'='*80}")

    results = check_ablation_study(study_name, study_path)

    if results is None:
        print(f"  ⚠ Study directory not found: {study_path}")
        continue

    if results['completed']:
        print(f"\n✓ COMPLETED: {len(results['completed'])} experiments")
        print("-" * 80)
        for exp_name, last, total in results['completed']:
            print(f"  {exp_name}: {last}/{total} epochs")
        total_completed += len(results['completed'])

    if results['early']:
        print(f"\n✗ ENDED EARLY: {len(results['early'])} experiments")
        print("-" * 80)
        for exp_name, last, total in results['early']:
            missing = total - last if last and total else "?"
            print(f"  {exp_name}: Only {last}/{total} epochs (missing {missing})")
        total_early += len(results['early'])

    if results['missing']:
        print(f"\n⚠ NO LOG FILE: {len(results['missing'])} experiments")
        print("-" * 80)
        for exp_name in results['missing']:
            print(f"  {exp_name}: No training.log found")
        total_missing += len(results['missing'])

    if results['errors']:
        print(f"\n⚠ ERRORS: {len(results['errors'])} experiments")
        print("-" * 80)
        for exp_name, error_msg in results['errors']:
            print(f"  {exp_name}: {error_msg}")
        total_errors += len(results['errors'])

print("\n" + "=" * 80)
print("OVERALL SUMMARY")
print("=" * 80)
print(f"  ✓ Completed: {total_completed}")
print(f"  ✗ Ended Early: {total_early}")
print(f"  ⚠ Missing Logs: {total_missing}")
print(f"  ⚠ Errors: {total_errors}")
print(f"  Total: {total_completed + total_early + total_missing + total_errors}")
print("=" * 80)
