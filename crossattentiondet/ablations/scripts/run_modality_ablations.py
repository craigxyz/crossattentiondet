#!/usr/bin/env python -u
"""
Master script to run all modality ablation experiments.

This script runs a minimal but comprehensive set of modality ablation experiments:
1. RGB + Thermal (no Event)
2. RGB + Event (no Thermal)
3. Thermal + Event (no RGB)
4. All modalities (baseline for comparison)

The experiments are designed to be CVPR-ready while minimizing computation time:
- Uses reduced epochs (10 instead of 25) for faster iteration
- Tests on a single backbone (mit_b1) for consistency
- Uses best-performing GAFF configuration from previous ablations

Usage:
    python crossattentiondet/ablations/scripts/run_modality_ablations.py \
        --data ../RGBX_Semantic_Segmentation/data/images \
        --labels ../RGBX_Semantic_Segmentation/data/labels \
        --output-dir results/modality_ablations \
        --epochs 10 \
        --backbone mit_b1
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime
import json


MODALITY_COMBINATIONS = [
    {
        'name': 'rgb_thermal',
        'modalities': 'rgb,thermal',
        'description': 'RGB + Thermal (no Event)'
    },
    {
        'name': 'rgb_event',
        'modalities': 'rgb,event',
        'description': 'RGB + Event (no Thermal)'
    },
    {
        'name': 'thermal_event',
        'modalities': 'thermal,event',
        'description': 'Thermal + Event (no RGB)'
    }
    # NOTE: All modalities baseline already exists in other experiments
    # (gaff_pilot_5epochs, gaff_ablations_full, cssa_ablations)
]


def run_experiment(data_dir, labels_dir, output_dir, backbone, epochs, batch_size, lr, exp_config):
    """Run a single modality ablation experiment."""

    exp_name = exp_config['name']
    modalities = exp_config['modalities']
    description = exp_config['description']

    print(f"\n{'='*80}")
    print(f"STARTING EXPERIMENT: {exp_name}")
    print(f"Description: {description}")
    print(f"Modalities: {modalities}")
    print(f"{'='*80}\n")

    # Build output directory for this experiment
    exp_output_dir = os.path.join(output_dir, exp_name)

    # Build command
    cmd = [
        sys.executable,
        'crossattentiondet/ablations/scripts/train_modality_ablation.py',
        '--data', data_dir,
        '--labels', labels_dir,
        '--output-dir', exp_output_dir,
        '--backbone', backbone,
        '--epochs', str(epochs),
        '--batch-size', str(batch_size),
        '--lr', str(lr),
        '--modalities', modalities
    ]

    print(f"Running command: {' '.join(cmd)}\n")

    # Run the experiment
    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n{'='*80}")
        print(f"EXPERIMENT {exp_name} COMPLETED SUCCESSFULLY")
        print(f"{'='*80}\n")
        return {
            'name': exp_name,
            'status': 'success',
            'modalities': modalities,
            'output_dir': exp_output_dir
        }
    except subprocess.CalledProcessError as e:
        print(f"\n{'='*80}")
        print(f"EXPERIMENT {exp_name} FAILED")
        print(f"Error: {e}")
        print(f"{'='*80}\n")
        return {
            'name': exp_name,
            'status': 'failed',
            'modalities': modalities,
            'output_dir': exp_output_dir,
            'error': str(e)
        }


def main():
    parser = argparse.ArgumentParser(description='Run all modality ablation experiments')

    # Dataset paths
    parser.add_argument('--data', type=str, required=True, help='Path to image data directory')
    parser.add_argument('--labels', type=str, required=True, help='Path to labels directory')
    parser.add_argument('--output-dir', type=str, default='results/modality_ablations',
                        help='Output directory for all experiments')

    # Training parameters
    parser.add_argument('--backbone', type=str, default='mit_b1',
                        choices=['mit_b0', 'mit_b1', 'mit_b2', 'mit_b3', 'mit_b4', 'mit_b5'],
                        help='Backbone architecture')
    parser.add_argument('--epochs', type=int, default=15,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.02, help='Learning rate')

    # Experiment selection
    parser.add_argument('--experiments', type=str, default='all',
                        help='Comma-separated list of experiments to run, or "all" (default: all)')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine which experiments to run
    if args.experiments.lower() == 'all':
        experiments_to_run = MODALITY_COMBINATIONS
    else:
        exp_names = [e.strip() for e in args.experiments.split(',')]
        experiments_to_run = [e for e in MODALITY_COMBINATIONS if e['name'] in exp_names]

        if not experiments_to_run:
            print(f"ERROR: No valid experiments found. Available: {[e['name'] for e in MODALITY_COMBINATIONS]}")
            sys.exit(1)

    # Print summary
    print("\n" + "="*80)
    print("MODALITY ABLATION EXPERIMENT SUITE")
    print("="*80)
    print(f"Output directory: {args.output_dir}")
    print(f"Backbone: {args.backbone}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"\nExperiments to run ({len(experiments_to_run)}):")
    for i, exp in enumerate(experiments_to_run, 1):
        print(f"  {i}. {exp['name']}: {exp['description']}")
    print("="*80 + "\n")

    # Run experiments
    results = []
    start_time = datetime.now()

    for i, exp_config in enumerate(experiments_to_run, 1):
        print(f"\n{'#'*80}")
        print(f"# EXPERIMENT {i}/{len(experiments_to_run)}")
        print(f"{'#'*80}\n")

        result = run_experiment(
            data_dir=args.data,
            labels_dir=args.labels,
            output_dir=args.output_dir,
            backbone=args.backbone,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            exp_config=exp_config
        )
        results.append(result)

    # Save summary
    end_time = datetime.now()
    total_time = (end_time - start_time).total_seconds() / 3600  # hours

    summary = {
        'config': {
            'data_dir': args.data,
            'labels_dir': args.labels,
            'output_dir': args.output_dir,
            'backbone': args.backbone,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.lr
        },
        'experiments': results,
        'summary': {
            'total_experiments': len(results),
            'successful': sum(1 for r in results if r['status'] == 'success'),
            'failed': sum(1 for r in results if r['status'] == 'failed'),
            'total_time_hours': total_time,
            'start_time': start_time.strftime('%Y-%m-%d %H:%M:%S'),
            'end_time': end_time.strftime('%Y-%m-%d %H:%M:%S')
        }
    }

    summary_path = os.path.join(args.output_dir, 'ablation_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # Print final summary
    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETE")
    print("="*80)
    print(f"Total experiments: {summary['summary']['total_experiments']}")
    print(f"Successful: {summary['summary']['successful']}")
    print(f"Failed: {summary['summary']['failed']}")
    print(f"Total time: {total_time:.2f} hours")
    print(f"\nSummary saved to: {summary_path}")
    print("="*80 + "\n")

    # Print results table
    print("\nRESULTS BY EXPERIMENT:")
    print("-" * 80)
    for result in results:
        status_symbol = "✓" if result['status'] == 'success' else "✗"
        print(f"{status_symbol} {result['name']:20s} | Modalities: {result['modalities']:25s} | {result['status']}")
    print("-" * 80)


if __name__ == '__main__':
    main()
