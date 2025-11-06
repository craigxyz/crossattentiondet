#!/usr/bin/env python
"""
Train all backbone variants serially for comparison.

This script trains each MiT backbone variant (mit_b0 through mit_b5) for a specified
number of epochs and saves separate model checkpoints for each variant.

Usage:
    python scripts/train_all_backbones.py --data data/ --epochs 5
    python scripts/train_all_backbones.py --data data/ --epochs 10 --batch-size 4 --lr 0.005
"""

import os
import sys
import argparse
import time
from datetime import datetime

# Add parent directory to path to import crossattentiondet
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crossattentiondet.config import Config
from crossattentiondet.training.trainer import Trainer


def train_backbone_variant(backbone_name, config):
    """
    Train a single backbone variant.

    Args:
        backbone_name: Name of the backbone variant (e.g., 'mit_b0')
        config: Configuration object with training parameters

    Returns:
        dict: Training results including time elapsed and final loss
    """
    print("\n" + "="*80)
    print(f"Training {backbone_name.upper()}")
    print("="*80)

    # Update config for this backbone
    config.backbone_type = backbone_name
    config.model_path = f'checkpoints/crossattentiondet_{backbone_name}.pth'

    # Create trainer
    print(f"\nInitializing trainer for {backbone_name}...")
    start_time = time.time()

    try:
        trainer = Trainer(config)

        # Train the model
        print(f"\nStarting training for {config.num_epochs} epochs...")
        trainer.train()

        elapsed_time = time.time() - start_time

        print(f"\n{backbone_name.upper()} training completed!")
        print(f"Time elapsed: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
        print(f"Model saved to: {config.model_path}")

        return {
            'backbone': backbone_name,
            'success': True,
            'time_seconds': elapsed_time,
            'epochs': config.num_epochs,
            'checkpoint': config.model_path
        }

    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"\n[ERROR] Training failed for {backbone_name}: {e}")
        return {
            'backbone': backbone_name,
            'success': False,
            'time_seconds': elapsed_time,
            'error': str(e)
        }


def main():
    """Main function to train all backbone variants serially."""

    parser = argparse.ArgumentParser(
        description="Train all MiT backbone variants serially for comparison",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data arguments
    parser.add_argument("--data", type=str, required=True,
                       help="Path to data directory containing images/ and labels/")

    # Training arguments
    parser.add_argument("--epochs", type=int, default=5,
                       help="Number of training epochs for each backbone")
    parser.add_argument("--batch-size", type=int, default=2,
                       help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.005,
                       help="Learning rate")

    # Backbone selection
    parser.add_argument("--backbones", nargs='+',
                       default=['mit_b0', 'mit_b1', 'mit_b2', 'mit_b4', 'mit_b5'],
                       choices=['mit_b0', 'mit_b1', 'mit_b2', 'mit_b4', 'mit_b5'],
                       help="List of backbones to train (default: all)")

    # Output directory
    parser.add_argument("--checkpoint-dir", type=str, default='checkpoints',
                       help="Directory to save model checkpoints")

    args = parser.parse_args()

    # Print configuration
    print("\n" + "="*80)
    print("TRAINING ALL BACKBONE VARIANTS")
    print("="*80)
    print(f"Data directory: {args.data}")
    print(f"Epochs per backbone: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Backbones to train: {', '.join(args.backbones)}")
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    print("="*80)

    # Create checkpoint directory if it doesn't exist
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    print(f"\nCheckpoint directory created/verified: {args.checkpoint_dir}")

    # Create base configuration
    config = Config()
    config.data_root = args.data
    config.image_dir = os.path.join(args.data, 'images')
    config.label_dir = os.path.join(args.data, 'labels')
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.lr

    # Track results for all backbones
    results = []
    total_start_time = time.time()

    # Train each backbone variant
    for i, backbone_name in enumerate(args.backbones, 1):
        print(f"\n\n{'#'*80}")
        print(f"# TRAINING BACKBONE {i}/{len(args.backbones)}: {backbone_name.upper()}")
        print(f"{'#'*80}")

        result = train_backbone_variant(backbone_name, config)
        results.append(result)

        # Print progress
        completed = sum(1 for r in results if r['success'])
        failed = sum(1 for r in results if not r['success'])
        print(f"\nProgress: {i}/{len(args.backbones)} backbones processed "
              f"(✓ {completed} successful, ✗ {failed} failed)")

    # Print final summary
    total_elapsed = time.time() - total_start_time

    print("\n\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)

    print(f"\nTotal time: {total_elapsed:.2f} seconds ({total_elapsed/60:.2f} minutes)")
    print(f"Epochs per backbone: {args.epochs}")
    print(f"\nResults:")
    print("-"*80)

    for result in results:
        backbone = result['backbone']
        if result['success']:
            time_min = result['time_seconds'] / 60
            print(f"  ✓ {backbone:8s} | {time_min:6.2f} min | {result['checkpoint']}")
        else:
            print(f"  ✗ {backbone:8s} | FAILED | {result.get('error', 'Unknown error')}")

    print("-"*80)

    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]

    print(f"\nSuccessful: {len(successful)}/{len(results)}")
    print(f"Failed: {len(failed)}/{len(results)}")

    if successful:
        avg_time = sum(r['time_seconds'] for r in successful) / len(successful)
        print(f"Average training time (successful): {avg_time/60:.2f} minutes")

        print("\nSpeed comparison (successful backbones):")
        sorted_results = sorted(successful, key=lambda x: x['time_seconds'])
        for r in sorted_results:
            time_min = r['time_seconds'] / 60
            print(f"  {r['backbone']:8s}: {time_min:6.2f} min")

    # Save summary to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = os.path.join(args.checkpoint_dir, f'training_summary_{timestamp}.txt')

    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("TRAINING SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Data directory: {args.data}\n")
        f.write(f"Epochs per backbone: {args.epochs}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Learning rate: {args.lr}\n\n")
        f.write(f"Total time: {total_elapsed:.2f} seconds ({total_elapsed/60:.2f} minutes)\n\n")
        f.write("Results:\n")
        f.write("-"*80 + "\n")
        for result in results:
            backbone = result['backbone']
            if result['success']:
                time_min = result['time_seconds'] / 60
                f.write(f"  ✓ {backbone:8s} | {time_min:6.2f} min | {result['checkpoint']}\n")
            else:
                f.write(f"  ✗ {backbone:8s} | FAILED | {result.get('error', 'Unknown error')}\n")
        f.write("-"*80 + "\n")
        f.write(f"\nSuccessful: {len(successful)}/{len(results)}\n")
        f.write(f"Failed: {len(failed)}/{len(results)}\n")

    print(f"\nSummary saved to: {summary_file}")
    print("\n" + "="*80)
    print("ALL TRAINING COMPLETED")
    print("="*80 + "\n")

    # Return exit code based on results
    if failed:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
