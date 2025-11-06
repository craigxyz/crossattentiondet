#!/usr/bin/env python
"""
Test/evaluate all backbone variants for comparison.

This script evaluates each trained MiT backbone variant checkpoint and compares
their detection accuracy, inference speed, and model size.

Usage:
    python scripts/test_all_backbones.py --data data/ --checkpoint-dir checkpoints/
    python scripts/test_all_backbones.py --data data/ --checkpoint-dir checkpoints/ --batch-size 4
"""

import os
import sys
import argparse
import time
import torch
from datetime import datetime

# Add parent directory to path to import crossattentiondet
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crossattentiondet.config import Config
from crossattentiondet.training.evaluator import Evaluator
from crossattentiondet.utils.metrics import METRICS_ENABLED


def get_model_size(model):
    """
    Calculate model size in MB.

    Args:
        model: PyTorch model

    Returns:
        float: Model size in megabytes
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb


def count_parameters(model):
    """
    Count total and trainable parameters.

    Args:
        model: PyTorch model

    Returns:
        tuple: (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def measure_inference_time(model, data_loader, device, num_batches=10):
    """
    Measure average inference time per image.

    Args:
        model: Trained model
        data_loader: DataLoader with test data
        device: Device to run inference on
        num_batches: Number of batches to measure (default: 10)

    Returns:
        tuple: (avg_time_ms, fps)
    """
    model.eval()
    times = []

    with torch.no_grad():
        for i, (images, targets) in enumerate(data_loader):
            if i >= num_batches:
                break

            images = list(image.to(device) for image in images)

            # Warm up GPU
            if i == 0:
                _ = model(images)
                continue

            # Measure time
            start_time = time.time()
            _ = model(images)
            elapsed = time.time() - start_time

            # Time per image
            batch_size = len(images)
            time_per_image = elapsed / batch_size
            times.append(time_per_image)

    avg_time = sum(times) / len(times) if times else 0
    avg_time_ms = avg_time * 1000
    fps = 1.0 / avg_time if avg_time > 0 else 0

    return avg_time_ms, fps


def test_backbone_variant(backbone_name, checkpoint_path, config):
    """
    Test a single backbone variant.

    Args:
        backbone_name: Name of the backbone variant
        checkpoint_path: Path to model checkpoint
        config: Configuration object

    Returns:
        dict: Test results including metrics and timing
    """
    print("\n" + "="*80)
    print(f"Testing {backbone_name.upper()}")
    print("="*80)

    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"[WARNING] Checkpoint not found: {checkpoint_path}")
        return {
            'backbone': backbone_name,
            'success': False,
            'error': 'Checkpoint not found'
        }

    # Update config for this backbone
    config.backbone_type = backbone_name
    config.model_path = checkpoint_path

    start_time = time.time()

    try:
        # Create evaluator
        print(f"\nInitializing evaluator for {backbone_name}...")
        evaluator = Evaluator(config)

        # Load checkpoint
        evaluator.load_checkpoint()

        # Get model statistics
        total_params, trainable_params = count_parameters(evaluator.model)
        model_size_mb = get_model_size(evaluator.model)

        print(f"\nModel Statistics:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Model size: {model_size_mb:.2f} MB")

        # Measure inference speed
        print(f"\nMeasuring inference speed...")
        avg_time_ms, fps = measure_inference_time(
            evaluator.model,
            evaluator.test_loader,
            config.device,
            num_batches=10
        )
        print(f"  Average inference time: {avg_time_ms:.2f} ms/image")
        print(f"  Throughput: {fps:.2f} FPS")

        # Run evaluation
        print(f"\nRunning evaluation on test set...")
        results = evaluator.evaluate()

        elapsed_time = time.time() - start_time

        # Extract key metrics
        if METRICS_ENABLED and results:
            map_50 = results.get('map_50', torch.tensor(0.0)).item()
            map_75 = results.get('map_75', torch.tensor(0.0)).item()
            map_overall = results.get('map', torch.tensor(0.0)).item()
        else:
            map_50 = map_75 = map_overall = 0.0

        print(f"\n{backbone_name.upper()} evaluation completed!")
        print(f"Time elapsed: {elapsed_time:.2f} seconds")

        return {
            'backbone': backbone_name,
            'success': True,
            'checkpoint': checkpoint_path,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': model_size_mb,
            'inference_time_ms': avg_time_ms,
            'fps': fps,
            'map_50': map_50,
            'map_75': map_75,
            'map': map_overall,
            'eval_time_seconds': elapsed_time
        }

    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"\n[ERROR] Evaluation failed for {backbone_name}: {e}")
        import traceback
        traceback.print_exc()
        return {
            'backbone': backbone_name,
            'success': False,
            'error': str(e),
            'eval_time_seconds': elapsed_time
        }


def main():
    """Main function to test all backbone variants."""

    parser = argparse.ArgumentParser(
        description="Test all MiT backbone variants for comparison",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data arguments
    parser.add_argument("--data", type=str, required=True,
                       help="Path to data directory containing images/ and labels/")

    # Checkpoint arguments
    parser.add_argument("--checkpoint-dir", type=str, default='checkpoints',
                       help="Directory containing model checkpoints")
    parser.add_argument("--checkpoint-pattern", type=str,
                       default='crossattentiondet_{backbone}.pth',
                       help="Checkpoint filename pattern (use {backbone} as placeholder)")

    # Test arguments
    parser.add_argument("--batch-size", type=int, default=2,
                       help="Batch size for evaluation")
    parser.add_argument("--backbones", nargs='+',
                       default=['mit_b0', 'mit_b1', 'mit_b2', 'mit_b4', 'mit_b5'],
                       choices=['mit_b0', 'mit_b1', 'mit_b2', 'mit_b4', 'mit_b5'],
                       help="List of backbones to test (default: all)")

    # Output arguments
    parser.add_argument("--results-dir", type=str, default='test_results',
                       help="Directory to save evaluation visualizations")

    args = parser.parse_args()

    # Check if metrics are available
    if not METRICS_ENABLED:
        print("\n[WARNING] torchmetrics not installed. mAP metrics will not be computed.")
        print("Install with: pip install torchmetrics")
        print("Continuing with inference speed and model size metrics only.\n")

    # Print configuration
    print("\n" + "="*80)
    print("TESTING ALL BACKBONE VARIANTS")
    print("="*80)
    print(f"Data directory: {args.data}")
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Backbones to test: {', '.join(args.backbones)}")
    print(f"Results directory: {args.results_dir}")
    print("="*80)

    # Create base configuration
    config = Config()
    config.data_root = args.data
    config.image_dir = os.path.join(args.data, 'images')
    config.label_dir = os.path.join(args.data, 'labels')
    config.batch_size = args.batch_size
    config.results_dir = args.results_dir

    # Track results for all backbones
    results = []
    total_start_time = time.time()

    # Test each backbone variant
    for i, backbone_name in enumerate(args.backbones, 1):
        print(f"\n\n{'#'*80}")
        print(f"# TESTING BACKBONE {i}/{len(args.backbones)}: {backbone_name.upper()}")
        print(f"{'#'*80}")

        # Build checkpoint path
        checkpoint_filename = args.checkpoint_pattern.format(backbone=backbone_name)
        checkpoint_path = os.path.join(args.checkpoint_dir, checkpoint_filename)

        result = test_backbone_variant(backbone_name, checkpoint_path, config)
        results.append(result)

        # Print progress
        completed = sum(1 for r in results if r['success'])
        failed = sum(1 for r in results if not r['success'])
        print(f"\nProgress: {i}/{len(args.backbones)} backbones tested "
              f"(✓ {completed} successful, ✗ {failed} failed)")

    # Print final summary
    total_elapsed = time.time() - total_start_time

    print("\n\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)

    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]

    if successful:
        print(f"\nModel Comparison:")
        print("-"*80)
        print(f"{'Backbone':<10} {'Params':<12} {'Size(MB)':<10} {'Time(ms)':<12} "
              f"{'FPS':<8} {'mAP@50':<10} {'mAP':<10}")
        print("-"*80)

        for r in successful:
            params_m = r['total_params'] / 1e6
            print(f"{r['backbone']:<10} {params_m:>10.1f}M {r['model_size_mb']:>9.2f} "
                  f"{r['inference_time_ms']:>11.2f} {r['fps']:>7.2f} "
                  f"{r['map_50']:>9.4f} {r['map']:>9.4f}")

        print("-"*80)

        # Speed ranking
        print(f"\nSpeed Ranking (fastest to slowest):")
        sorted_by_speed = sorted(successful, key=lambda x: x['inference_time_ms'])
        for i, r in enumerate(sorted_by_speed, 1):
            print(f"  {i}. {r['backbone']:<10} - {r['inference_time_ms']:.2f} ms/image ({r['fps']:.2f} FPS)")

        # Accuracy ranking (if metrics available)
        if METRICS_ENABLED:
            print(f"\nAccuracy Ranking (mAP, best to worst):")
            sorted_by_map = sorted(successful, key=lambda x: x['map'], reverse=True)
            for i, r in enumerate(sorted_by_map, 1):
                print(f"  {i}. {r['backbone']:<10} - mAP: {r['map']:.4f}, mAP@50: {r['map_50']:.4f}")

        # Size ranking
        print(f"\nModel Size Ranking (smallest to largest):")
        sorted_by_size = sorted(successful, key=lambda x: x['total_params'])
        for i, r in enumerate(sorted_by_size, 1):
            params_m = r['total_params'] / 1e6
            print(f"  {i}. {r['backbone']:<10} - {params_m:.1f}M params, {r['model_size_mb']:.2f} MB")

    if failed:
        print(f"\nFailed Evaluations:")
        print("-"*80)
        for r in failed:
            print(f"  ✗ {r['backbone']:8s} | {r.get('error', 'Unknown error')}")
        print("-"*80)

    print(f"\nTotal evaluation time: {total_elapsed:.2f} seconds ({total_elapsed/60:.2f} minutes)")
    print(f"Successful: {len(successful)}/{len(results)}")
    print(f"Failed: {len(failed)}/{len(results)}")

    # Save summary to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = os.path.join(args.checkpoint_dir, f'evaluation_summary_{timestamp}.txt')

    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("EVALUATION SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Data directory: {args.data}\n")
        f.write(f"Batch size: {args.batch_size}\n\n")
        f.write(f"Total evaluation time: {total_elapsed:.2f} seconds ({total_elapsed/60:.2f} minutes)\n\n")

        if successful:
            f.write("Model Comparison:\n")
            f.write("-"*80 + "\n")
            f.write(f"{'Backbone':<10} {'Params':<12} {'Size(MB)':<10} {'Time(ms)':<12} "
                   f"{'FPS':<8} {'mAP@50':<10} {'mAP':<10}\n")
            f.write("-"*80 + "\n")
            for r in successful:
                params_m = r['total_params'] / 1e6
                f.write(f"{r['backbone']:<10} {params_m:>10.1f}M {r['model_size_mb']:>9.2f} "
                       f"{r['inference_time_ms']:>11.2f} {r['fps']:>7.2f} "
                       f"{r['map_50']:>9.4f} {r['map']:>9.4f}\n")
            f.write("-"*80 + "\n\n")

            f.write("Speed Ranking:\n")
            sorted_by_speed = sorted(successful, key=lambda x: x['inference_time_ms'])
            for i, r in enumerate(sorted_by_speed, 1):
                f.write(f"  {i}. {r['backbone']:<10} - {r['inference_time_ms']:.2f} ms/image ({r['fps']:.2f} FPS)\n")

            if METRICS_ENABLED:
                f.write("\nAccuracy Ranking (mAP):\n")
                sorted_by_map = sorted(successful, key=lambda x: x['map'], reverse=True)
                for i, r in enumerate(sorted_by_map, 1):
                    f.write(f"  {i}. {r['backbone']:<10} - mAP: {r['map']:.4f}, mAP@50: {r['map_50']:.4f}\n")

        f.write(f"\nSuccessful: {len(successful)}/{len(results)}\n")
        f.write(f"Failed: {len(failed)}/{len(results)}\n")

    print(f"\nSummary saved to: {summary_file}")
    print("\n" + "="*80)
    print("ALL EVALUATIONS COMPLETED")
    print("="*80 + "\n")

    # Return exit code based on results
    if failed:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
