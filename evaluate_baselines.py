#!/usr/bin/env python
"""
Evaluate baseline checkpoints and save results.

Evaluates all baseline models (mit_b0, mit_b1) from training_logs.
"""
import sys
import os
import json
import torch
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from crossattentiondet.config import Config
from crossattentiondet.training.evaluator import Evaluator


def evaluate_checkpoint(checkpoint_path, backbone, data_dir, labels_dir, output_dir):
    """Evaluate a single checkpoint and save results."""

    print(f"\n{'='*80}")
    print(f"Evaluating {backbone} baseline checkpoint")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"{'='*80}\n")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load checkpoint to get backbone info
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Get backbone from checkpoint if available
    checkpoint_backbone = checkpoint.get('backbone', backbone)
    print(f"Checkpoint backbone: {checkpoint_backbone}")
    print(f"Requested backbone: {backbone}")

    # Create config
    config = Config()
    config.image_dir = data_dir
    config.label_dir = labels_dir
    config.model_path = checkpoint_path
    config.backbone = checkpoint_backbone  # Use checkpoint's backbone
    config.results_dir = output_dir
    config.batch_size = 8  # Safe batch size for evaluation
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create evaluator
    evaluator = Evaluator(config)
    evaluator.load_checkpoint(checkpoint_path)

    # Run evaluation
    results = evaluator.evaluate()

    if results:
        # Save results to JSON
        output_data = {
            "experiment_id": f"baseline_{backbone}",
            "config": {
                "backbone": backbone,
                "fusion_type": "frm_ffm",
                "epochs": 15,
                "batch_size": 16,
                "learning_rate": 0.02,
                "checkpoint": checkpoint_path
            },
            "results": results,
            "model": {
                "total_params": evaluator.model.parameters().__sizeof__(),
                "backbone": backbone
            },
            "timestamp": datetime.now().isoformat()
        }

        # Save to file
        output_file = os.path.join(output_dir, 'final_results.json')
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\n✅ Results saved to: {output_file}")
        print(f"\n📊 Summary:")
        print(f"  mAP: {results.get('mAP', 'N/A'):.4f}" if 'mAP' in results else "  mAP: N/A")
        print(f"  mAP@50: {results.get('mAP_50', 'N/A'):.4f}" if 'mAP_50' in results else "  mAP@50: N/A")
        print(f"  mAP@75: {results.get('mAP_75', 'N/A'):.4f}" if 'mAP_75' in results else "  mAP@75: N/A")

        return results
    else:
        print(f"\n❌ Evaluation failed for {backbone}")
        return None


def main():
    """Evaluate all baseline checkpoints."""

    # Configuration
    data_dir = "../RGBX_Semantic_Segmentation/data/images"
    labels_dir = "../RGBX_Semantic_Segmentation/data/labels"
    training_logs_base = "training_logs/run_20251107_102948"

    # Baselines to evaluate
    baselines = [
        {
            'backbone': 'mit_b0',
            'checkpoint': f'{training_logs_base}/mit_b0/checkpoints/best_model.pth',
            'output_dir': 'results/baseline_mit_b0_evaluated'
        },
        {
            'backbone': 'mit_b1',
            'checkpoint': f'{training_logs_base}/mit_b1/checkpoints/best_model.pth',
            'output_dir': 'results/baseline_mit_b1_evaluated'
        }
    ]

    print("\n" + "="*80)
    print("BASELINE MODEL EVALUATION")
    print("="*80)
    print(f"\nData directory: {data_dir}")
    print(f"Labels directory: {labels_dir}")
    print(f"Number of models to evaluate: {len(baselines)}\n")

    all_results = {}

    # Evaluate each baseline
    for i, baseline in enumerate(baselines, 1):
        print(f"\n[{i}/{len(baselines)}] Evaluating {baseline['backbone']}...")

        # Check if checkpoint exists
        if not os.path.exists(baseline['checkpoint']):
            print(f"❌ Checkpoint not found: {baseline['checkpoint']}")
            continue

        # Run evaluation
        results = evaluate_checkpoint(
            checkpoint_path=baseline['checkpoint'],
            backbone=baseline['backbone'],
            data_dir=data_dir,
            labels_dir=labels_dir,
            output_dir=baseline['output_dir']
        )

        if results:
            all_results[baseline['backbone']] = results

    # Summary
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"\nSuccessfully evaluated: {len(all_results)}/{len(baselines)} models\n")

    if all_results:
        print("Results Summary:")
        print("-" * 80)
        print(f"{'Backbone':<15} {'mAP':<10} {'mAP@50':<10} {'mAP@75':<10}")
        print("-" * 80)
        for backbone, results in all_results.items():
            mAP = results.get('mAP', 0.0)
            mAP_50 = results.get('mAP_50', 0.0)
            mAP_75 = results.get('mAP_75', 0.0)
            print(f"{backbone:<15} {mAP:<10.4f} {mAP_50:<10.4f} {mAP_75:<10.4f}")
        print("-" * 80)

        # Save summary
        summary = {
            "timestamp": datetime.now().isoformat(),
            "num_evaluated": len(all_results),
            "results": all_results
        }

        summary_file = "results/baseline_evaluation_summary.json"
        os.makedirs("results", exist_ok=True)
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n📄 Summary saved to: {summary_file}")

    print("\n✅ All evaluations complete!\n")


if __name__ == '__main__':
    main()
