#!/usr/bin/env python3
"""
Evaluate 15-epoch trained baseline models on test set.

This script evaluates the fully trained baseline models (mit_b0, mit_b1, mit_b2)
that were trained for 15 epochs with all 3 modalities (RGB+Thermal+Event).

Models are located in:
  - training_logs/run_20251107_102948/mit_b0/checkpoints/best_model.pth
  - training_logs/run_20251107_102948/mit_b1/checkpoints/best_model.pth
  - training_logs/run_20251107_102948/mit_b2/checkpoints/best_model.pth
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from crossattentiondet.data.dataset import NpyYoloDataset, collate_fn
from crossattentiondet.models.encoder import RGBXTransformer
from crossattentiondet.models.detection_head import FasterRCNNHead
from crossattentiondet.utils.metrics import evaluate


class BaselineEvaluator:
    """Evaluate 15-epoch baseline models."""

    def __init__(self, data_dir, output_dir="results/baseline_15epoch_evaluated"):
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Model configurations
        self.models_to_eval = [
            {
                'name': 'mit_b0',
                'checkpoint': 'training_logs/run_20251107_102948/mit_b0/checkpoints/best_model.pth',
                'training_info': 'training_logs/run_20251107_102948/mit_b0/logs/training_info.json'
            },
            {
                'name': 'mit_b1',
                'checkpoint': 'training_logs/run_20251107_102948/mit_b1/checkpoints/best_model.pth',
                'training_info': 'training_logs/run_20251107_102948/mit_b1/logs/training_info.json'
            },
            {
                'name': 'mit_b2',
                'checkpoint': 'training_logs/run_20251107_102948/mit_b2/checkpoints/best_model.pth',
                'training_info': 'training_logs/run_20251107_102948/mit_b2/logs/training_info.json'
            }
        ]

        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Setup test dataset
        print(f"\nLoading test dataset from {data_dir}...")
        image_dir = os.path.join(data_dir, 'images')
        label_dir = os.path.join(data_dir, 'labels')

        self.test_dataset = NpyYoloDataset(
            image_dir,
            label_dir,
            mode='test',
            test_size=0.2,
            random_state=42
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=1
        )

        print(f"Test dataset: {len(self.test_dataset)} images")

        # Auto-detect number of classes
        all_labels = []
        for _, label_path in self.test_dataset.samples[:100]:  # Check first 100
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            all_labels.append(int(parts[0]))

        self.num_classes = len(set(all_labels)) + 1  # +1 for background
        print(f"Detected {self.num_classes} classes (including background)")

    def build_model(self, backbone_name):
        """Build model with specified backbone."""
        print(f"\nBuilding model with backbone: {backbone_name}")

        # Build encoder
        encoder = RGBXTransformer(backbone_name=backbone_name)

        # Build detection head
        model = FasterRCNNHead(
            encoder=encoder,
            num_classes=self.num_classes,
            fpn_out_channels=256
        )

        return model

    def load_checkpoint(self, model, checkpoint_path):
        """Load model checkpoint."""
        print(f"Loading checkpoint: {checkpoint_path}")

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                # Assume the dict IS the state dict
                model.load_state_dict(checkpoint)
        else:
            # Direct state dict
            model.load_state_dict(checkpoint)

        print("Checkpoint loaded successfully")
        return model

    def load_training_info(self, training_info_path):
        """Load training information."""
        if os.path.exists(training_info_path):
            with open(training_info_path, 'r') as f:
                return json.load(f)
        return {}

    def evaluate_model(self, model_config):
        """Evaluate a single model."""
        name = model_config['name']
        checkpoint_path = model_config['checkpoint']
        training_info_path = model_config['training_info']

        print("\n" + "="*80)
        print(f"EVALUATING: {name.upper()}")
        print("="*80)

        # Check if checkpoint exists
        if not os.path.exists(checkpoint_path):
            print(f"❌ Checkpoint not found: {checkpoint_path}")
            return None

        # Load training info
        training_info = self.load_training_info(training_info_path)

        # Build model
        model = self.build_model(name)
        model = self.load_checkpoint(model, checkpoint_path)
        model.to(self.device)
        model.eval()

        # Create output directory for this model
        model_output_dir = os.path.join(self.output_dir, name)
        os.makedirs(model_output_dir, exist_ok=True)

        # Run evaluation
        print("\nRunning evaluation on test set...")
        start_time = time.time()

        eval_results = evaluate(
            model,
            self.test_loader,
            self.device,
            results_dir=model_output_dir
        )

        eval_time = time.time() - start_time

        # Compile full results
        results = {
            'experiment_id': f'baseline_15epoch_{name}',
            'backbone': name,
            'architecture': 'baseline_FRM_FFM',
            'config': {
                'epochs': training_info.get('config', {}).get('num_epochs', 15),
                'batch_size': training_info.get('config', {}).get('batch_size', 16),
                'learning_rate': training_info.get('config', {}).get('learning_rate', 0.02),
                'modalities': ['rgb', 'thermal', 'event'],
                'modality_config': 'rgb+thermal+event (all 3)'
            },
            'training': {
                'best_epoch': training_info.get('training', {}).get('best_epoch', 15),
                'best_loss': training_info.get('training', {}).get('best_loss', 'N/A'),
                'final_loss': training_info.get('training', {}).get('best_loss', 'N/A'),
                'total_time_hours': training_info.get('total_training_time_hours', 'N/A'),
                'total_epochs': training_info.get('training', {}).get('total_epochs_completed', 15)
            },
            'model': {
                'total_params': training_info.get('model', {}).get('total_parameters', 'N/A'),
                'trainable_params': training_info.get('model', {}).get('trainable_parameters', 'N/A'),
                'total_params_M': training_info.get('model', {}).get('parameters_M', 'N/A'),
                'trainable_params_M': training_info.get('model', {}).get('parameters_M', 'N/A')
            },
            'evaluation': {
                'metrics': eval_results,
                'eval_time_sec': eval_time,
                'eval_time_min': eval_time / 60,
                'num_test_images': len(self.test_dataset)
            },
            'checkpoint_path': checkpoint_path,
            'timestamp': datetime.now().isoformat()
        }

        # Save results
        results_file = os.path.join(model_output_dir, 'final_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ Results saved to: {results_file}")

        # Print summary
        print("\n" + "-"*80)
        print("EVALUATION SUMMARY")
        print("-"*80)
        print(f"Backbone: {name}")
        print(f"Parameters: {results['model']['total_params_M']:.2f}M" if isinstance(results['model']['total_params_M'], (int, float)) else f"Parameters: {results['model']['total_params_M']}")
        print(f"Training Loss: {results['training']['best_loss']:.4f}" if isinstance(results['training']['best_loss'], (int, float)) else f"Training Loss: {results['training']['best_loss']}")
        print(f"Training Time: {results['training']['total_time_hours']:.2f}h" if isinstance(results['training']['total_time_hours'], (int, float)) else f"Training Time: {results['training']['total_time_hours']}")
        print(f"\nTest Set Performance:")
        print(f"  mAP:       {eval_results.get('mAP', 'N/A'):.4f}" if isinstance(eval_results.get('mAP'), (int, float)) else f"  mAP:       {eval_results.get('mAP', 'N/A')}")
        print(f"  mAP@50:    {eval_results.get('mAP_50', 'N/A'):.4f}" if isinstance(eval_results.get('mAP_50'), (int, float)) else f"  mAP@50:    {eval_results.get('mAP_50', 'N/A')}")
        print(f"  mAP@75:    {eval_results.get('mAP_75', 'N/A'):.4f}" if isinstance(eval_results.get('mAP_75'), (int, float)) else f"  mAP@75:    {eval_results.get('mAP_75', 'N/A')}")
        print(f"  mAP_small: {eval_results.get('mAP_small', 'N/A'):.4f}" if isinstance(eval_results.get('mAP_small'), (int, float)) else f"  mAP_small: {eval_results.get('mAP_small', 'N/A')}")
        print(f"  mAP_medium:{eval_results.get('mAP_medium', 'N/A'):.4f}" if isinstance(eval_results.get('mAP_medium'), (int, float)) else f"  mAP_medium:{eval_results.get('mAP_medium', 'N/A')}")
        print(f"  mAP_large: {eval_results.get('mAP_large', 'N/A'):.4f}" if isinstance(eval_results.get('mAP_large'), (int, float)) else f"  mAP_large: {eval_results.get('mAP_large', 'N/A')}")
        print(f"\nEvaluation Time: {eval_time:.2f}s ({eval_time/60:.2f} min)")
        print("-"*80)

        return results

    def run(self):
        """Run evaluation on all models."""
        print("\n" + "="*80)
        print("15-EPOCH BASELINE MODEL EVALUATION")
        print("="*80)
        print(f"Output directory: {self.output_dir}")
        print(f"Models to evaluate: {len(self.models_to_eval)}")

        all_results = []
        successful = 0
        failed = 0

        start_time = time.time()

        for model_config in self.models_to_eval:
            try:
                result = self.evaluate_model(model_config)
                if result:
                    all_results.append(result)
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"\n❌ ERROR evaluating {model_config['name']}: {str(e)}")
                import traceback
                traceback.print_exc()
                failed += 1

        total_time = time.time() - start_time

        # Create summary
        summary = {
            'evaluation_date': datetime.now().isoformat(),
            'total_models': len(self.models_to_eval),
            'successful': successful,
            'failed': failed,
            'total_time_sec': total_time,
            'total_time_min': total_time / 60,
            'total_time_hours': total_time / 3600,
            'models': []
        }

        # Add model summaries
        for result in all_results:
            summary['models'].append({
                'backbone': result['backbone'],
                'mAP': result['evaluation']['metrics'].get('mAP', 'N/A'),
                'mAP_50': result['evaluation']['metrics'].get('mAP_50', 'N/A'),
                'mAP_75': result['evaluation']['metrics'].get('mAP_75', 'N/A'),
                'training_loss': result['training']['best_loss'],
                'training_time_hours': result['training']['total_time_hours'],
                'params_M': result['model']['total_params_M']
            })

        # Save summary
        summary_file = os.path.join(self.output_dir, 'evaluation_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        # Print final summary
        print("\n" + "="*80)
        print("FINAL SUMMARY")
        print("="*80)
        print(f"Total models evaluated: {successful}/{len(self.models_to_eval)}")
        print(f"Failed: {failed}")
        print(f"Total time: {total_time/60:.2f} minutes ({total_time/3600:.2f} hours)")
        print(f"\nResults saved to: {self.output_dir}")
        print(f"Summary: {summary_file}")

        if all_results:
            print("\n" + "-"*80)
            print("MODEL COMPARISON")
            print("-"*80)
            print(f"{'Backbone':<12} {'Params':<10} {'Loss':<10} {'mAP':<10} {'mAP@50':<10} {'mAP@75':<10}")
            print("-"*80)
            for result in sorted(all_results, key=lambda x: x['evaluation']['metrics'].get('mAP', 0), reverse=True):
                name = result['backbone']
                params = result['model']['total_params_M']
                loss = result['training']['best_loss']
                mAP = result['evaluation']['metrics'].get('mAP', 0)
                mAP50 = result['evaluation']['metrics'].get('mAP_50', 0)
                mAP75 = result['evaluation']['metrics'].get('mAP_75', 0)

                params_str = f"{params:.1f}M" if isinstance(params, (int, float)) else str(params)
                loss_str = f"{loss:.4f}" if isinstance(loss, (int, float)) else str(loss)
                mAP_str = f"{mAP:.4f}" if isinstance(mAP, (int, float)) else str(mAP)
                mAP50_str = f"{mAP50:.4f}" if isinstance(mAP50, (int, float)) else str(mAP50)
                mAP75_str = f"{mAP75:.4f}" if isinstance(mAP75, (int, float)) else str(mAP75)

                print(f"{name:<12} {params_str:<10} {loss_str:<10} {mAP_str:<10} {mAP50_str:<10} {mAP75_str:<10}")
            print("="*80)

        return summary


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate 15-epoch baseline models')
    parser.add_argument(
        '--data',
        type=str,
        default='../RGBX_Semantic_Segmentation/data',
        help='Path to data directory'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/baseline_15epoch_evaluated',
        help='Output directory for results'
    )

    args = parser.parse_args()

    # Run evaluation
    evaluator = BaselineEvaluator(args.data, args.output)
    summary = evaluator.run()

    print("\n✓ Evaluation complete!")

    return 0 if summary['successful'] > 0 else 1


if __name__ == '__main__':
    sys.exit(main())
