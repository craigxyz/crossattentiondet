#!/usr/bin/env python
"""
Comprehensive training script for all backbone variants with detailed per-epoch logging.

This script trains each MiT backbone variant (mit_b0 through mit_b5) for a specified
number of epochs with:
- Per-epoch metrics logging (loss, time, learning rate)
- Checkpoint saving after each epoch
- Best model tracking
- Final weight saving for each backbone
- Comprehensive CSV and JSON logs
- Training curves and summary reports

Usage:
    python scripts/train_all_backbones_comprehensive.py --data data/ --epochs 30
    python scripts/train_all_backbones_comprehensive.py --data data/ --epochs 50 --batch-size 4 --lr 0.005
    python scripts/train_all_backbones_comprehensive.py --data data/ --epochs 30 --backbones mit_b0 mit_b1
"""

import os
import sys
import argparse
import time
import json
import csv
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Add parent directory to path to import crossattentiondet
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crossattentiondet.config import Config
from crossattentiondet.models.encoder import get_encoder
from crossattentiondet.models.backbone import CrossAttentionBackbone
from crossattentiondet.data.dataset import NpyYoloDataset, collate_fn

from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign


class ComprehensiveTrainer:
    """Enhanced trainer with detailed logging and checkpoint management."""

    def __init__(self, config, backbone_name, log_dir):
        self.config = config
        self.backbone_name = backbone_name
        self.log_dir = log_dir

        # Create directories
        self.checkpoint_dir = os.path.join(log_dir, backbone_name, 'checkpoints')
        self.log_file_dir = os.path.join(log_dir, backbone_name, 'logs')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_file_dir, exist_ok=True)

        # Setup logging files
        self.epoch_log_csv = os.path.join(self.log_file_dir, 'epoch_metrics.csv')
        self.batch_log_csv = os.path.join(self.log_file_dir, 'batch_metrics.csv')
        self.training_info_json = os.path.join(self.log_file_dir, 'training_info.json')

        # Initialize CSV files
        self._init_csv_files()

        # Training state
        self.best_loss = float('inf')
        self.best_epoch = -1
        self.epoch_losses = []
        self.batch_losses = []
        self.epoch_times = []

        # Auto-detect number of classes
        config.auto_detect_num_classes()

        # Setup dataset and dataloader
        print(f"\n[{backbone_name}] Setting up dataset...")
        self.train_dataset = NpyYoloDataset(
            config.image_dir,
            config.label_dir,
            mode='train',
            test_size=config.test_size,
            random_state=config.random_state
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=config.num_workers if hasattr(config, 'num_workers') else 0
        )

        # Build model
        print(f"[{backbone_name}] Building model...")
        self.model = self._build_model()
        self.model.to(config.device)

        # Count parameters
        self.total_params = sum(p.numel() for p in self.model.parameters())
        self.trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[{backbone_name}] Total parameters: {self.total_params:,} ({self.total_params/1e6:.2f}M)")
        print(f"[{backbone_name}] Trainable parameters: {self.trainable_params:,} ({self.trainable_params/1e6:.2f}M)")

        # Setup optimizer and scheduler
        print(f"[{backbone_name}] Setting up optimizer...")
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay
        )

        # Optional: Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config.lr_step_size if hasattr(config, 'lr_step_size') else 10,
            gamma=config.lr_gamma if hasattr(config, 'lr_gamma') else 0.1
        )

        # Save initial training info
        self._save_training_info(status='initialized')

    def _init_csv_files(self):
        """Initialize CSV log files with headers."""
        # Epoch metrics CSV
        with open(self.epoch_log_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch', 'avg_loss', 'total_loss', 'epoch_time_sec',
                'learning_rate', 'is_best', 'checkpoint_path', 'timestamp'
            ])

        # Batch metrics CSV
        with open(self.batch_log_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch', 'batch', 'loss', 'loss_classifier', 'loss_box_reg',
                'loss_objectness', 'loss_rpn_box_reg', 'timestamp'
            ])

    def _build_model(self):
        """Build the CrossAttentionDet model."""
        config = self.config

        # Instantiate the encoder
        encoder_base = get_encoder(
            self.backbone_name,
            in_chans_rgb=config.in_chans_rgb,
            in_chans_x=config.in_chans_x
        )

        # Wrap with FPN
        backbone = CrossAttentionBackbone(encoder_base, fpn_out_channels=config.fpn_out_channels)

        # Define anchor generator
        anchor_generator = AnchorGenerator(
            sizes=config.anchor_sizes,
            aspect_ratios=config.anchor_aspect_ratios
        )

        # Define RoI pooling
        roi_pooler = MultiScaleRoIAlign(
            featmap_names=config.roi_featmap_names,
            output_size=config.roi_output_size,
            sampling_ratio=config.roi_sampling_ratio
        )

        # Create Faster R-CNN model
        model = FasterRCNN(
            backbone,
            num_classes=config.num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler,
            image_mean=config.image_mean,
            image_std=config.image_std
        )

        return model

    def _save_training_info(self, status='training', additional_info=None):
        """Save training information to JSON file."""
        info = {
            'backbone': self.backbone_name,
            'status': status,
            'timestamp': datetime.now().isoformat(),
            'config': {
                'num_epochs': self.config.num_epochs,
                'batch_size': self.config.batch_size,
                'learning_rate': self.config.learning_rate,
                'momentum': self.config.momentum,
                'weight_decay': self.config.weight_decay,
                'device': str(self.config.device),
                'num_classes': self.config.num_classes,
                'fpn_out_channels': self.config.fpn_out_channels,
            },
            'model': {
                'total_parameters': self.total_params,
                'trainable_parameters': self.trainable_params,
                'parameters_M': round(self.total_params / 1e6, 2),
            },
            'training': {
                'best_epoch': self.best_epoch,
                'best_loss': self.best_loss,
                'total_epochs_completed': len(self.epoch_losses),
                'epoch_losses': self.epoch_losses,
                'epoch_times': self.epoch_times,
            }
        }

        if additional_info:
            info.update(additional_info)

        with open(self.training_info_json, 'w') as f:
            json.dump(info, f, indent=2)

    def train_epoch(self, epoch):
        """Train for one epoch with detailed logging."""
        self.model.train()
        epoch_start_time = time.time()

        total_loss = 0
        batch_losses = []

        print(f"\n[{self.backbone_name}] Epoch {epoch+1}/{self.config.num_epochs}")
        print("-" * 80)

        for batch_idx, (images, targets) in enumerate(self.train_loader):
            batch_start_time = time.time()

            # Move to device
            images = list(image.to(self.config.device) for image in images)
            targets = [{k: v.to(self.config.device) for k, v in t.items()} for t in targets]

            # Forward pass
            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # Backward pass
            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()

            # Record metrics
            loss_value = losses.item()
            total_loss += loss_value
            batch_losses.append(loss_value)

            # Log batch metrics to CSV
            with open(self.batch_log_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch + 1,
                    batch_idx + 1,
                    loss_value,
                    loss_dict.get('loss_classifier', 0).item() if 'loss_classifier' in loss_dict else 0,
                    loss_dict.get('loss_box_reg', 0).item() if 'loss_box_reg' in loss_dict else 0,
                    loss_dict.get('loss_objectness', 0).item() if 'loss_objectness' in loss_dict else 0,
                    loss_dict.get('loss_rpn_box_reg', 0).item() if 'loss_rpn_box_reg' in loss_dict else 0,
                    datetime.now().isoformat()
                ])

            # Print progress
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(self.train_loader):
                batch_time = time.time() - batch_start_time
                print(f"  Batch [{batch_idx+1}/{len(self.train_loader)}] "
                      f"Loss: {loss_value:.4f} "
                      f"Time: {batch_time:.2f}s")

        # Epoch statistics
        avg_loss = total_loss / len(self.train_loader)
        epoch_time = time.time() - epoch_start_time
        current_lr = self.optimizer.param_groups[0]['lr']

        # Update learning rate
        self.scheduler.step()

        # Track if this is the best epoch
        is_best = avg_loss < self.best_loss
        if is_best:
            self.best_loss = avg_loss
            self.best_epoch = epoch + 1

        # Save epoch checkpoint
        checkpoint_path = self._save_epoch_checkpoint(epoch + 1, avg_loss, is_best)

        # Log epoch metrics to CSV
        with open(self.epoch_log_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                avg_loss,
                total_loss,
                epoch_time,
                current_lr,
                is_best,
                checkpoint_path,
                datetime.now().isoformat()
            ])

        # Update internal state
        self.epoch_losses.append(avg_loss)
        self.epoch_times.append(epoch_time)

        # Print epoch summary
        print("-" * 80)
        print(f"[{self.backbone_name}] Epoch {epoch+1} Summary:")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Total Loss: {total_loss:.4f}")
        print(f"  Epoch Time: {epoch_time:.2f}s ({epoch_time/60:.2f}min)")
        print(f"  Learning Rate: {current_lr:.6f}")
        print(f"  Best Loss: {self.best_loss:.4f} (Epoch {self.best_epoch})")
        if is_best:
            print(f"  *** NEW BEST MODEL ***")
        print("-" * 80)

        return avg_loss, epoch_time

    def _save_epoch_checkpoint(self, epoch, loss, is_best):
        """Save checkpoint for current epoch."""
        checkpoint = {
            'epoch': epoch,
            'backbone': self.backbone_name,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'best_loss': self.best_loss,
            'best_epoch': self.best_epoch,
            'config': vars(self.config)
        }

        # Save epoch checkpoint
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f'epoch_{epoch:03d}_loss_{loss:.4f}.pth'
        )
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"  Saved best model to: {best_path}")

        return checkpoint_path

    def train(self):
        """Full training loop."""
        print(f"\n{'='*80}")
        print(f"Starting training for {self.backbone_name.upper()}")
        print(f"{'='*80}")
        print(f"Total epochs: {self.config.num_epochs}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Initial learning rate: {self.config.learning_rate}")
        print(f"Device: {self.config.device}")
        print(f"{'='*80}\n")

        training_start_time = time.time()

        try:
            for epoch in range(self.config.num_epochs):
                avg_loss, epoch_time = self.train_epoch(epoch)

                # Update training info after each epoch
                self._save_training_info(status='training')

            # Training completed successfully
            total_training_time = time.time() - training_start_time

            # Save final model
            final_model_path = self._save_final_model()

            # Save final training info
            self._save_training_info(
                status='completed',
                additional_info={
                    'total_training_time_sec': total_training_time,
                    'total_training_time_min': total_training_time / 60,
                    'total_training_time_hours': total_training_time / 3600,
                    'final_model_path': final_model_path,
                    'best_model_path': os.path.join(self.checkpoint_dir, 'best_model.pth'),
                }
            )

            print(f"\n{'='*80}")
            print(f"Training completed for {self.backbone_name.upper()}")
            print(f"{'='*80}")
            print(f"Total training time: {total_training_time/60:.2f} minutes ({total_training_time/3600:.2f} hours)")
            print(f"Best epoch: {self.best_epoch}")
            print(f"Best loss: {self.best_loss:.4f}")
            print(f"Final model saved to: {final_model_path}")
            print(f"Best model saved to: {os.path.join(self.checkpoint_dir, 'best_model.pth')}")
            print(f"Logs saved to: {self.log_file_dir}")
            print(f"{'='*80}\n")

            return {
                'success': True,
                'backbone': self.backbone_name,
                'total_time_sec': total_training_time,
                'best_epoch': self.best_epoch,
                'best_loss': self.best_loss,
                'final_loss': self.epoch_losses[-1],
                'final_model_path': final_model_path,
                'best_model_path': os.path.join(self.checkpoint_dir, 'best_model.pth'),
                'log_dir': self.log_file_dir
            }

        except Exception as e:
            total_training_time = time.time() - training_start_time
            print(f"\n[ERROR] Training failed for {self.backbone_name}: {e}")

            self._save_training_info(
                status='failed',
                additional_info={
                    'error': str(e),
                    'time_elapsed_sec': total_training_time
                }
            )

            return {
                'success': False,
                'backbone': self.backbone_name,
                'error': str(e),
                'total_time_sec': total_training_time
            }

    def _save_final_model(self):
        """Save final trained model weights."""
        final_path = os.path.join(
            self.checkpoint_dir,
            f'{self.backbone_name}_final.pth'
        )

        # Save just the model state dict for inference
        torch.save(self.model.state_dict(), final_path)

        # Also save a complete checkpoint
        complete_path = os.path.join(
            self.checkpoint_dir,
            f'{self.backbone_name}_final_complete.pth'
        )
        torch.save({
            'backbone': self.backbone_name,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epoch': self.config.num_epochs,
            'best_loss': self.best_loss,
            'best_epoch': self.best_epoch,
            'final_loss': self.epoch_losses[-1],
            'epoch_losses': self.epoch_losses,
            'config': vars(self.config)
        }, complete_path)

        return final_path


def main():
    """Main function to train all backbone variants."""
    parser = argparse.ArgumentParser(
        description="Comprehensive training script for all MiT backbone variants",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data arguments
    parser.add_argument("--data", type=str, required=True,
                       help="Path to data directory containing images/ and labels/")

    # Training arguments
    parser.add_argument("--epochs", type=int, default=30,
                       help="Number of training epochs for each backbone")
    parser.add_argument("--batch-size", type=int, default=2,
                       help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.005,
                       help="Initial learning rate")
    parser.add_argument("--lr-step-size", type=int, default=10,
                       help="Learning rate scheduler step size")
    parser.add_argument("--lr-gamma", type=float, default=0.1,
                       help="Learning rate scheduler gamma")
    parser.add_argument("--num-workers", type=int, default=4,
                       help="Number of data loading workers")

    # Backbone selection
    parser.add_argument("--backbones", nargs='+',
                       default=['mit_b0', 'mit_b1', 'mit_b2', 'mit_b4', 'mit_b5'],
                       choices=['mit_b0', 'mit_b1', 'mit_b2', 'mit_b4', 'mit_b5'],
                       help="List of backbones to train (default: all)")

    # Output directory
    parser.add_argument("--log-dir", type=str, default='training_logs',
                       help="Directory to save training logs and checkpoints")

    args = parser.parse_args()

    # Create log directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.log_dir, f'run_{timestamp}')
    os.makedirs(log_dir, exist_ok=True)

    # Print configuration
    print("\n" + "="*80)
    print("COMPREHENSIVE TRAINING - ALL BACKBONE VARIANTS")
    print("="*80)
    print(f"Data directory: {args.data}")
    print(f"Epochs per backbone: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Initial learning rate: {args.lr}")
    print(f"LR step size: {args.lr_step_size}, gamma: {args.lr_gamma}")
    print(f"Backbones to train: {', '.join(args.backbones)}")
    print(f"Log directory: {log_dir}")
    print(f"Number of workers: {args.num_workers}")
    print("="*80)

    # Save run configuration
    run_config_path = os.path.join(log_dir, 'run_config.json')
    with open(run_config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    print(f"\nRun configuration saved to: {run_config_path}\n")

    # Track results for all backbones
    all_results = []
    total_start_time = time.time()

    # Train each backbone variant
    for i, backbone_name in enumerate(args.backbones, 1):
        print(f"\n\n{'#'*80}")
        print(f"# TRAINING BACKBONE {i}/{len(args.backbones)}: {backbone_name.upper()}")
        print(f"{'#'*80}\n")

        # Create base configuration
        config = Config()
        config.data_root = args.data
        config.image_dir = os.path.join(args.data, 'images')
        config.label_dir = os.path.join(args.data, 'labels')
        config.num_epochs = args.epochs
        config.batch_size = args.batch_size
        config.learning_rate = args.lr
        config.lr_step_size = args.lr_step_size
        config.lr_gamma = args.lr_gamma
        config.num_workers = args.num_workers

        # Create trainer and train
        trainer = ComprehensiveTrainer(config, backbone_name, log_dir)
        result = trainer.train()
        all_results.append(result)

        # Print progress
        completed = sum(1 for r in all_results if r['success'])
        failed = sum(1 for r in all_results if not r['success'])
        print(f"\nProgress: {i}/{len(args.backbones)} backbones processed "
              f"(✓ {completed} successful, ✗ {failed} failed)\n")

    # Print final summary
    total_elapsed = time.time() - total_start_time

    print("\n\n" + "="*80)
    print("FINAL TRAINING SUMMARY")
    print("="*80)

    print(f"\nTotal time: {total_elapsed/60:.2f} minutes ({total_elapsed/3600:.2f} hours)")
    print(f"Epochs per backbone: {args.epochs}")
    print(f"\nResults:")
    print("-"*80)

    for result in all_results:
        backbone = result['backbone']
        if result['success']:
            time_hours = result['total_time_sec'] / 3600
            print(f"  ✓ {backbone:8s} | {time_hours:6.2f}h | "
                  f"Best: {result['best_loss']:.4f} (ep {result['best_epoch']}) | "
                  f"Final: {result['final_loss']:.4f}")
        else:
            print(f"  ✗ {backbone:8s} | FAILED | {result.get('error', 'Unknown error')}")

    print("-"*80)

    successful = [r for r in all_results if r['success']]
    failed = [r for r in all_results if not r['success']]

    print(f"\nSuccessful: {len(successful)}/{len(all_results)}")
    print(f"Failed: {len(failed)}/{len(all_results)}")

    if successful:
        avg_time = sum(r['total_time_sec'] for r in successful) / len(successful)
        print(f"Average training time (successful): {avg_time/3600:.2f} hours")

        print("\nSpeed comparison (successful backbones):")
        sorted_results = sorted(successful, key=lambda x: x['total_time_sec'])
        for r in sorted_results:
            time_hours = r['total_time_sec'] / 3600
            print(f"  {r['backbone']:8s}: {time_hours:6.2f}h")

        print("\nBest performing models (by best loss):")
        sorted_by_loss = sorted(successful, key=lambda x: x['best_loss'])
        for r in sorted_by_loss:
            print(f"  {r['backbone']:8s}: {r['best_loss']:.4f} (epoch {r['best_epoch']})")

    # Save final summary
    summary_path = os.path.join(log_dir, 'final_summary.json')
    with open(summary_path, 'w') as f:
        json.dump({
            'run_config': vars(args),
            'total_time_sec': total_elapsed,
            'total_time_hours': total_elapsed / 3600,
            'results': all_results,
            'successful': len(successful),
            'failed': len(failed),
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)

    print(f"\nFinal summary saved to: {summary_path}")
    print(f"All logs saved to: {log_dir}")
    print("\n" + "="*80)
    print("ALL TRAINING COMPLETED")
    print("="*80 + "\n")

    # Return exit code based on results
    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
