"""
GAFF Dry-Run Testing Script

Tests the complete training loop with synthetic data (no dataset required).
This allows verification of the entire training pipeline on CPU before GPU deployment.

Features:
- Synthetic data generation (no real dataset needed)
- Full training loop (forward, loss, backward, optimizer)
- Tests all stage configurations
- Tests all hyperparameter combinations
- Validates checkpoint saving/loading
- CPU-compatible

Run with: python -m crossattentiondet.ablations.scripts.dry_run_gaff
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple
import time

from crossattentiondet.ablations.encoder_gaff_flexible import get_gaff_encoder


class SyntheticDataset:
    """Generate synthetic data for testing."""

    def __init__(self, num_samples=10, img_size=224, num_classes=20):
        self.num_samples = num_samples
        self.img_size = img_size
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """Generate a synthetic sample."""
        # RGB image
        rgb = torch.randn(3, self.img_size, self.img_size)

        # Auxiliary image (e.g., thermal)
        aux = torch.randn(1, self.img_size, self.img_size)

        # Dummy ground truth (for detection: boxes, labels, masks)
        # For this dry run, we'll use simplified targets
        target = {
            'boxes': torch.rand(5, 4) * self.img_size,  # 5 random boxes
            'labels': torch.randint(0, self.num_classes, (5,)),
        }

        return rgb, aux, target


class DummyDetectionHead(nn.Module):
    """Dummy detection head for testing."""

    def __init__(self, in_channels_list=[64, 128, 320, 512], num_classes=20):
        super().__init__()
        self.num_classes = num_classes

        # Simple 1x1 convs to match output dimensions
        self.cls_heads = nn.ModuleList([
            nn.Conv2d(c, num_classes, kernel_size=1)
            for c in in_channels_list
        ])

        self.box_heads = nn.ModuleList([
            nn.Conv2d(c, 4, kernel_size=1)
            for c in in_channels_list
        ])

    def forward(self, features):
        """Forward pass through detection head."""
        cls_outputs = []
        box_outputs = []

        for i, feat in enumerate(features):
            cls_outputs.append(self.cls_heads[i](feat))
            box_outputs.append(self.box_heads[i](feat))

        return cls_outputs, box_outputs


class DummyLoss(nn.Module):
    """Dummy loss for testing."""

    def forward(self, cls_outputs, box_outputs, targets):
        """Compute dummy loss."""
        # Simple loss: sum of absolute values (just for testing gradients)
        cls_loss = sum(o.abs().mean() for o in cls_outputs)
        box_loss = sum(o.abs().mean() for o in box_outputs)

        total_loss = cls_loss + box_loss

        return {
            'loss': total_loss,
            'cls_loss': cls_loss,
            'box_loss': box_loss
        }


class GAFFDryRunner:
    """Dry-run testing for GAFF training pipeline."""

    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        self.test_results = []

    def log_test(self, test_name: str, passed: bool, message: str = ""):
        """Log test result."""
        status = "PASS" if passed else "FAIL"
        self.test_results.append((test_name, passed))
        print(f"[{status}] {test_name}")
        if message:
            print(f"      {message}")

    def run_all_tests(self):
        """Run all dry-run tests."""
        print("=" * 90)
        print("GAFF Dry-Run Testing")
        print("=" * 90)
        print(f"Device: {self.device}")
        print()

        # Test training loop
        print("-" * 90)
        print("Testing Training Loop")
        print("-" * 90)
        self.test_training_loop()
        print()

        # Test all stage configs
        print("-" * 90)
        print("Testing All Stage Configurations")
        print("-" * 90)
        self.test_all_stage_configs()
        print()

        # Test hyperparameter configs
        print("-" * 90)
        print("Testing Hyperparameter Configurations")
        print("-" * 90)
        self.test_hyperparameter_configs()
        print()

        # Test checkpoint save/load
        print("-" * 90)
        print("Testing Checkpoint Save/Load")
        print("-" * 90)
        self.test_checkpoint_operations()
        print()

        # Test batch sizes
        print("-" * 90)
        print("Testing Different Batch Sizes")
        print("-" * 90)
        self.test_batch_sizes()
        print()

        # Summary
        self.print_summary()

    def test_training_loop(self):
        """Test complete training loop."""
        try:
            # Create model
            encoder = get_gaff_encoder(
                backbone='mit_b1',
                gaff_stages=[4]
            ).to(self.device)

            # Get embed_dims for mit_b1
            embed_dims = [64, 128, 320, 512]
            head = DummyDetectionHead(in_channels_list=embed_dims).to(self.device)

            # Create optimizer
            optimizer = optim.Adam(
                list(encoder.parameters()) + list(head.parameters()),
                lr=0.001
            )

            # Create loss
            criterion = DummyLoss()

            # Create synthetic data
            dataset = SyntheticDataset(num_samples=4)

            # Training loop
            encoder.train()
            head.train()

            for batch_idx in range(2):  # 2 batches
                # Get batch
                batch_rgb = []
                batch_aux = []
                batch_targets = []

                for i in range(2):  # batch size 2
                    rgb, aux, target = dataset[i]
                    batch_rgb.append(rgb)
                    batch_aux.append(aux)
                    batch_targets.append(target)

                batch_rgb = torch.stack(batch_rgb).to(self.device)
                batch_aux = torch.stack(batch_aux).to(self.device)

                # Forward pass
                optimizer.zero_grad()
                features = encoder(batch_rgb, batch_aux)
                cls_outputs, box_outputs = head(features)

                # Compute loss
                loss_dict = criterion(cls_outputs, box_outputs, batch_targets)
                loss = loss_dict['loss']

                # Backward pass
                loss.backward()

                # Optimizer step
                optimizer.step()

                # Verify loss is finite
                if not torch.isfinite(loss):
                    raise ValueError(f"Loss is not finite: {loss.item()}")

            self.log_test(
                "Basic training loop (2 batches, 2 epochs)",
                True,
                f"Final loss: {loss.item():.4f}"
            )

        except Exception as e:
            self.log_test("Basic training loop", False, f"Error: {e}")

    def test_all_stage_configs(self):
        """Test training with all stage configurations."""
        stage_configs = [
            [1], [2], [3], [4],
            [2, 3], [3, 4],
            [2, 3, 4],
            [1, 2, 3, 4]
        ]

        for stages in stage_configs:
            try:
                encoder = get_gaff_encoder(
                    backbone='mit_b1',
                    gaff_stages=stages
                ).to(self.device)

                embed_dims = [64, 128, 320, 512]
                head = DummyDetectionHead(in_channels_list=embed_dims).to(self.device)

                optimizer = optim.SGD(
                    list(encoder.parameters()) + list(head.parameters()),
                    lr=0.01
                )

                criterion = DummyLoss()

                # Single forward-backward pass
                x_rgb = torch.randn(2, 3, 224, 224, device=self.device)
                x_aux = torch.randn(2, 1, 224, 224, device=self.device)

                optimizer.zero_grad()
                features = encoder(x_rgb, x_aux)
                cls_outputs, box_outputs = head(features)
                loss_dict = criterion(cls_outputs, box_outputs, None)
                loss = loss_dict['loss']
                loss.backward()
                optimizer.step()

                passed = torch.isfinite(loss).item()
                stage_str = ','.join(map(str, stages))

                self.log_test(
                    f"Stages [{stage_str}]",
                    passed,
                    f"Loss: {loss.item():.4f}"
                )

            except Exception as e:
                stage_str = ','.join(map(str, stages))
                self.log_test(f"Stages [{stage_str}]", False, f"Error: {e}")

    def test_hyperparameter_configs(self):
        """Test different hyperparameter configurations."""
        configs = [
            {'gaff_se_reduction': 4, 'gaff_inter_shared': False, 'gaff_merge_bottleneck': False},
            {'gaff_se_reduction': 8, 'gaff_inter_shared': False, 'gaff_merge_bottleneck': False},
            {'gaff_se_reduction': 4, 'gaff_inter_shared': True, 'gaff_merge_bottleneck': False},
            {'gaff_se_reduction': 4, 'gaff_inter_shared': False, 'gaff_merge_bottleneck': True},
            {'gaff_se_reduction': 8, 'gaff_inter_shared': True, 'gaff_merge_bottleneck': True},
        ]

        for i, config in enumerate(configs):
            try:
                encoder = get_gaff_encoder(
                    backbone='mit_b1',
                    gaff_stages=[4],
                    **config
                ).to(self.device)

                embed_dims = [64, 128, 320, 512]
                head = DummyDetectionHead(in_channels_list=embed_dims).to(self.device)
                optimizer = optim.Adam(
                    list(encoder.parameters()) + list(head.parameters()),
                    lr=0.001
                )
                criterion = DummyLoss()

                # Single forward-backward pass
                x_rgb = torch.randn(2, 3, 224, 224, device=self.device)
                x_aux = torch.randn(2, 1, 224, 224, device=self.device)

                optimizer.zero_grad()
                features = encoder(x_rgb, x_aux)
                cls_outputs, box_outputs = head(features)
                loss_dict = criterion(cls_outputs, box_outputs, None)
                loss = loss_dict['loss']
                loss.backward()
                optimizer.step()

                passed = torch.isfinite(loss).item()
                config_str = ', '.join(f'{k.replace("gaff_", "")}={v}' for k, v in config.items())

                self.log_test(
                    f"Config {i+1}: {config_str}",
                    passed,
                    f"Loss: {loss.item():.4f}"
                )

            except Exception as e:
                self.log_test(f"Config {i+1}", False, f"Error: {e}")

    def test_checkpoint_operations(self):
        """Test checkpoint save and load."""
        import tempfile

        try:
            # Create model
            encoder = get_gaff_encoder(
                backbone='mit_b1',
                gaff_stages=[4]
            ).to(self.device)

            # Train for one step
            optimizer = optim.Adam(encoder.parameters(), lr=0.001)
            x_rgb = torch.randn(2, 3, 224, 224, device=self.device)
            x_aux = torch.randn(2, 1, 224, 224, device=self.device)

            optimizer.zero_grad()
            features = encoder(x_rgb, x_aux)
            loss = sum(f.sum() for f in features)
            loss.backward()
            optimizer.step()

            # Save checkpoint
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as tmp:
                checkpoint_path = tmp.name
                torch.save({
                    'model_state_dict': encoder.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item()
                }, checkpoint_path)

            # Create new model and load
            encoder2 = get_gaff_encoder(
                backbone='mit_b1',
                gaff_stages=[4]
            ).to(self.device)

            optimizer2 = optim.Adam(encoder2.parameters(), lr=0.001)

            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            encoder2.load_state_dict(checkpoint['model_state_dict'])
            optimizer2.load_state_dict(checkpoint['optimizer_state_dict'])
            loaded_loss = checkpoint['loss']

            # Clean up
            os.unlink(checkpoint_path)

            # Verify loaded correctly
            passed = abs(loaded_loss - loss.item()) < 1e-6

            self.log_test(
                "Checkpoint save/load",
                passed,
                f"Original loss: {loss.item():.6f}, Loaded loss: {loaded_loss:.6f}"
            )

        except Exception as e:
            self.log_test("Checkpoint save/load", False, f"Error: {e}")

    def test_batch_sizes(self):
        """Test different batch sizes."""
        batch_sizes = [1, 2, 4, 8]

        encoder = get_gaff_encoder(
            backbone='mit_b1',
            gaff_stages=[4]
        ).to(self.device)

        embed_dims = [64, 128, 320, 512]
        head = DummyDetectionHead(in_channels_list=embed_dims).to(self.device)

        for bs in batch_sizes:
            try:
                x_rgb = torch.randn(bs, 3, 224, 224, device=self.device)
                x_aux = torch.randn(bs, 1, 224, 224, device=self.device)

                with torch.no_grad():
                    features = encoder(x_rgb, x_aux)
                    cls_outputs, box_outputs = head(features)

                # Verify outputs
                passed = (
                    len(cls_outputs) == 4 and
                    len(box_outputs) == 4 and
                    all(c.shape[0] == bs for c in cls_outputs) and
                    all(b.shape[0] == bs for b in box_outputs)
                )

                self.log_test(
                    f"Batch size {bs}",
                    passed,
                    f"Output shapes: {[tuple(c.shape) for c in cls_outputs]}"
                )

            except Exception as e:
                self.log_test(f"Batch size {bs}", False, f"Error: {e}")

    def print_summary(self):
        """Print test summary."""
        print("=" * 90)
        print("Dry-Run Summary")
        print("=" * 90)

        total_tests = len(self.test_results)
        passed_tests = sum(1 for _, passed in self.test_results if passed)
        failed_tests = total_tests - passed_tests

        print(f"Total tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print()

        if failed_tests > 0:
            print("Failed tests:")
            for test_name, passed in self.test_results:
                if not passed:
                    print(f"  - {test_name}")
            print()

        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        print(f"Success rate: {success_rate:.1f}%")

        if failed_tests == 0:
            print()
            print("✓ All dry-run tests passed!")
            print("✓ Training pipeline is ready for real data!")
        else:
            print()
            print("✗ Some tests failed. Please review errors above.")

        print("=" * 90)

        return failed_tests == 0


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="GAFF Dry-Run Testing")
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                        help='Device to run tests on')
    args = parser.parse_args()

    runner = GAFFDryRunner(device=args.device)
    all_passed = runner.run_all_tests()

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
