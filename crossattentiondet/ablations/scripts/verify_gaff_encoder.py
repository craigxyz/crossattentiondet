"""
GAFF Encoder Verification Script

Standalone verification script that tests GAFF encoders without requiring a dataset.
Uses synthetic data to verify encoder functionality on CPU.

Features:
- Tests all backbones (mit_b0-b5)
- Tests all stage combinations
- Forward and backward pass validation
- Output shape verification
- Parameter count analysis
- No dataset required
- CPU-only compatible

Run with: python -m crossattentiondet.ablations.scripts.verify_gaff_encoder
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import torch
import torch.nn as nn
import time
from typing import List

from crossattentiondet.ablations.encoder_gaff_flexible import (
    mit_b0_gaff_flexible,
    mit_b1_gaff_flexible,
    mit_b2_gaff_flexible,
    mit_b3_gaff_flexible,
    mit_b4_gaff_flexible,
    mit_b5_gaff_flexible,
    get_gaff_encoder
)


class GAFFEncoderVerifier:
    """Comprehensive GAFF encoder verification."""

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

    def verify_all(self):
        """Run all verification tests."""
        print("=" * 90)
        print("GAFF Encoder Verification")
        print("=" * 90)
        print(f"Device: {self.device}")
        print()

        # Test all backbones
        print("-" * 90)
        print("Testing All Backbones")
        print("-" * 90)
        self.test_all_backbones()
        print()

        # Test stage configurations
        print("-" * 90)
        print("Testing Stage Configurations")
        print("-" * 90)
        self.test_stage_configurations()
        print()

        # Test hyperparameter configurations
        print("-" * 90)
        print("Testing Hyperparameter Configurations")
        print("-" * 90)
        self.test_hyperparameter_configs()
        print()

        # Test forward pass shapes
        print("-" * 90)
        print("Testing Output Shapes")
        print("-" * 90)
        self.test_output_shapes()
        print()

        # Test gradient flow
        print("-" * 90)
        print("Testing Gradient Flow")
        print("-" * 90)
        self.test_gradient_flow()
        print()

        # Parameter analysis
        print("-" * 90)
        print("Parameter Count Analysis")
        print("-" * 90)
        self.analyze_parameters()
        print()

        # Performance benchmarks
        print("-" * 90)
        print("CPU Inference Benchmarks")
        print("-" * 90)
        self.benchmark_inference()
        print()

        # Summary
        self.print_summary()

    def test_all_backbones(self):
        """Test all backbone variants."""
        backbones = ['mit_b0', 'mit_b1', 'mit_b2', 'mit_b3', 'mit_b4', 'mit_b5']

        for backbone in backbones:
            try:
                encoder = get_gaff_encoder(
                    backbone=backbone,
                    gaff_stages=[4],
                    gaff_se_reduction=4,
                    gaff_inter_shared=False,
                    gaff_merge_bottleneck=False
                ).to(self.device)

                # Test forward pass
                x_rgb = torch.randn(1, 3, 224, 224, device=self.device)
                x_aux = torch.randn(1, 1, 224, 224, device=self.device)

                with torch.no_grad():
                    outs = encoder(x_rgb, x_aux)

                # Verify 4 outputs
                passed = len(outs) == 4 and all(isinstance(o, torch.Tensor) for o in outs)
                params = sum(p.numel() for p in encoder.parameters())

                self.log_test(
                    f"Backbone {backbone}",
                    passed,
                    f"Outputs: {len(outs)}, Params: {params:,}"
                )

            except Exception as e:
                self.log_test(f"Backbone {backbone}", False, f"Error: {e}")

    def test_stage_configurations(self):
        """Test different stage configurations."""
        stage_configs = [
            [1], [2], [3], [4],  # Single stages
            [1, 2], [2, 3], [3, 4],  # Pairs
            [1, 2, 3], [2, 3, 4],  # Triples
            [1, 2, 3, 4]  # All stages
        ]

        for stages in stage_configs:
            try:
                encoder = get_gaff_encoder(
                    backbone='mit_b1',
                    gaff_stages=stages
                ).to(self.device)

                x_rgb = torch.randn(2, 3, 224, 224, device=self.device)
                x_aux = torch.randn(2, 1, 224, 224, device=self.device)

                with torch.no_grad():
                    outs = encoder(x_rgb, x_aux)

                passed = len(outs) == 4 and all(not torch.isnan(o).any() for o in outs)
                stage_str = ','.join(map(str, stages))

                self.log_test(
                    f"Stages [{stage_str}]",
                    passed,
                    f"Output shapes: {[tuple(o.shape) for o in outs]}"
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

                x_rgb = torch.randn(2, 3, 224, 224, device=self.device)
                x_aux = torch.randn(2, 1, 224, 224, device=self.device)

                with torch.no_grad():
                    outs = encoder(x_rgb, x_aux)

                passed = len(outs) == 4 and all(not torch.isnan(o).any() for o in outs)
                config_str = ', '.join(f'{k.replace("gaff_", "")}={v}' for k, v in config.items())

                self.log_test(f"Config {i+1}: {config_str}", passed)

            except Exception as e:
                self.log_test(f"Config {i+1}", False, f"Error: {e}")

    def test_output_shapes(self):
        """Test output shapes for different input sizes."""
        test_cases = [
            (1, 224, 224),  # Single sample, standard size
            (2, 224, 224),  # Batch of 2
            (4, 224, 224),  # Batch of 4
            (1, 256, 256),  # Larger input
            (1, 512, 512),  # Even larger
        ]

        encoder = get_gaff_encoder(backbone='mit_b1', gaff_stages=[4]).to(self.device)

        # Expected output dimensions per stage for mit_b1
        expected_dims = [64, 128, 320, 512]

        for batch_size, h, w in test_cases:
            try:
                x_rgb = torch.randn(batch_size, 3, h, w, device=self.device)
                x_aux = torch.randn(batch_size, 1, h, w, device=self.device)

                with torch.no_grad():
                    outs = encoder(x_rgb, x_aux)

                # Verify shapes
                expected_spatial = [h // 4, h // 8, h // 16, h // 32]
                shapes_correct = True

                for i, (out, exp_dim, exp_h, exp_w) in enumerate(zip(
                    outs, expected_dims, expected_spatial, expected_spatial
                )):
                    if out.shape != (batch_size, exp_dim, exp_h, exp_w):
                        shapes_correct = False
                        break

                self.log_test(
                    f"Input (B={batch_size}, H={h}, W={w})",
                    shapes_correct,
                    f"Output shapes: {[tuple(o.shape) for o in outs]}"
                )

            except Exception as e:
                self.log_test(f"Input (B={batch_size}, H={h}, W={w})", False, f"Error: {e}")

    def test_gradient_flow(self):
        """Test gradient flow through encoder."""
        encoder = get_gaff_encoder(backbone='mit_b1', gaff_stages=[2, 4]).to(self.device)

        x_rgb = torch.randn(2, 3, 224, 224, device=self.device, requires_grad=True)
        x_aux = torch.randn(2, 1, 224, 224, device=self.device, requires_grad=True)

        # Forward pass
        outs = encoder(x_rgb, x_aux)

        # Compute loss (sum of all outputs)
        loss = sum(o.sum() for o in outs)

        # Backward pass
        loss.backward()

        # Check gradients
        rgb_has_grad = x_rgb.grad is not None and not torch.isnan(x_rgb.grad).any()
        aux_has_grad = x_aux.grad is not None and not torch.isnan(x_aux.grad).any()

        passed = rgb_has_grad and aux_has_grad

        self.log_test(
            "Gradient flow test",
            passed,
            f"RGB grad: {rgb_has_grad}, Aux grad: {aux_has_grad}"
        )

        # Test gradient through GAFF stages specifically
        encoder.zero_grad()
        x_rgb = torch.randn(2, 3, 224, 224, device=self.device)
        x_aux = torch.randn(2, 1, 224, 224, device=self.device)

        outs = encoder(x_rgb, x_aux)
        loss = sum(o.sum() for o in outs)
        loss.backward()

        # Check GAFF module gradients
        gaff_grads_ok = True
        for i, gaff in enumerate(encoder.GAFFs):
            for param in gaff.parameters():
                if param.requires_grad:
                    if param.grad is None or torch.isnan(param.grad).any():
                        gaff_grads_ok = False
                        break

        self.log_test("GAFF module gradients", gaff_grads_ok)

    def analyze_parameters(self):
        """Analyze parameter counts for all backbones."""
        backbones = ['mit_b0', 'mit_b1', 'mit_b2', 'mit_b3', 'mit_b4', 'mit_b5']

        print(f"{'Backbone':<12s} {'Total Params':<15s} {'GAFF Params':<15s} {'GAFF %':<10s}")
        print("-" * 60)

        for backbone in backbones:
            try:
                encoder = get_gaff_encoder(
                    backbone=backbone,
                    gaff_stages=[1, 2, 3, 4]  # All stages with GAFF
                ).to(self.device)

                total_params = sum(p.numel() for p in encoder.parameters())
                gaff_params = sum(
                    p.numel() for gaff in encoder.GAFFs for p in gaff.parameters()
                )
                gaff_percentage = (gaff_params / total_params * 100) if total_params > 0 else 0

                print(f"{backbone:<12s} {total_params:>13,}  {gaff_params:>13,}  {gaff_percentage:>8.2f}%")

            except Exception as e:
                print(f"{backbone:<12s} Error: {e}")

        print()

    def benchmark_inference(self):
        """Benchmark inference time on CPU."""
        print("Benchmarking mit_b1 with different configurations...")
        print(f"{'Config':<40s} {'Mean (ms)':<12s} {'Std (ms)':<12s}")
        print("-" * 70)

        configs = [
            ('GAFF @ stage 4 only', {'gaff_stages': [4]}),
            ('GAFF @ stages 2,3,4', {'gaff_stages': [2, 3, 4]}),
            ('GAFF @ all stages', {'gaff_stages': [1, 2, 3, 4]}),
        ]

        for name, kwargs in configs:
            try:
                encoder = get_gaff_encoder(backbone='mit_b1', **kwargs).to(self.device)
                encoder.eval()

                x_rgb = torch.randn(1, 3, 224, 224, device=self.device)
                x_aux = torch.randn(1, 1, 224, 224, device=self.device)

                # Warmup
                with torch.no_grad():
                    for _ in range(3):
                        _ = encoder(x_rgb, x_aux)

                # Benchmark
                times = []
                num_iters = 10
                for _ in range(num_iters):
                    start = time.time()
                    with torch.no_grad():
                        _ = encoder(x_rgb, x_aux)
                    elapsed = (time.time() - start) * 1000
                    times.append(elapsed)

                mean_time = sum(times) / len(times)
                std_time = (sum((t - mean_time) ** 2 for t in times) / len(times)) ** 0.5

                print(f"{name:<40s} {mean_time:>10.2f}  {std_time:>10.2f}")

            except Exception as e:
                print(f"{name:<40s} Error: {e}")

        print()

    def print_summary(self):
        """Print verification summary."""
        print("=" * 90)
        print("Verification Summary")
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
            print("✓ All tests passed!")
            print("✓ GAFF encoder is ready for training!")
        else:
            print()
            print("✗ Some tests failed. Please review errors above.")

        print("=" * 90)

        return failed_tests == 0


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="GAFF Encoder Verification")
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                        help='Device to run verification on')
    args = parser.parse_args()

    verifier = GAFFEncoderVerifier(device=args.device)
    all_passed = verifier.verify_all()

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
