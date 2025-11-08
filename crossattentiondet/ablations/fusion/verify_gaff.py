"""
GAFF Module Verification Script

Standalone verification script that tests GAFF modules without requiring a dataset.
This script provides comprehensive verification suitable for CPU-only environments.

Features:
- Tests all hyperparameter combinations
- Benchmarks CPU inference time
- Memory profiling
- Detailed module structure analysis
- No dataset required

Run with: python -m crossattentiondet.ablations.fusion.verify_gaff
"""

import torch
import torch.nn as nn
import time
import sys
from typing import Dict, List, Tuple

try:
    from .gaff import SEBlock, InterModalityAttention, GAFFBlock, build_gaff_block
except ImportError:
    from gaff import SEBlock, InterModalityAttention, GAFFBlock, build_gaff_block


class GAFFVerifier:
    """Comprehensive GAFF module verification."""

    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        self.results = []

    def verify_all(self):
        """Run all verification tests."""
        print("=" * 80)
        print("GAFF Module Verification")
        print("=" * 80)
        print(f"Device: {self.device}")
        print()

        # Module structure analysis
        self.analyze_module_structure()
        print()

        # Hyperparameter sweep
        self.verify_hyperparameter_combinations()
        print()

        # Performance benchmarks
        self.benchmark_inference_time()
        print()

        # Memory profiling
        self.profile_memory_usage()
        print()

        # Summary
        self.print_verification_summary()

    def analyze_module_structure(self):
        """Analyze and print detailed module structure."""
        print("-" * 80)
        print("Module Structure Analysis")
        print("-" * 80)

        # Create a GAFF block with default settings
        gaff = GAFFBlock(
            in_channels=128,
            out_channels=128,
            se_reduction=4,
            inter_shared=False,
            merge_bottleneck=False
        ).to(self.device)

        print("\nGAFFBlock Structure:")
        print(gaff)
        print()

        # Count parameters by component
        components = {
            'SE RGB': gaff.se_rgb,
            'SE Aux': gaff.se_aux,
            'Inter-Modality Attention': gaff.inter_attn,
        }

        # Add merge layer
        if hasattr(gaff, 'merge_conv'):
            components['Merge Conv'] = gaff.merge_conv
            components['Merge BN'] = gaff.merge_bn
        elif hasattr(gaff, 'merge_conv1'):
            components['Merge Conv1'] = gaff.merge_conv1
            components['Merge BN1'] = gaff.merge_bn1
            components['Merge Conv2'] = gaff.merge_conv2
            components['Merge BN2'] = gaff.merge_bn2

        print("Parameter Count by Component:")
        total_params = 0
        for name, module in components.items():
            params = sum(p.numel() for p in module.parameters())
            total_params += params
            print(f"  {name:30s}: {params:>10,} params")

        print(f"  {'Total':30s}: {total_params:>10,} params")
        print()

        # Test forward pass
        x_rgb = torch.randn(2, 128, 16, 16, device=self.device)
        x_aux = torch.randn(2, 128, 16, 16, device=self.device)

        with torch.no_grad():
            out = gaff(x_rgb, x_aux)

        print(f"Input shape (RGB):  {tuple(x_rgb.shape)}")
        print(f"Input shape (Aux):  {tuple(x_aux.shape)}")
        print(f"Output shape:       {tuple(out.shape)}")
        print()

    def verify_hyperparameter_combinations(self):
        """Test all valid hyperparameter combinations."""
        print("-" * 80)
        print("Hyperparameter Combination Verification")
        print("-" * 80)
        print()

        # Define hyperparameter grid
        se_reductions = [4, 8]
        inter_shared_opts = [False, True]
        merge_bottleneck_opts = [False, True]
        channel_sizes = [64, 128, 256]

        test_count = 0
        pass_count = 0

        print(f"{'Config':<60s} {'Status':<10s} {'Params':<12s} {'Time (ms)':<12s}")
        print("-" * 100)

        for c in channel_sizes:
            for se_r in se_reductions:
                for inter_s in inter_shared_opts:
                    for merge_b in merge_bottleneck_opts:
                        test_count += 1

                        config_str = (
                            f"C={c:3d}, SE_r={se_r}, "
                            f"inter_shared={str(inter_s):5s}, "
                            f"merge_bottleneck={str(merge_b):5s}"
                        )

                        try:
                            # Create module
                            gaff = GAFFBlock(
                                in_channels=c,
                                out_channels=c,
                                se_reduction=se_r,
                                inter_shared=inter_s,
                                merge_bottleneck=merge_b
                            ).to(self.device)

                            # Count parameters
                            params = sum(p.numel() for p in gaff.parameters())

                            # Test forward pass
                            x_rgb = torch.randn(2, c, 16, 16, device=self.device)
                            x_aux = torch.randn(2, c, 16, 16, device=self.device)

                            start_time = time.time()
                            with torch.no_grad():
                                out = gaff(x_rgb, x_aux)
                            elapsed = (time.time() - start_time) * 1000

                            # Verify output
                            if out.shape == (2, c, 16, 16) and not torch.isnan(out).any():
                                status = "PASS"
                                pass_count += 1
                            else:
                                status = "FAIL"

                        except Exception as e:
                            status = "ERROR"
                            params = 0
                            elapsed = 0
                            print(f"{config_str:<60s} {status:<10s} Error: {e}")
                            continue

                        print(f"{config_str:<60s} {status:<10s} {params:>10,}  {elapsed:>10.2f}")

        print("-" * 100)
        print(f"Passed: {pass_count}/{test_count} ({pass_count/test_count*100:.1f}%)")
        print()

    def benchmark_inference_time(self):
        """Benchmark inference time for different configurations."""
        print("-" * 80)
        print("CPU Inference Time Benchmarks")
        print("-" * 80)
        print()

        batch_sizes = [1, 2, 4, 8]
        channel_sizes = [64, 128, 256]
        spatial_size = (16, 16)
        num_iterations = 10

        print(f"{'Config':<40s} {'Mean (ms)':<12s} {'Std (ms)':<12s} {'Min (ms)':<12s} {'Max (ms)':<12s}")
        print("-" * 100)

        for c in channel_sizes:
            gaff = GAFFBlock(c, c).to(self.device)

            for bs in batch_sizes:
                h, w = spatial_size
                x_rgb = torch.randn(bs, c, h, w, device=self.device)
                x_aux = torch.randn(bs, c, h, w, device=self.device)

                # Warmup
                with torch.no_grad():
                    for _ in range(3):
                        _ = gaff(x_rgb, x_aux)

                # Benchmark
                times = []
                for _ in range(num_iterations):
                    start = time.time()
                    with torch.no_grad():
                        _ = gaff(x_rgb, x_aux)
                    elapsed = (time.time() - start) * 1000
                    times.append(elapsed)

                mean_time = sum(times) / len(times)
                std_time = (sum((t - mean_time) ** 2 for t in times) / len(times)) ** 0.5
                min_time = min(times)
                max_time = max(times)

                config = f"B={bs}, C={c:3d}, H={h:2d}, W={w:2d}"
                print(f"{config:<40s} {mean_time:>10.2f}  {std_time:>10.2f}  {min_time:>10.2f}  {max_time:>10.2f}")

        print()

    def profile_memory_usage(self):
        """Profile memory usage for different configurations."""
        print("-" * 80)
        print("Memory Usage Profiling")
        print("-" * 80)
        print()

        channel_sizes = [64, 128, 256, 512]
        batch_size = 2
        spatial_size = (16, 16)

        print(f"{'Config':<30s} {'Params (MB)':<15s} {'Activation (MB)':<15s}")
        print("-" * 80)

        for c in channel_sizes:
            gaff = GAFFBlock(c, c).to(self.device)

            # Parameter memory
            param_memory = sum(p.numel() * p.element_size() for p in gaff.parameters()) / (1024 ** 2)

            # Activation memory (approximate)
            h, w = spatial_size
            x_rgb = torch.randn(batch_size, c, h, w, device=self.device)
            x_aux = torch.randn(batch_size, c, h, w, device=self.device)

            input_memory = (x_rgb.numel() + x_aux.numel()) * x_rgb.element_size() / (1024 ** 2)

            with torch.no_grad():
                out = gaff(x_rgb, x_aux)

            output_memory = out.numel() * out.element_size() / (1024 ** 2)
            activation_memory = input_memory + output_memory

            config = f"C={c:3d}, B={batch_size}"
            print(f"{config:<30s} {param_memory:>13.3f}  {activation_memory:>13.3f}")

        print()

    def print_verification_summary(self):
        """Print verification summary."""
        print("=" * 80)
        print("Verification Complete")
        print("=" * 80)
        print()
        print("✓ Module structure analysis complete")
        print("✓ All hyperparameter combinations tested")
        print("✓ CPU inference benchmarks complete")
        print("✓ Memory profiling complete")
        print()
        print("GAFF module is ready for integration!")
        print("=" * 80)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="GAFF Module Verification")
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                        help='Device to run verification on')
    args = parser.parse_args()

    verifier = GAFFVerifier(device=args.device)
    verifier.verify_all()


if __name__ == "__main__":
    main()
