"""
Unit tests for GAFF (Guided Attentive Feature Fusion) module.

Tests cover:
1. Shape correctness for various input configurations
2. Gradient flow through all components
3. Parameter count validation
4. Output validity (no NaN/Inf)
5. Configuration validation
6. CPU stress tests

Run with: python -m crossattentiondet.ablations.fusion.test_gaff
"""

import torch
import torch.nn as nn
import sys
from typing import Tuple

try:
    from .gaff import SEBlock, InterModalityAttention, GAFFBlock, build_gaff_block
    from .base import FusionBlock
except ImportError:
    # Fallback for direct execution
    from gaff import SEBlock, InterModalityAttention, GAFFBlock, build_gaff_block
    from base import FusionBlock


class TestGAFF:
    """Test suite for GAFF fusion module."""

    def __init__(self):
        self.device = torch.device('cpu')
        self.test_results = []

    def log_test(self, test_name: str, passed: bool, message: str = ""):
        """Log test result."""
        status = "PASS" if passed else "FAIL"
        self.test_results.append((test_name, passed))
        print(f"[{status}] {test_name}")
        if message:
            print(f"      {message}")

    def run_all_tests(self):
        """Run all tests."""
        print("=" * 70)
        print("GAFF Module Unit Tests")
        print("=" * 70)
        print()

        # SE Block tests
        print("Testing SEBlock...")
        self.test_se_block_shape()
        self.test_se_block_gradients()
        self.test_se_block_parameters()
        self.test_se_block_reductions()
        print()

        # Inter-modality attention tests
        print("Testing InterModalityAttention...")
        self.test_inter_attn_shape()
        self.test_inter_attn_gradients()
        self.test_inter_attn_shared_vs_separate()
        print()

        # GAFF Block tests
        print("Testing GAFFBlock...")
        self.test_gaff_block_shape()
        self.test_gaff_block_gradients()
        self.test_gaff_block_parameters()
        self.test_gaff_block_configurations()
        self.test_gaff_block_output_validity()
        print()

        # Stress tests
        print("Running CPU stress tests...")
        self.test_large_batch_size()
        self.test_various_channel_sizes()
        self.test_various_spatial_sizes()
        print()

        # Integration tests
        print("Running integration tests...")
        self.test_inheritance()
        self.test_factory_function()
        print()

        # Summary
        self.print_summary()

    # ===== SE Block Tests =====

    def test_se_block_shape(self):
        """Test SEBlock output shape."""
        test_cases = [
            (1, 64, 32, 32),
            (2, 128, 16, 16),
            (4, 256, 8, 8),
            (8, 512, 4, 4),
        ]

        for b, c, h, w in test_cases:
            se = SEBlock(c, reduction=4).to(self.device)
            x = torch.randn(b, c, h, w, device=self.device)
            out = se(x)

            passed = out.shape == x.shape
            self.log_test(
                f"SEBlock shape test (B={b}, C={c}, H={h}, W={w})",
                passed,
                f"Expected {x.shape}, got {out.shape}"
            )

    def test_se_block_gradients(self):
        """Test gradient flow through SEBlock."""
        se = SEBlock(64, reduction=4).to(self.device)
        x = torch.randn(2, 64, 16, 16, device=self.device, requires_grad=True)

        out = se(x)
        loss = out.sum()
        loss.backward()

        passed = x.grad is not None and not torch.isnan(x.grad).any()
        self.log_test("SEBlock gradient flow", passed)

    def test_se_block_parameters(self):
        """Test SEBlock parameter count."""
        channels = 128
        reduction = 4
        se = SEBlock(channels, reduction=reduction).to(self.device)

        # Expected params: fc1 (C * C/r) + fc2 (C/r * C)
        reduced = channels // reduction
        expected_params = channels * reduced + reduced * channels

        total_params = sum(p.numel() for p in se.parameters())
        passed = total_params == expected_params

        self.log_test(
            f"SEBlock parameter count (C={channels}, r={reduction})",
            passed,
            f"Expected {expected_params}, got {total_params}"
        )

    def test_se_block_reductions(self):
        """Test SEBlock with different reduction ratios."""
        reductions = [2, 4, 8, 16]
        channels = 256

        for r in reductions:
            se = SEBlock(channels, reduction=r).to(self.device)
            x = torch.randn(2, channels, 8, 8, device=self.device)
            out = se(x)

            passed = out.shape == x.shape and not torch.isnan(out).any()
            self.log_test(f"SEBlock with reduction={r}", passed)

    # ===== Inter-Modality Attention Tests =====

    def test_inter_attn_shape(self):
        """Test InterModalityAttention output shapes."""
        channels = 128
        inter_attn = InterModalityAttention(channels, shared=False).to(self.device)

        x_rgb = torch.randn(2, channels, 16, 16, device=self.device)
        x_aux = torch.randn(2, channels, 16, 16, device=self.device)

        w_rgb, w_aux = inter_attn(x_rgb, x_aux)

        passed = (w_rgb.shape == x_rgb.shape and w_aux.shape == x_aux.shape)
        self.log_test(
            "InterModalityAttention shape test",
            passed,
            f"w_rgb: {w_rgb.shape}, w_aux: {w_aux.shape}"
        )

    def test_inter_attn_gradients(self):
        """Test gradient flow through InterModalityAttention."""
        inter_attn = InterModalityAttention(64, shared=False).to(self.device)

        x_rgb = torch.randn(2, 64, 16, 16, device=self.device, requires_grad=True)
        x_aux = torch.randn(2, 64, 16, 16, device=self.device, requires_grad=True)

        w_rgb, w_aux = inter_attn(x_rgb, x_aux)
        loss = (w_rgb.sum() + w_aux.sum())
        loss.backward()

        passed = (x_rgb.grad is not None and x_aux.grad is not None and
                  not torch.isnan(x_rgb.grad).any() and not torch.isnan(x_aux.grad).any())
        self.log_test("InterModalityAttention gradient flow", passed)

    def test_inter_attn_shared_vs_separate(self):
        """Test InterModalityAttention with shared and separate convs."""
        channels = 128
        x_rgb = torch.randn(2, channels, 16, 16, device=self.device)
        x_aux = torch.randn(2, channels, 16, 16, device=self.device)

        # Test shared
        inter_attn_shared = InterModalityAttention(channels, shared=True).to(self.device)
        w_rgb_s, w_aux_s = inter_attn_shared(x_rgb, x_aux)

        # Test separate
        inter_attn_sep = InterModalityAttention(channels, shared=False).to(self.device)
        w_rgb_ns, w_aux_ns = inter_attn_sep(x_rgb, x_aux)

        passed_shared = (w_rgb_s.shape == x_rgb.shape and not torch.isnan(w_rgb_s).any())
        passed_sep = (w_rgb_ns.shape == x_rgb.shape and not torch.isnan(w_rgb_ns).any())

        self.log_test("InterModalityAttention (shared=True)", passed_shared)
        self.log_test("InterModalityAttention (shared=False)", passed_sep)

    # ===== GAFF Block Tests =====

    def test_gaff_block_shape(self):
        """Test GAFFBlock output shape."""
        test_cases = [
            (64, 64),    # Same in/out channels
            (128, 128),
            (256, 256),
            (64, 128),   # Different in/out channels
            (128, 256),
        ]

        for in_c, out_c in test_cases:
            gaff = GAFFBlock(in_c, out_c).to(self.device)
            x_rgb = torch.randn(2, in_c, 16, 16, device=self.device)
            x_aux = torch.randn(2, in_c, 16, 16, device=self.device)

            out = gaff(x_rgb, x_aux)

            passed = out.shape == (2, out_c, 16, 16)
            self.log_test(
                f"GAFFBlock shape test (in={in_c}, out={out_c})",
                passed,
                f"Expected (2, {out_c}, 16, 16), got {out.shape}"
            )

    def test_gaff_block_gradients(self):
        """Test gradient flow through GAFFBlock."""
        gaff = GAFFBlock(64, 64).to(self.device)

        x_rgb = torch.randn(2, 64, 16, 16, device=self.device, requires_grad=True)
        x_aux = torch.randn(2, 64, 16, 16, device=self.device, requires_grad=True)

        out = gaff(x_rgb, x_aux)
        loss = out.sum()
        loss.backward()

        passed = (x_rgb.grad is not None and x_aux.grad is not None and
                  not torch.isnan(x_rgb.grad).any() and not torch.isnan(x_aux.grad).any())
        self.log_test("GAFFBlock gradient flow", passed)

    def test_gaff_block_parameters(self):
        """Test GAFFBlock parameter count."""
        in_channels = 128
        gaff = GAFFBlock(in_channels, in_channels).to(self.device)

        total_params = sum(p.numel() for p in gaff.parameters())

        # Just verify it's reasonable (not zero, not too large)
        passed = 10000 < total_params < 1000000

        self.log_test(
            f"GAFFBlock parameter count (C={in_channels})",
            passed,
            f"Total params: {total_params:,}"
        )

    def test_gaff_block_configurations(self):
        """Test GAFFBlock with various configurations."""
        configs = [
            {'se_reduction': 4, 'inter_shared': False, 'merge_bottleneck': False},
            {'se_reduction': 8, 'inter_shared': False, 'merge_bottleneck': False},
            {'se_reduction': 4, 'inter_shared': True, 'merge_bottleneck': False},
            {'se_reduction': 4, 'inter_shared': False, 'merge_bottleneck': True},
            {'se_reduction': 8, 'inter_shared': True, 'merge_bottleneck': True},
        ]

        for i, config in enumerate(configs):
            gaff = GAFFBlock(64, 64, **config).to(self.device)
            x_rgb = torch.randn(2, 64, 16, 16, device=self.device)
            x_aux = torch.randn(2, 64, 16, 16, device=self.device)

            out = gaff(x_rgb, x_aux)

            passed = (out.shape == (2, 64, 16, 16) and not torch.isnan(out).any())
            config_str = ', '.join(f'{k}={v}' for k, v in config.items())
            self.log_test(f"GAFFBlock config {i+1}: {config_str}", passed)

    def test_gaff_block_output_validity(self):
        """Test GAFFBlock output for NaN/Inf values."""
        gaff = GAFFBlock(128, 128).to(self.device)

        x_rgb = torch.randn(4, 128, 16, 16, device=self.device)
        x_aux = torch.randn(4, 128, 16, 16, device=self.device)

        out = gaff(x_rgb, x_aux)

        has_nan = torch.isnan(out).any().item()
        has_inf = torch.isinf(out).any().item()

        passed = not has_nan and not has_inf
        self.log_test(
            "GAFFBlock output validity (no NaN/Inf)",
            passed,
            f"NaN: {has_nan}, Inf: {has_inf}"
        )

    # ===== Stress Tests =====

    def test_large_batch_size(self):
        """Test with large batch size on CPU."""
        batch_sizes = [1, 8, 16, 32]

        for bs in batch_sizes:
            gaff = GAFFBlock(64, 64).to(self.device)
            x_rgb = torch.randn(bs, 64, 16, 16, device=self.device)
            x_aux = torch.randn(bs, 64, 16, 16, device=self.device)

            try:
                out = gaff(x_rgb, x_aux)
                passed = out.shape == (bs, 64, 16, 16)
            except Exception as e:
                passed = False
                print(f"      Error: {e}")

            self.log_test(f"Large batch size test (B={bs})", passed)

    def test_various_channel_sizes(self):
        """Test with various channel dimensions."""
        channel_sizes = [32, 64, 128, 256, 512]

        for c in channel_sizes:
            gaff = GAFFBlock(c, c).to(self.device)
            x_rgb = torch.randn(2, c, 8, 8, device=self.device)
            x_aux = torch.randn(2, c, 8, 8, device=self.device)

            try:
                out = gaff(x_rgb, x_aux)
                passed = out.shape == (2, c, 8, 8)
            except Exception as e:
                passed = False
                print(f"      Error: {e}")

            self.log_test(f"Channel size test (C={c})", passed)

    def test_various_spatial_sizes(self):
        """Test with various spatial dimensions."""
        spatial_sizes = [(4, 4), (8, 8), (16, 16), (32, 32), (64, 64)]

        for h, w in spatial_sizes:
            gaff = GAFFBlock(64, 64).to(self.device)
            x_rgb = torch.randn(2, 64, h, w, device=self.device)
            x_aux = torch.randn(2, 64, h, w, device=self.device)

            try:
                out = gaff(x_rgb, x_aux)
                passed = out.shape == (2, 64, h, w)
            except Exception as e:
                passed = False
                print(f"      Error: {e}")

            self.log_test(f"Spatial size test (H={h}, W={w})", passed)

    # ===== Integration Tests =====

    def test_inheritance(self):
        """Test that GAFFBlock properly inherits from FusionBlock."""
        gaff = GAFFBlock(64, 64)
        passed = isinstance(gaff, FusionBlock)
        self.log_test("GAFFBlock inherits from FusionBlock", passed)

    def test_factory_function(self):
        """Test build_gaff_block factory function."""
        gaff = build_gaff_block(
            in_channels=128,
            out_channels=128,
            se_reduction=8,
            inter_shared=True,
            merge_bottleneck=True
        ).to(self.device)

        x_rgb = torch.randn(2, 128, 16, 16, device=self.device)
        x_aux = torch.randn(2, 128, 16, 16, device=self.device)

        try:
            out = gaff(x_rgb, x_aux)
            passed = out.shape == (2, 128, 16, 16)
        except Exception as e:
            passed = False
            print(f"      Error: {e}")

        self.log_test("build_gaff_block factory function", passed)

    # ===== Summary =====

    def print_summary(self):
        """Print test summary."""
        print("=" * 70)
        print("Test Summary")
        print("=" * 70)

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
        print("=" * 70)

        return failed_tests == 0


def main():
    """Run all tests."""
    tester = TestGAFF()
    all_passed = tester.run_all_tests()

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
