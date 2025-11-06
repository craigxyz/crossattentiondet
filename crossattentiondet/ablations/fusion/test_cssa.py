"""
Unit tests for CSSA fusion module.
"""

import torch
from cssa import CSSABlock


def test_cssa_shapes():
    """Test that CSSA module produces correct output shapes."""
    print("Testing CSSA module with various input shapes...")

    # Test configurations
    test_configs = [
        (2, 64, 32, 32),   # Small feature map
        (2, 128, 64, 64),  # Medium feature map
        (2, 256, 64, 64),  # Typical FPN size
        (2, 320, 40, 40),  # SegFormer stage 3 size
        (2, 512, 20, 20),  # SegFormer stage 4 size
    ]

    for B, C, H, W in test_configs:
        print(f"\nTest case: B={B}, C={C}, H={H}, W={W}")

        # Create CSSA block
        cssa = CSSABlock(switching_thresh=0.5, kernel_size=3)
        cssa.eval()

        # Create dummy inputs
        x_rgb = torch.randn(B, C, H, W)
        x_aux = torch.randn(B, C, H, W)

        # Forward pass
        with torch.no_grad():
            fused = cssa(x_rgb, x_aux)

        # Check output shape
        expected_shape = (B, C, H, W)
        assert fused.shape == expected_shape, \
            f"Expected shape {expected_shape}, got {fused.shape}"

        # Check output is not NaN or Inf
        assert not torch.isnan(fused).any(), "Output contains NaN values"
        assert not torch.isinf(fused).any(), "Output contains Inf values"

        print(f"  ✓ Output shape: {fused.shape}")
        print(f"  ✓ Output range: [{fused.min().item():.4f}, {fused.max().item():.4f}]")
        print(f"  ✓ Output mean: {fused.mean().item():.4f}, std: {fused.std().item():.4f}")

    print("\n✅ All shape tests passed!")


def test_cssa_gradients():
    """Test that CSSA module allows gradient flow."""
    print("\nTesting gradient flow through CSSA module...")

    B, C, H, W = 2, 256, 64, 64

    # Create CSSA block
    cssa = CSSABlock(switching_thresh=0.5, kernel_size=3)
    cssa.train()

    # Create dummy inputs with gradient tracking
    x_rgb = torch.randn(B, C, H, W, requires_grad=True)
    x_aux = torch.randn(B, C, H, W, requires_grad=True)

    # Forward pass
    fused = cssa(x_rgb, x_aux)

    # Compute dummy loss and backward
    loss = fused.sum()
    loss.backward()

    # Check gradients exist
    assert x_rgb.grad is not None, "No gradient for RGB input"
    assert x_aux.grad is not None, "No gradient for auxiliary input"

    # Check gradients are not all zero
    assert x_rgb.grad.abs().sum() > 0, "RGB gradients are all zero"
    assert x_aux.grad.abs().sum() > 0, "Auxiliary gradients are all zero"

    print(f"  ✓ RGB gradient norm: {x_rgb.grad.norm().item():.4f}")
    print(f"  ✓ Aux gradient norm: {x_aux.grad.norm().item():.4f}")
    print("✅ Gradient flow test passed!")


def test_cssa_parameters():
    """Test CSSA module parameter count."""
    print("\nTesting CSSA module parameters...")

    cssa = CSSABlock(switching_thresh=0.5, kernel_size=3)

    total_params = sum(p.numel() for p in cssa.parameters())
    trainable_params = sum(p.numel() for p in cssa.parameters() if p.requires_grad)

    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Parameter size: {total_params * 4 / 1024:.2f} KB (fp32)")

    # CSSA should be lightweight (< 10K parameters for typical setup)
    assert trainable_params < 10000, \
        f"CSSA has too many parameters: {trainable_params:,}"

    print("✅ Parameter count test passed!")


if __name__ == "__main__":
    print("=" * 60)
    print("CSSA Module Unit Tests")
    print("=" * 60)

    test_cssa_shapes()
    test_cssa_gradients()
    test_cssa_parameters()

    print("\n" + "=" * 60)
    print("All tests passed! ✅")
    print("=" * 60)
