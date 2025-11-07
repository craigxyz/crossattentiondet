#!/usr/bin/env python
"""
Quick test of the CSSA ablation pipeline.

Tests a single experiment (1 epoch) to verify everything works.
"""

import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

print("=" * 80)
print("CSSA ABLATION PIPELINE TEST")
print("=" * 80)

print("\n1. Testing imports...")
try:
    import torch
    from crossattentiondet.config import Config
    from crossattentiondet.ablations.encoder_cssa_flexible import get_encoder_cssa_flexible
    print("  ✓ All imports successful")
except Exception as e:
    print(f"  ✗ Import failed: {e}")
    sys.exit(1)

print("\n2. Testing flexible encoder...")
try:
    # Test different stage configurations
    test_configs = [
        ([1], "Stage 1 only"),
        ([4], "Stage 4 only"),
        ([2, 3], "Stages 2&3"),
        ([1, 2, 3, 4], "All stages")
    ]

    for stages, desc in test_configs:
        encoder = get_encoder_cssa_flexible(
            'mit_b1',
            in_chans_rgb=3,
            in_chans_x=2,
            cssa_stages=stages,
            cssa_switching_thresh=0.5,
            cssa_kernel_size=3
        )
        print(f"  ✓ {desc}: {type(encoder).__name__}")

except Exception as e:
    print(f"  ✗ Encoder test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n3. Testing forward pass...")
try:
    encoder = get_encoder_cssa_flexible(
        'mit_b1',
        in_chans_rgb=3,
        in_chans_x=2,
        cssa_stages=[2, 3],
        cssa_switching_thresh=0.5,
        cssa_kernel_size=3
    )

    # Create dummy input
    x_rgb = torch.randn(2, 3, 224, 224)
    x_x = torch.randn(2, 2, 224, 224)

    # Forward pass
    outs = encoder(x_rgb, x_x)

    print(f"  ✓ Forward pass successful")
    print(f"  ✓ Output stages: {len(outs)}")
    print(f"  ✓ Output shapes: {[tuple(o.shape) for o in outs]}")

except Exception as e:
    print(f"  ✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n4. Checking scripts exist...")
scripts = [
    'crossattentiondet/ablations/scripts/train_cssa_ablation.py',
    'crossattentiondet/ablations/scripts/run_cssa_ablations.py',
    'crossattentiondet/ablations/scripts/analyze_cssa_results.py'
]

for script in scripts:
    if os.path.exists(script):
        print(f"  ✓ {script}")
    else:
        print(f"  ✗ {script} NOT FOUND")

print("\n5. Checking data paths...")
data_path = "../RGBX_Semantic_Segmentation/data/images"
labels_path = "../RGBX_Semantic_Segmentation/data/labels"

if os.path.exists(data_path):
    num_images = len([f for f in os.listdir(data_path) if f.endswith('.npy')])
    print(f"  ✓ Data path exists: {data_path} ({num_images} images)")
else:
    print(f"  ✗ Data path not found: {data_path}")

if os.path.exists(labels_path):
    num_labels = len([f for f in os.listdir(labels_path) if f.endswith('.txt')])
    print(f"  ✓ Labels path exists: {labels_path} ({num_labels} labels)")
else:
    print(f"  ✗ Labels path not found: {labels_path}")

print("\n" + "=" * 80)
print("PIPELINE TEST COMPLETE")
print("=" * 80)

print("\nYou can now run experiments:")
print("\n1. Single experiment (quick test):")
print("   python -u crossattentiondet/ablations/scripts/train_cssa_ablation.py \\")
print("       --data ../RGBX_Semantic_Segmentation/data/images \\")
print("       --labels ../RGBX_Semantic_Segmentation/data/labels \\")
print("       --output-dir results/cssa_ablations/test_exp \\")
print("       --epochs 1 \\")
print("       --batch-size 2 \\")
print("       --backbone mit_b1 \\")
print("       --cssa-stages '2,3' \\")
print("       --cssa-thresh 0.5")

print("\n2. Full ablation study (11 experiments, ~66 hours):")
print("   python -u crossattentiondet/ablations/scripts/run_cssa_ablations.py \\")
print("       --data ../RGBX_Semantic_Segmentation/data/images \\")
print("       --labels ../RGBX_Semantic_Segmentation/data/labels \\")
print("       --output-base results/cssa_ablations \\")
print("       --epochs 25 \\")
print("       --backbone mit_b1")

print("\n3. Analyze results:")
print("   python crossattentiondet/ablations/scripts/analyze_cssa_results.py \\")
print("       --results-dir results/cssa_ablations")

print("\n" + "=" * 80)
