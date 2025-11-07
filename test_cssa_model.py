#!/usr/bin/env python
"""
Quick test to verify CSSA model builds correctly with 5-channel input.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection import FasterRCNN

from crossattentiondet.config import Config
from crossattentiondet.models.backbone import CrossAttentionBackbone
from crossattentiondet.ablations.encoder_cssa import get_encoder_cssa

print("=" * 60)
print("Testing CSSA Model Build with 5-Channel Input")
print("=" * 60)

# Create config
config = Config()
config.num_classes = 2  # Including background

print("\n1. Building CSSA encoder...")
encoder_base = get_encoder_cssa(
    'mit_b1',
    in_chans_rgb=3,
    in_chans_x=2,
    cssa_switching_thresh=0.5,
    cssa_kernel_size=3
)
print("✓ Encoder created")

print("\n2. Wrapping with FPN...")
backbone = CrossAttentionBackbone(encoder_base, fpn_out_channels=256)
print(f"✓ Backbone created. Output channels: {backbone.out_channels}")

print("\n3. Creating Faster R-CNN...")
anchor_generator = AnchorGenerator(
    sizes=config.anchor_sizes,
    aspect_ratios=config.anchor_aspect_ratios
)

roi_pooler = MultiScaleRoIAlign(
    featmap_names=config.roi_featmap_names,
    output_size=config.roi_output_size,
    sampling_ratio=config.roi_sampling_ratio
)

print(f"   Config image_mean: {config.image_mean}")
print(f"   Config image_std: {config.image_std}")

model = FasterRCNN(
    backbone,
    num_classes=config.num_classes,
    rpn_anchor_generator=anchor_generator,
    box_roi_pool=roi_pooler,
    image_mean=config.image_mean,
    image_std=config.image_std
)
print("✓ Faster R-CNN model created")

print("\n4. Testing forward pass with dummy 5-channel input...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"   Using device: {device}")

model.to(device)
model.eval()

# Create dummy 5-channel image - FasterRCNN expects list of (C, H, W) tensors
dummy_image = torch.randn(5, 800, 800).to(device)
print(f"   Input shape: {dummy_image.shape}")

try:
    with torch.no_grad():
        # Inference mode - pass as list
        output = model([dummy_image])
    print("✓ Forward pass successful!")
    print(f"   Output boxes: {output[0]['boxes'].shape}")
    print(f"   Output scores: {output[0]['scores'].shape}")
    print(f"   Output labels: {output[0]['labels'].shape}")
except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("SUCCESS! CSSA model works with 5-channel input")
print("=" * 60)
