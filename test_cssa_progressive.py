#!/usr/bin/env python
"""
Progressive testing script for CSSA training.
Tests each component step-by-step to catch errors early.
"""

import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

print("=" * 60)
print("STEP 1: Testing imports")
print("=" * 60)

try:
    import torch
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✓ CUDA device: {torch.cuda.get_device_name(0)}")

    from torchvision.ops import MultiScaleRoIAlign
    from torchvision.models.detection.anchor_utils import AnchorGenerator
    from torchvision.models.detection import FasterRCNN
    print("✓ Torchvision imports successful")

    from crossattentiondet.config import Config, get_num_classes
    print("✓ Config import successful")

    from crossattentiondet.data.dataset import NpyYoloDataset
    print("✓ Dataset import successful")

    from crossattentiondet.models.backbone import CrossAttentionBackbone
    print("✓ Backbone import successful")

    from crossattentiondet.ablations.encoder_cssa import get_encoder_cssa
    print("✓ CSSA encoder import successful")

    print("\n✓ ALL IMPORTS SUCCESSFUL\n")

except Exception as e:
    print(f"\n✗ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("=" * 60)
print("STEP 2: Testing config initialization")
print("=" * 60)

try:
    config = Config()
    config.data_dir = "/mmfs1/project/pva23/csi3/RGBX_Semantic_Segmentation/data/images"
    config.labels_dir = "/mmfs1/project/pva23/csi3/RGBX_Semantic_Segmentation/data/labels"
    config.model_path = "checkpoints/test_cssa.pth"
    config.backbone_type = "mit_b1"
    config.epochs = 1
    config.batch_size = 2
    config.lr = 0.005

    print(f"✓ Config created")
    print(f"  Data dir: {config.data_dir}")
    print(f"  Labels dir: {config.labels_dir}")
    print(f"  Backbone: {config.backbone_type}")
    print(f"  Batch size: {config.batch_size}")

    print("\n✓ CONFIG INITIALIZATION SUCCESSFUL\n")

except Exception as e:
    print(f"\n✗ Config initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("=" * 60)
print("STEP 3: Testing dataset loading")
print("=" * 60)

try:
    # Check if directories exist
    if not os.path.exists(config.data_dir):
        raise FileNotFoundError(f"Data directory not found: {config.data_dir}")
    if not os.path.exists(config.labels_dir):
        raise FileNotFoundError(f"Labels directory not found: {config.labels_dir}")

    print(f"✓ Data directory exists: {config.data_dir}")
    print(f"✓ Labels directory exists: {config.labels_dir}")

    # Try loading dataset
    train_dataset = NpyYoloDataset(
        image_dir=config.data_dir,
        label_dir=config.labels_dir,
        mode='train'
    )

    print(f"✓ Train dataset loaded: {len(train_dataset)} images")

    # Get num_classes
    if config.num_classes is None:
        config.num_classes = get_num_classes(config.labels_dir) + 1
        print(f"✓ Detected {config.num_classes} classes (including background)")

    # Try loading one sample
    print("\n  Testing single sample load...")
    img, target = train_dataset[0]
    print(f"  ✓ Image shape: {img.shape}")
    print(f"  ✓ Target keys: {target.keys()}")
    print(f"  ✓ Boxes shape: {target['boxes'].shape}")
    print(f"  ✓ Labels shape: {target['labels'].shape}")

    print("\n✓ DATASET LOADING SUCCESSFUL\n")

except Exception as e:
    print(f"\n✗ Dataset loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("=" * 60)
print("STEP 4: Testing CSSA encoder instantiation")
print("=" * 60)

try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    encoder_base = get_encoder_cssa(
        config.backbone_type,
        in_chans_rgb=config.in_chans_rgb,
        in_chans_x=config.in_chans_x,
        cssa_switching_thresh=0.5,
        cssa_kernel_size=3
    )

    print(f"✓ CSSA encoder created: {type(encoder_base).__name__}")
    print(f"  Input channels RGB: {config.in_chans_rgb}")
    print(f"  Input channels X: {config.in_chans_x}")

    print("\n✓ CSSA ENCODER INSTANTIATION SUCCESSFUL\n")

except Exception as e:
    print(f"\n✗ CSSA encoder instantiation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("=" * 60)
print("STEP 5: Testing model building")
print("=" * 60)

try:
    # Wrap with FPN
    backbone = CrossAttentionBackbone(encoder_base, fpn_out_channels=config.fpn_out_channels)
    print(f"✓ Backbone with FPN created")
    print(f"  FPN output channels: {backbone.out_channels}")

    # Anchor generator
    anchor_generator = AnchorGenerator(
        sizes=config.anchor_sizes,
        aspect_ratios=config.anchor_aspect_ratios
    )
    print(f"✓ Anchor generator created")

    # RoI pooler
    roi_pooler = MultiScaleRoIAlign(
        featmap_names=config.roi_featmap_names,
        output_size=config.roi_output_size,
        sampling_ratio=config.roi_sampling_ratio
    )
    print(f"✓ RoI pooler created")

    # Build Faster R-CNN
    model = FasterRCNN(
        backbone,
        num_classes=config.num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        image_mean=config.image_mean,
        image_std=config.image_std
    )

    print(f"✓ Faster R-CNN model created")
    print(f"  Num classes: {config.num_classes}")

    # Move to device
    model.to(device)
    print(f"✓ Model moved to {device}")

    print("\n✓ MODEL BUILDING SUCCESSFUL\n")

except Exception as e:
    print(f"\n✗ Model building failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("=" * 60)
print("STEP 6: Testing single forward pass")
print("=" * 60)

try:
    model.train()

    # Get a small batch
    from torch.utils.data import DataLoader

    def collate_fn(batch):
        return tuple(zip(*batch))

    test_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    images, targets = next(iter(test_loader))
    print(f"✓ Batch loaded: {len(images)} images")

    # Move to device
    images = [img.to(device) for img in images]
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    print(f"✓ Batch moved to {device}")

    # Forward pass
    print("\n  Running forward pass...")
    loss_dict = model(images, targets)
    print(f"✓ Forward pass successful")
    print(f"  Loss dict keys: {loss_dict.keys()}")

    losses = sum(loss for loss in loss_dict.values())
    print(f"✓ Total loss: {losses.item():.4f}")

    # Test backward pass
    print("\n  Testing backward pass...")
    losses.backward()
    print(f"✓ Backward pass successful")

    print("\n✓ FORWARD/BACKWARD PASS SUCCESSFUL\n")

except Exception as e:
    print(f"\n✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("=" * 60)
print("ALL TESTS PASSED!")
print("=" * 60)
print("\nThe model is ready for training. You can now run:")
print("python crossattentiondet/ablations/scripts/train_cssa.py \\")
print("    --data ../RGBX_Semantic_Segmentation/data/images \\")
print("    --labels ../RGBX_Semantic_Segmentation/data/labels \\")
print("    --epochs 1 \\")
print("    --batch-size 2 \\")
print("    --backbone mit_b1")
print("=" * 60)
