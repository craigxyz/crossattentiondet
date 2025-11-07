# Comprehensive Training Guide

## Overview

This guide explains how to use the comprehensive training script (`train_all_backbones_comprehensive.py`) to train all backbone variants with detailed logging and checkpoint management.

## Features

### 1. **Per-Epoch Logging**
- Average loss per epoch
- Total loss per epoch
- Epoch training time
- Learning rate tracking
- Best model tracking

### 2. **Per-Batch Logging**
- Individual batch losses
- Component losses (classifier, box regression, RPN losses)
- Batch processing time
- Timestamps for all metrics

### 3. **Checkpoint Management**
- Saves checkpoint after every epoch
- Tracks best model (lowest loss)
- Saves final model weights
- Complete checkpoints with optimizer state for resuming

### 4. **Comprehensive Logs**
- CSV files for easy analysis (epoch_metrics.csv, batch_metrics.csv)
- JSON files for configuration and metadata
- Training summaries for each backbone
- Final summary across all backbones

## Quick Start

### Basic Usage

Train all backbones for 30 epochs:
```bash
python scripts/train_all_backbones_comprehensive.py --data data/ --epochs 30
```

### Custom Configuration

Train specific backbones with custom settings:
```bash
python scripts/train_all_backbones_comprehensive.py \
    --data data/ \
    --epochs 50 \
    --batch-size 4 \
    --lr 0.005 \
    --backbones mit_b0 mit_b1 mit_b2 \
    --num-workers 8
```

### With Learning Rate Scheduling

```bash
python scripts/train_all_backbones_comprehensive.py \
    --data data/ \
    --epochs 50 \
    --batch-size 4 \
    --lr 0.01 \
    --lr-step-size 15 \
    --lr-gamma 0.1 \
    --num-workers 8
```

## Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data` | str | **required** | Path to data directory containing images/ and labels/ |
| `--epochs` | int | 30 | Number of training epochs for each backbone |
| `--batch-size` | int | 2 | Batch size for training |
| `--lr` | float | 0.005 | Initial learning rate |
| `--lr-step-size` | int | 10 | Learning rate decay step (epochs) |
| `--lr-gamma` | float | 0.1 | Learning rate decay factor |
| `--num-workers` | int | 4 | Number of data loading workers |
| `--backbones` | list | all | Backbones to train: mit_b0, mit_b1, mit_b2, mit_b4, mit_b5 |
| `--log-dir` | str | training_logs | Directory to save logs and checkpoints |

## Output Structure

The script creates a comprehensive logging structure:

```
training_logs/
└── run_20251107_120000/           # Timestamped run directory
    ├── run_config.json             # Run configuration
    ├── final_summary.json          # Final summary across all backbones
    ├── mit_b0/                     # Per-backbone directory
    │   ├── checkpoints/
    │   │   ├── epoch_001_loss_2.3456.pth
    │   │   ├── epoch_002_loss_2.1234.pth
    │   │   ├── ...
    │   │   ├── best_model.pth      # Best checkpoint (lowest loss)
    │   │   ├── mit_b0_final.pth    # Final model weights only
    │   │   └── mit_b0_final_complete.pth  # Final complete checkpoint
    │   └── logs/
    │       ├── epoch_metrics.csv   # Per-epoch metrics
    │       ├── batch_metrics.csv   # Per-batch metrics
    │       └── training_info.json  # Training metadata
    ├── mit_b1/
    │   ├── checkpoints/
    │   └── logs/
    ├── mit_b2/
    │   ├── checkpoints/
    │   └── logs/
    └── ...
```

## Log Files Explained

### 1. epoch_metrics.csv

Columns:
- `epoch`: Epoch number
- `avg_loss`: Average loss for the epoch
- `total_loss`: Total accumulated loss
- `epoch_time_sec`: Time taken for epoch (seconds)
- `learning_rate`: Current learning rate
- `is_best`: Whether this is the best model so far
- `checkpoint_path`: Path to saved checkpoint
- `timestamp`: ISO format timestamp

### 2. batch_metrics.csv

Columns:
- `epoch`: Epoch number
- `batch`: Batch number within epoch
- `loss`: Total batch loss
- `loss_classifier`: Classification loss component
- `loss_box_reg`: Bounding box regression loss
- `loss_objectness`: RPN objectness loss
- `loss_rpn_box_reg`: RPN box regression loss
- `timestamp`: ISO format timestamp

### 3. training_info.json

Contains:
- Backbone name and status
- Complete configuration
- Model parameters count
- Best epoch and loss
- All epoch losses and times
- Final model paths

### 4. final_summary.json

Contains:
- Run configuration
- Total training time
- Results for all backbones
- Success/failure statistics
- Timestamp

## Checkpoint Types

### 1. Epoch Checkpoints
**File:** `epoch_XXX_loss_Y.YYYY.pth`
- Saved after every epoch
- Contains model, optimizer, and scheduler state
- Can be used to resume training

### 2. Best Model
**File:** `best_model.pth`
- Automatically updated when a new best (lowest loss) is achieved
- Complete checkpoint with all states

### 3. Final Model Weights
**File:** `{backbone}_final.pth`
- Contains only model state dict
- Lightweight, ideal for inference
- Saved at end of training

### 4. Final Complete Checkpoint
**File:** `{backbone}_final_complete.pth`
- Complete checkpoint with model, optimizer, scheduler
- All training history included
- Can be used for analysis or resuming

## Backbone Comparison

| Backbone | Parameters | Typical Training Time (30 epochs) |
|----------|-----------|-----------------------------------|
| mit_b0 | 27.8M | ~2-3 hours |
| mit_b1 | 60.0M | ~3-4 hours |
| mit_b2 | 82.1M | ~4-6 hours |
| mit_b4 | 155.4M | ~8-12 hours |
| mit_b5 | 196.6M | ~10-15 hours |

*Times are approximate and depend on hardware, batch size, and dataset size*

## Recommended Training Schedules

### Quick Testing (Small Dataset)
```bash
python scripts/train_all_backbones_comprehensive.py \
    --data data/ \
    --epochs 10 \
    --batch-size 4 \
    --backbones mit_b0 mit_b1
```

### Standard Training
```bash
python scripts/train_all_backbones_comprehensive.py \
    --data data/ \
    --epochs 30 \
    --batch-size 2 \
    --lr 0.005
```

### Extended Training (Best Performance)
```bash
python scripts/train_all_backbones_comprehensive.py \
    --data data/ \
    --epochs 50 \
    --batch-size 4 \
    --lr 0.01 \
    --lr-step-size 15 \
    --lr-gamma 0.1 \
    --num-workers 8
```

### Large Models Only
```bash
python scripts/train_all_backbones_comprehensive.py \
    --data data/ \
    --epochs 30 \
    --backbones mit_b4 mit_b5 \
    --batch-size 2
```

## Analyzing Results

### View Training Progress
```bash
# View epoch metrics
cat training_logs/run_*/mit_b1/logs/epoch_metrics.csv

# View training info
cat training_logs/run_*/mit_b1/logs/training_info.json | jq
```

### Compare Backbones
```bash
# View final summary
cat training_logs/run_*/final_summary.json | jq
```

### Load Best Model for Inference
```python
import torch
from crossattentiondet.config import Config
from crossattentiondet.models.encoder import get_encoder
from crossattentiondet.models.backbone import CrossAttentionBackbone
from torchvision.models.detection.faster_rcnn import FasterRCNN

# Load checkpoint
checkpoint = torch.load('training_logs/run_*/mit_b1/checkpoints/best_model.pth')

# Build model
config = Config()
encoder = get_encoder('mit_b1', in_chans_rgb=3, in_chans_x=2)
backbone = CrossAttentionBackbone(encoder, fpn_out_channels=256)
model = FasterRCNN(backbone, num_classes=config.num_classes)

# Load weights
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

## Tips for Long Training Sessions

### 1. Use tmux or screen
```bash
# Start tmux session
tmux new -s training

# Run training
python scripts/train_all_backbones_comprehensive.py --data data/ --epochs 50

# Detach: Ctrl+B then D
# Reattach: tmux attach -t training
```

### 2. Monitor Progress
```bash
# Watch epoch metrics in real-time
watch -n 5 "tail -20 training_logs/run_*/mit_b1/logs/epoch_metrics.csv"

# Monitor GPU usage
watch -n 1 nvidia-smi
```

### 3. Redirect Output
```bash
python scripts/train_all_backbones_comprehensive.py \
    --data data/ --epochs 50 2>&1 | tee training_output.log
```

## Troubleshooting

### Out of Memory
- Reduce batch size: `--batch-size 1`
- Train smaller models first: `--backbones mit_b0 mit_b1`
- Reduce number of workers: `--num-workers 0`

### Slow Training
- Increase batch size if memory allows: `--batch-size 4`
- Increase workers: `--num-workers 8`
- Use GPU if available (automatic)

### Training Divergence
- Lower learning rate: `--lr 0.001`
- Adjust scheduler: `--lr-step-size 5 --lr-gamma 0.5`
- Check data quality

## Next Steps

After training completes:

1. **Evaluate models** - Use `scripts/test_all_backbones.py`
2. **Compare performance** - Analyze the CSV logs
3. **Select best backbone** - Based on mAP and inference time
4. **Fine-tune** - Resume training from best checkpoint with lower learning rate

## Questions?

See also:
- `README.md` - Project overview
- `QUICK_START.md` - Quick start guide
- `scripts/train_all_backbones.py` - Simpler training script
