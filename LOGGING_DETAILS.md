# Comprehensive Logging System for Modality Ablations

## Overview

The modality ablation training script includes extensive logging to capture all aspects of training and evaluation. This document details what is logged and where to find it.

## Output Files Per Experiment

Each experiment (RGB+Thermal, RGB+Event, Thermal+Event) creates 13 output files:

### 1. **training.log**
- **Format**: Text log with timestamps
- **Content**:
  - Detailed console output with timestamps
  - Experiment configuration
  - Model architecture info (parameters)
  - Per-batch progress (every 10 batches)
  - Per-epoch summaries
  - Evaluation results at epochs 5, 10, 15
  - Final results
- **Use**: Human-readable training progress, debugging

### 2. **metrics_per_epoch.csv**
- **Format**: CSV with headers
- **Columns**:
  - `epoch`: Epoch number (1-15)
  - `train_loss`: Average training loss for the epoch
  - `test_mAP`: Test set mAP (logged at epochs 5, 10, 15)
  - `test_mAP_50`: Test set mAP@IoU=0.50
  - `test_mAP_75`: Test set mAP@IoU=0.75
  - `learning_rate`: Current learning rate
  - `epoch_time_min`: Epoch duration in minutes
  - `timestamp`: Date and time of completion
- **Frequency**: Every epoch (15 rows total)
- **Use**: Plot training curves, track convergence

### 3. **metrics_per_batch.csv**
- **Format**: CSV with headers
- **Columns**:
  - `epoch`: Current epoch
  - `batch`: Batch number within epoch
  - `loss_total`: Total combined loss
  - `loss_classifier`: Classification head loss
  - `loss_box_reg`: Bounding box regression loss
  - `loss_objectness`: RPN objectness loss
  - `loss_rpn_box_reg`: RPN box regression loss
  - `learning_rate`: Current learning rate
  - `time_per_batch_sec`: Batch processing time in seconds
  - `timestamp`: Date and time
- **Frequency**: Every 10 batches
- **Use**: Analyze loss component contributions, debug training issues

### 4. **evaluation_history.json**
- **Format**: JSON array of evaluation results
- **Content**: Complete evaluation metrics at epochs 5, 10, 15, and final
  ```json
  [
    {
      "epoch": 5,
      "timestamp": "2025-11-08 10:30:15",
      "metrics": {
        "mAP": 0.4523,
        "mAP_50": 0.7234,
        "mAP_75": 0.4892,
        "mAP_small": 0.2134,
        "mAP_medium": 0.4987,
        "mAP_large": 0.6123,
        "mAR_1": 0.3456,
        "mAR_10": 0.5234,
        "mAR_100": 0.5892,
        "mAR_small": 0.3123,
        "mAR_medium": 0.6234,
        "mAR_large": 0.7123,
        "num_test_images": 250
      }
    },
    // ... epochs 10, 15, and final
  ]
  ```
- **Use**: Track performance evolution, analyze convergence

### 5. **final_results.json**
- **Format**: Structured JSON
- **Content**:
  ```json
  {
    "experiment_id": "rgb_thermal",
    "config": {
      "active_modalities": ["rgb", "thermal"],
      "modality_config": "rgb+thermal",
      "architecture": "baseline_FRM_FFM",
      "backbone": "mit_b1",
      "epochs": 15,
      "batch_size": 32,
      "learning_rate": 0.04
    },
    "final_test_results": {
      // All mAP and mAR metrics
    },
    "evaluation_history": [
      // All evaluation points (epochs 5, 10, 15, final)
    ],
    "training": {
      "best_train_loss": 0.3456,
      "total_time_hours": 2.5,
      "total_epochs": 15
    },
    "model": {
      "total_params": 13500000,
      "trainable_params": 13500000,
      "total_params_M": 13.5,
      "trainable_params_M": 13.5
    },
    "timestamp": "2025-11-08 12:45:30",
    "output_files": {
      // List of all output files
    }
  }
  ```
- **Use**: Complete experiment summary, paper results, comparison

### 6. **config.json**
- **Format**: JSON
- **Content**: Experiment configuration parameters
- **Use**: Reproduce experiments, track hyperparameters

### 7. **model_info.json**
- **Format**: JSON
- **Content**: Model architecture statistics
  - Total parameters
  - Trainable parameters
  - Parameters in millions
- **Use**: Model complexity comparison

### 8-10. **checkpoint_epoch_X.pth** (X = 5, 10, 15)
- **Format**: PyTorch checkpoint
- **Content**:
  - `model_state_dict`: Model weights
  - `optimizer_state_dict`: Optimizer state
  - `epoch`: Current epoch
  - `loss`: Current loss
  - `best_loss`: Best loss so far
  - `config`: Experiment configuration
- **Use**: Resume training from specific epoch, analyze intermediate models

### 11. **checkpoint_latest.pth**
- **Format**: PyTorch checkpoint (same as above)
- **Content**: Most recent checkpoint (epoch 15)
- **Use**: Quick access to latest state for resuming

### 12. **checkpoint_best.pth**
- **Format**: PyTorch checkpoint (full state)
- **Content**: Checkpoint from epoch with lowest training loss
- **Use**: Resume from best performing epoch

### 13. **model_best_weights.pth**
- **Format**: PyTorch state_dict
- **Content**: Model weights only (no optimizer)
- **Use**: Deployment, inference, loading for evaluation

## Evaluation Metrics Logged

### Mean Average Precision (mAP)
- `mAP`: Average precision across all IoU thresholds (0.5:0.95)
- `mAP_50`: Precision at IoU=0.50
- `mAP_75`: Precision at IoU=0.75
- `mAP_small`: Precision for small objects (area < 32²)
- `mAP_medium`: Precision for medium objects (32² < area < 96²)
- `mAP_large`: Precision for large objects (area > 96²)

### Mean Average Recall (mAR)
- `mAR_1`: Recall with 1 detection per image
- `mAR_10`: Recall with up to 10 detections per image
- `mAR_100`: Recall with up to 100 detections per image
- `mAR_small`: Recall for small objects
- `mAR_medium`: Recall for medium objects
- `mAR_large`: Recall for large objects

### Loss Components
- `loss_total`: Sum of all losses
- `loss_classifier`: Classification head loss
- `loss_box_reg`: Box regression loss (detection head)
- `loss_objectness`: RPN objectness score loss
- `loss_rpn_box_reg`: RPN box regression loss

## Evaluation Schedule

- **Per-batch logging**: Every 10 batches during training
- **Per-epoch logging**: Every epoch (15 times)
- **Test set evaluation**: Epochs 5, 10, 15, and final (4 times total)
- **Checkpoints**: Epochs 5, 10, 15 (3 checkpoints + latest + best)

## Storage Requirements

Per experiment (approximate):
- Logs and CSVs: ~50-100 MB
- Checkpoints (5 total): ~1.2-1.5 GB each = ~6-7.5 GB
- Total per experiment: ~6-8 GB
- Total for 3 experiments: ~18-24 GB

## Quick Access Examples

### Get final mAP for all experiments
```bash
grep "Test mAP:" results/modality_ablations/*/training.log | tail -3
```

### Get best training loss
```bash
jq '.training.best_train_loss' results/modality_ablations/*/final_results.json
```

### Compare training times
```bash
jq '.training.total_time_hours' results/modality_ablations/*/final_results.json
```

### Extract all mAP values
```python
import json
import glob

for path in glob.glob('results/modality_ablations/*/final_results.json'):
    with open(path) as f:
        data = json.load(f)
        exp_name = data['experiment_id']
        mAP = data['final_test_results']['mAP']
        print(f"{exp_name}: mAP = {mAP:.4f}")
```

## Data Integrity

All files include:
- **Timestamps**: Every logged entry has a timestamp
- **Configuration tracking**: Experiment config saved multiple times
- **Versioning**: Architecture type explicitly logged ("baseline_FRM_FFM")
- **Resumability**: Full training state saved for resuming

## Reproducibility

To reproduce an experiment:
1. Load `config.json` to get exact hyperparameters
2. Use same data paths
3. Set same random seed (if specified)
4. Or load `checkpoint_epoch_X.pth` and resume training

## Analysis Recommendations

1. **For CVPR paper**:
   - Use `final_results.json` for result tables
   - Use `metrics_per_epoch.csv` for training curves
   - Use `evaluation_history.json` for convergence analysis

2. **For debugging**:
   - Check `training.log` for detailed progress
   - Analyze `metrics_per_batch.csv` for loss component issues
   - Compare across modality combinations

3. **For visualization**:
   - Plot training/validation curves from CSV files
   - Create loss component breakdown charts
   - Show mAP progression over training
