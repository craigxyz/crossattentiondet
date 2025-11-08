# Modality Ablation Experiments - Quick Reference

## What Was Created

A complete modality ablation system to test different combinations of RGB, Thermal, and Event modalities for your CVPR paper.

**IMPORTANT:** These experiments use the **BASELINE architecture** (FRM+FFM fusion), NOT the GAFF or CSSA ablations. This isolates the effect of modality combinations from fusion strategy effects.

### Files Created

1. **`crossattentiondet/ablations/backbone_modality.py`**
   - Modality-configurable backbone wrapper
   - Selectively masks input modalities (zeros out channels)

2. **`crossattentiondet/ablations/scripts/train_modality_ablation.py`**
   - Training script for individual modality experiments
   - Comprehensive logging and metrics tracking

3. **`crossattentiondet/ablations/scripts/run_modality_ablations.py`**
   - Master script to run all experiments sequentially
   - Generates summary report

4. **`crossattentiondet/ablations/scripts/QUICKSTART_MODALITY.sh`**
   - One-command launch script
   - Pre-configured with recommended settings

5. **`crossattentiondet/ablations/README_MODALITY.md`**
   - Complete documentation
   - Usage examples, troubleshooting, analysis tips

## Modality Combinations Tested

| Experiment       | RGB | Thermal | Event | Description                    |
|------------------|-----|---------|-------|--------------------------------|
| rgb_thermal      | ✓   | ✓       | ✗     | No event data                  |
| rgb_event        | ✓   | ✗       | ✓     | No thermal data                |
| thermal_event    | ✗   | ✓       | ✓     | No RGB (RGB-free scenario)     |

**Note:** All modalities baseline (RGB+Thermal+Event) already exists in your other experiments:
- `gaff_pilot_5epochs/`
- `results/gaff_ablations_full/`
- `results/cssa_ablations/`

## Quick Start (Recommended)

### Option 1: One-Command Launch

```bash
# Edit paths in the script first, then run:
./crossattentiondet/ablations/scripts/QUICKSTART_MODALITY.sh
```

### Option 2: Simple Launch Script (Recommended)

```bash
# Edit paths in run_modality_experiments.sh first, then:
./run_modality_experiments.sh
```

### Option 3: Manual Launch

```bash
python crossattentiondet/ablations/scripts/run_modality_ablations.py \
    --data /path/to/your/images \
    --labels /path/to/your/labels \
    --output-dir results/modality_ablations \
    --epochs 15 \
    --backbone mit_b1 \
    --batch-size 32 \
    --lr 0.04
```

### Option 4: Individual Experiments

```bash
# Just RGB + Thermal
python crossattentiondet/ablations/scripts/train_modality_ablation.py \
    --data /path/to/images \
    --labels /path/to/labels \
    --modalities "rgb,thermal" \
    --output-dir results/modality_ablations/rgb_thermal \
    --epochs 15 \
    --batch-size 32 \
    --lr 0.04

# Just RGB + Event
python crossattentiondet/ablations/scripts/train_modality_ablation.py \
    --data /path/to/images \
    --labels /path/to/labels \
    --modalities "rgb,event" \
    --output-dir results/modality_ablations/rgb_event \
    --epochs 15 \
    --batch-size 32 \
    --lr 0.04

# Just Thermal + Event
python crossattentiondet/ablations/scripts/train_modality_ablation.py \
    --data /path/to/images \
    --labels /path/to/labels \
    --modalities "thermal,event" \
    --output-dir results/modality_ablations/thermal_event \
    --epochs 15 \
    --batch-size 32 \
    --lr 0.04
```

## Settings for CVPR

### Current Configuration (Default)
- **Architecture**: Baseline FRM+FFM (NOT GAFF/CSSA)
- **Epochs**: 15
- **Backbone**: mit_b1
- **Batch Size**: 32
- **Learning Rate**: 0.04 (scaled with batch size)
- **Experiments**: 3 modality combinations
- **Time**: ~6-9 hours total on A100

### For More Robust Results (Optional)
- **Epochs**: 25-30
- **Backbone**: mit_b1 or mit_b2
- **Multiple runs**: 3 runs with different seeds for error bars
- **Experiments**: 3 modality combinations
- **Time**: ~18-27 hours total on A100 (per run)

## Expected Output Structure

Each experiment creates a comprehensive set of output files for detailed analysis:

```
results/modality_ablations/
├── rgb_thermal/
│   ├── training.log                  ← Detailed training log with timestamps
│   ├── metrics_per_epoch.csv         ← Per-epoch: train loss, test mAP/mAP@50/mAP@75, LR, time
│   ├── metrics_per_batch.csv         ← Per-batch: all loss components (total, classifier, box_reg, objectness, rpn)
│   ├── evaluation_history.json       ← All evaluation results throughout training (epochs 5, 10, 15)
│   ├── final_results.json            ← Complete final results with all metrics and metadata
│   ├── config.json                   ← Experiment configuration
│   ├── model_info.json               ← Model architecture info (total params, trainable params)
│   ├── checkpoint_epoch_5.pth        ← Full checkpoint at epoch 5 (model + optimizer + state)
│   ├── checkpoint_epoch_10.pth       ← Full checkpoint at epoch 10
│   ├── checkpoint_epoch_15.pth       ← Full checkpoint at epoch 15 (final)
│   ├── checkpoint_latest.pth         ← Latest checkpoint (for resuming training)
│   ├── checkpoint_best.pth           ← Best checkpoint (full training state)
│   └── model_best_weights.pth        ← Best model weights only (for deployment/inference)
├── rgb_event/
│   └── ... (same structure)
├── thermal_event/
│   └── ... (same structure)
└── ablation_summary.json             ← Overall summary of all 3 experiments
```

### Logged Metrics

**Per-Batch CSV** (every 10 batches):
- Total loss + individual components (classifier, box_reg, objectness, rpn_box_reg)
- Learning rate
- Batch processing time

**Per-Epoch CSV** (every epoch):
- Training loss (average)
- Test set evaluation (mAP, mAP@50, mAP@75) at epochs 5, 10, 15
- Learning rate
- Epoch time

**Evaluation History JSON** (epochs 5, 10, 15 + final):
- All mAP metrics (mAP, mAP@50, mAP@75, mAP_small, mAP_medium, mAP_large)
- All mAR metrics (mAR@1, mAR@10, mAR@100, mAR_small, mAR_medium, mAR_large)
- Number of test images
- Timestamp for each evaluation

**Final Results JSON**:
- Complete configuration
- Final test set metrics (all mAP and mAR)
- Full evaluation history
- Training statistics (best loss, total time, epochs)
- Model info (parameters)
- List of all output files

## Quick Analysis

### Load and Compare Final Results

```python
import json
import pandas as pd

# Load all results (3 modality combinations)
experiments = ['rgb_thermal', 'rgb_event', 'thermal_event']
results = []

for exp in experiments:
    with open(f'results/modality_ablations/{exp}/final_results.json') as f:
        data = json.load(f)
        test_results = data['final_test_results']
        results.append({
            'Modalities': data['config']['modality_config'],
            'Architecture': data['config']['architecture'],
            'mAP': test_results['mAP'],
            'mAP@50': test_results['mAP_50'],
            'mAP@75': test_results['mAP_75'],
            'mAR@100': test_results['mAR_100'],
            'Train_Loss': data['training']['best_train_loss'],
            'Time_Hours': data['training']['total_time_hours']
        })

df = pd.DataFrame(results)
print(df)

# Add baseline from existing experiments (for comparison)
# with open('gaff_pilot_5epochs/final_results.json') as f:
#     baseline = json.load(f)
#     # Add baseline row to comparison
```

### Plot Training Progress

```python
import matplotlib.pyplot as plt

# Load per-epoch metrics for an experiment
epoch_data = pd.read_csv('results/modality_ablations/rgb_thermal/metrics_per_epoch.csv')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Training loss
ax1.plot(epoch_data['epoch'], epoch_data['train_loss'], marker='o')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Training Loss')
ax1.set_title('Training Loss Over Time')
ax1.grid(True)

# Test mAP
ax2.plot(epoch_data['epoch'], epoch_data['test_mAP'], marker='s', label='mAP')
ax2.plot(epoch_data['epoch'], epoch_data['test_mAP_50'], marker='^', label='mAP@50')
ax2.plot(epoch_data['epoch'], epoch_data['test_mAP_75'], marker='v', label='mAP@75')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('mAP')
ax2.set_title('Test Set Performance')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('training_progress.png')
```

### Analyze Loss Components

```python
# Load per-batch metrics to analyze loss breakdown
batch_data = pd.read_csv('results/modality_ablations/rgb_thermal/metrics_per_batch.csv')

# Calculate average loss per epoch
loss_components = batch_data.groupby('epoch')[
    ['loss_total', 'loss_classifier', 'loss_box_reg', 'loss_objectness', 'loss_rpn_box_reg']
].mean()

print("Loss Component Breakdown:")
print(loss_components)

# Plot loss components
loss_components.plot(kind='bar', stacked=False, figsize=(12, 6))
plt.ylabel('Loss Value')
plt.xlabel('Epoch')
plt.title('Loss Components by Epoch')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('loss_components.png')
```

### Access Full Evaluation History

```python
# Load complete evaluation history across training
with open('results/modality_ablations/rgb_thermal/evaluation_history.json') as f:
    eval_history = json.load(f)

for eval_point in eval_history:
    print(f"\nEpoch {eval_point['epoch']} ({eval_point['timestamp']}):")
    metrics = eval_point['metrics']
    print(f"  mAP: {metrics['mAP']:.4f}")
    print(f"  mAP@50: {metrics['mAP_50']:.4f}")
    print(f"  mAR@100: {metrics['mAR_100']:.4f}")
```

## Key Parameters

| Parameter       | Default    | Description                              |
|-----------------|------------|------------------------------------------|
| --modalities    | (required) | Comma-separated list: rgb,thermal,event  |
| --epochs        | 15         | Number of training epochs                |
| --backbone      | mit_b1     | Backbone architecture                    |
| --batch-size    | 32         | Training batch size                      |
| --lr            | 0.04       | Learning rate (scaled with batch size)   |
| --data          | (required) | Path to image data directory             |
| --labels        | (required) | Path to labels directory                 |
| --output-dir    | (required) | Output directory for results             |

## Computation Time Estimates

| GPU    | Per Experiment | All 3 Experiments |
|--------|----------------|-------------------|
| A100   | 2-3 hours      | 6-9 hours         |
| V100   | 3-4 hours      | 9-12 hours        |
| 3090   | 3-4 hours      | 9-12 hours        |

*(Based on 15 epochs, mit_b1, batch_size=32, lr=0.04, baseline FRM+FFM architecture)*

## CVPR Paper Suggestions

### Table for Paper

```latex
\begin{table}
\centering
\caption{Modality ablation results on [Dataset Name]}
\begin{tabular}{lccc}
\hline
Modalities & mAP & mAP@50 & mAP@75 \\
\hline
RGB + Thermal + Event & XX.X & XX.X & XX.X \\
RGB + Thermal & XX.X & XX.X & XX.X \\
RGB + Event & XX.X & XX.X & XX.X \\
Thermal + Event & XX.X & XX.X & XX.X \\
\hline
\end{tabular}
\end{table}
```

### Key Claims to Support

1. **Modality Complementarity**: "When event data is removed (RGB+Thermal), performance drops by X%, demonstrating the complementary information provided by event cameras."

2. **RGB Necessity**: "The Thermal+Event configuration achieves XX% mAP, showing that while thermal and event modalities contain useful information, RGB remains critical for..."

3. **Best Pair**: "Among two-modality configurations, RGB+Thermal achieves the highest performance (XX% mAP), outperforming RGB+Event by X%."

## Troubleshooting

**Out of memory?**
```bash
--batch-size 8  # or even 4
```

**Want faster results?**
```bash
--epochs 5  # Quick proof-of-concept
```

**Need more robust results?**
```bash
--epochs 25  # Better convergence
```

## Next Steps

1. **Run experiments**: Use quick start script or manual commands
2. **Analyze results**: Check `final_results.json` in each experiment directory
3. **Create visualizations**: Bar charts, radar plots comparing modality combinations
4. **Write paper section**: Use results to support claims about modality importance

## Support

- Full documentation: `crossattentiondet/ablations/README_MODALITY.md`
- GAFF ablations: `crossattentiondet/ablations/fusion/README_GAFF.md`
- Main README: `README.md`
