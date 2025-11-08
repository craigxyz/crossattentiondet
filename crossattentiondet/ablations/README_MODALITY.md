# Modality Ablation Experiments

This directory contains scripts and infrastructure for running modality ablation experiments to understand the individual and combined contributions of different input modalities (RGB, Thermal, Event) to detection performance.

## Overview

The modality ablation system allows you to selectively enable/disable input modalities to answer questions like:
- How much does each modality contribute to detection performance?
- Which modality pairs work best together?
- Is RGB necessary when thermal and event data are available?
- What is the complementary information provided by thermal vs. event modalities?

## Key Components

### 1. ModalityConfigurableBackbone (`backbone_modality.py`)

A wrapper around the standard CrossAttentionBackbone that allows selective masking of input modalities.

**Features:**
- Zero out specific modality channels based on configuration
- Supports any combination of RGB, Thermal, and Event
- Compatible with existing encoder architectures
- No architectural changes required - just input masking

**Usage:**
```python
from crossattentiondet.ablations.backbone_modality import ModalityConfigurableBackbone

# Create backbone with only RGB and Thermal active
backbone = ModalityConfigurableBackbone(
    encoder,
    fpn_out_channels=256,
    active_modalities=['rgb', 'thermal']  # Event will be zeroed out
)
```

### 2. Training Script (`scripts/train_modality_ablation.py`)

Enhanced training script specifically for modality ablation experiments.

**Key Features:**
- Configurable modality combinations via `--modalities` flag
- Comprehensive logging (training.log, CSV metrics, JSON results)
- Compatible with existing GAFF/CSSA ablation infrastructure
- Reduced default epochs (10) for faster iteration

**Usage:**
```bash
python crossattentiondet/ablations/scripts/train_modality_ablation.py \
    --data /path/to/images \
    --labels /path/to/labels \
    --output-dir results/modality_ablations/rgb_thermal \
    --modalities "rgb,thermal" \
    --backbone mit_b1 \
    --epochs 10 \
    --batch-size 16
```

### 3. Master Runner Script (`scripts/run_modality_ablations.py`)

Orchestrates all modality ablation experiments with a single command.

**Tested Combinations:**
1. **RGB + Thermal** - No event data
2. **RGB + Event** - No thermal data
3. **Thermal + Event** - No RGB data
4. **All modalities** - Baseline for comparison

**Usage:**
```bash
# Run all modality ablation experiments
python crossattentiondet/ablations/scripts/run_modality_ablations.py \
    --data /path/to/images \
    --labels /path/to/labels \
    --output-dir results/modality_ablations \
    --epochs 10 \
    --backbone mit_b1

# Run specific experiments only
python crossattentiondet/ablations/scripts/run_modality_ablations.py \
    --data /path/to/images \
    --labels /path/to/labels \
    --output-dir results/modality_ablations \
    --experiments "rgb_thermal,rgb_event"
```

## Quick Start

### Prerequisites

Ensure your environment is set up with the required dependencies:
```bash
# From project root
pip install -r requirements.txt
```

### Running All Experiments

The easiest way to run all modality ablations:

```bash
python crossattentiondet/ablations/scripts/run_modality_ablations.py \
    --data ../RGBX_Semantic_Segmentation/data/images \
    --labels ../RGBX_Semantic_Segmentation/data/labels \
    --output-dir results/modality_ablations \
    --epochs 10 \
    --backbone mit_b1 \
    --batch-size 16
```

### Running Individual Experiments

For more control, run experiments individually:

```bash
# RGB + Thermal only
python crossattentiondet/ablations/scripts/train_modality_ablation.py \
    --data ../RGBX_Semantic_Segmentation/data/images \
    --labels ../RGBX_Semantic_Segmentation/data/labels \
    --output-dir results/modality_ablations/rgb_thermal \
    --modalities "rgb,thermal" \
    --epochs 10

# RGB + Event only
python crossattentiondet/ablations/scripts/train_modality_ablation.py \
    --data ../RGBX_Semantic_Segmentation/data/images \
    --labels ../RGBX_Semantic_Segmentation/data/labels \
    --output-dir results/modality_ablations/rgb_event \
    --modalities "rgb,event" \
    --epochs 10

# Thermal + Event only
python crossattentiondet/ablations/scripts/train_modality_ablation.py \
    --data ../RGBX_Semantic_Segmentation/data/images \
    --labels ../RGBX_Semantic_Segmentation/data/labels \
    --output-dir results/modality_ablations/thermal_event \
    --modalities "thermal,event" \
    --epochs 10
```

## Output Structure

Each experiment creates a dedicated output directory with comprehensive logging:

```
results/modality_ablations/
├── rgb_thermal/
│   ├── training.log              # Detailed training log
│   ├── metrics_per_epoch.csv     # Epoch-level metrics
│   ├── metrics_per_batch.csv     # Batch-level metrics
│   ├── config.json               # Experiment configuration
│   ├── model_info.json           # Model architecture info
│   ├── final_results.json        # Final evaluation results
│   ├── checkpoint.pth            # Latest checkpoint
│   └── checkpoint_best.pth       # Best checkpoint by loss
├── rgb_event/
│   └── ...
├── thermal_event/
│   └── ...
├── all_modalities/
│   └── ...
└── ablation_summary.json         # Summary of all experiments
```

## Configuration Details

### Default GAFF Configuration

The modality ablation experiments use the following GAFF configuration (adjust based on your GAFF ablation results):

```python
gaff_stages=[1, 2, 3, 4]      # Use GAFF at all stages
gaff_se_reduction=4            # SE block reduction ratio
gaff_inter_shared=False        # No shared inter-modality attention
gaff_merge_bottleneck=False    # No bottleneck in merge layer
```

### Training Hyperparameters

**Recommended for fast iteration:**
- Epochs: 10 (vs. 25 for full training)
- Batch size: 16
- Learning rate: 0.02
- Optimizer: SGD with momentum 0.9
- Weight decay: 0.0001

**For CVPR-quality results, consider:**
- Epochs: 25-50
- Multiple backbone variants (mit_b1, mit_b2)
- Cross-validation across multiple runs

## Expected Computation Time

**Per experiment (10 epochs, mit_b1, batch_size=16):**
- On A100 GPU: ~2-3 hours
- On V100 GPU: ~3-4 hours

**Total for all 4 experiments:**
- On A100 GPU: ~8-12 hours
- On V100 GPU: ~12-16 hours

## Analysis and Visualization

### Loading Results

```python
import json

# Load results for a specific experiment
with open('results/modality_ablations/rgb_thermal/final_results.json', 'r') as f:
    results = json.load(f)

print(f"mAP: {results['results']['mAP']}")
print(f"mAP@50: {results['results']['mAP_50']}")
print(f"mAP@75: {results['results']['mAP_75']}")
```

### Comparing All Experiments

```python
import json
import pandas as pd

# Load summary
with open('results/modality_ablations/ablation_summary.json', 'r') as f:
    summary = json.load(f)

# Extract mAP results
results_data = []
for exp in summary['experiments']:
    if exp['status'] == 'success':
        with open(f"{exp['output_dir']}/final_results.json", 'r') as f:
            exp_results = json.load(f)
            results_data.append({
                'Modalities': exp['modalities'],
                'mAP': exp_results['results']['mAP'],
                'mAP@50': exp_results['results']['mAP_50'],
                'mAP@75': exp_results['results']['mAP_75']
            })

df = pd.DataFrame(results_data)
print(df.to_string(index=False))
```

### Expected Insights

Based on typical multi-modal object detection results, you might expect:

1. **RGB + Thermal** - Strong performance, especially in low-light/night scenarios
2. **RGB + Event** - Good motion capture, may struggle with static objects
3. **Thermal + Event** - Interesting for RGB-free scenarios, performance depends on task
4. **All modalities** - Best overall performance due to complementary information

## CVPR Presentation Tips

### Key Results to Report

1. **Performance Comparison Table**
   ```
   | Modalities      | mAP   | mAP@50 | mAP@75 | Relative ∆ |
   |-----------------|-------|--------|--------|------------|
   | RGB+Thermal+Event | XX.X | XX.X   | XX.X   | baseline   |
   | RGB+Thermal     | XX.X | XX.X   | XX.X   | -X.X%      |
   | RGB+Event       | XX.X | XX.X   | XX.X   | -X.X%      |
   | Thermal+Event   | XX.X | XX.X   | XX.X   | -X.X%      |
   ```

2. **Modality Contribution Analysis**
   - Which modality contributes most to detection?
   - Are modalities complementary or redundant?
   - Performance degradation when each modality is removed

3. **Computational Efficiency**
   - Model parameters (should be identical across experiments)
   - Inference time comparison (if relevant)

### Visualization Ideas

1. **Bar chart**: mAP comparison across all modality combinations
2. **Radar plot**: Multi-metric comparison (mAP, mAP@50, mAP@75, etc.)
3. **Confusion analysis**: Which objects benefit most from each modality?
4. **Qualitative results**: Side-by-side detection visualizations

## Troubleshooting

### Common Issues

**Issue: Out of memory**
```bash
# Reduce batch size
python ... --batch-size 8
```

**Issue: Experiment fails partway through**
```bash
# Check individual experiment logs
cat results/modality_ablations/rgb_thermal/training.log

# Resume from specific experiment
python scripts/run_modality_ablations.py \
    --experiments "thermal_event,all_modalities"  # Skip completed ones
```

**Issue: Poor performance with thermal+event only**
- This is expected - RGB provides critical texture/shape information
- Useful for understanding modality importance, not necessarily for deployment

## Extension Ideas

### Adding More Combinations

To test single-modality experiments, modify `run_modality_ablations.py`:

```python
MODALITY_COMBINATIONS = [
    # ... existing combinations ...
    {'name': 'rgb_only', 'modalities': 'rgb', 'description': 'RGB only'},
    {'name': 'thermal_only', 'modalities': 'thermal', 'description': 'Thermal only'},
    {'name': 'event_only', 'modalities': 'event', 'description': 'Event only'},
]
```

### Testing Different Fusion Strategies

Modify `train_modality_ablation.py` to test different GAFF configurations:

```python
# In build_model():
encoder_base = get_gaff_encoder(
    # ...
    gaff_stages=[4],  # Try different stage configurations
    gaff_se_reduction=8,  # Try different reduction ratios
)
```

## References

For more information on the underlying architecture:
- GAFF ablations: `crossattentiondet/ablations/fusion/README_GAFF.md`
- CSSA ablations: `crossattentiondet/ablations/README_CSSA.md`
- Main framework: `README.md`

## Contact

For questions or issues related to modality ablation experiments, please open an issue in the repository.
