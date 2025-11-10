# 15-Epoch Baseline Model Evaluation

## Overview

This script evaluates the fully trained 15-epoch baseline models that use all 3 modalities (RGB+Thermal+Event) with the standard FRM+FFM fusion architecture.

## Models to be Evaluated

| Backbone | Parameters | Training Loss | Training Time | Checkpoint Path |
|----------|------------|---------------|---------------|-----------------|
| **mit_b0** | 27.79M | 0.1057 | 2.68 hours | `training_logs/run_20251107_102948/mit_b0/checkpoints/best_model.pth` |
| **mit_b1** | 60.01M | 0.1027 | 3.44 hours | `training_logs/run_20251107_102948/mit_b1/checkpoints/best_model.pth` |
| **mit_b2** | 82.10M | 0.1043 | 6.76 hours | `training_logs/run_20251107_102948/mit_b2/checkpoints/best_model.pth` |

All models were trained with:
- **Epochs:** 15
- **Batch Size:** 16 (effective, mit_b2 used gradient accumulation)
- **Learning Rate:** 0.02
- **Modalities:** RGB + Thermal + Event (all 3)
- **Architecture:** Standard FRM+FFM baseline

## Usage

### Basic Evaluation

Run evaluation on all three models:

```bash
python evaluate_15epoch_baselines.py
```

### Custom Data Path

Specify a custom data directory:

```bash
python evaluate_15epoch_baselines.py --data /path/to/data
```

### Custom Output Directory

Save results to a custom location:

```bash
python evaluate_15epoch_baselines.py --output results/my_baseline_eval
```

## Output Structure

The script creates the following output structure:

```
results/baseline_15epoch_evaluated/
├── evaluation_summary.json          # Overall summary of all evaluations
├── mit_b0/
│   ├── final_results.json          # Complete results for mit_b0
│   └── predictions/                # Detection predictions (if saved)
├── mit_b1/
│   ├── final_results.json          # Complete results for mit_b1
│   └── predictions/
└── mit_b2/
    ├── final_results.json          # Complete results for mit_b2
    └── predictions/
```

## Results Format

Each `final_results.json` contains:

```json
{
  "experiment_id": "baseline_15epoch_mit_b1",
  "backbone": "mit_b1",
  "architecture": "baseline_FRM_FFM",
  "config": {
    "epochs": 15,
    "batch_size": 16,
    "learning_rate": 0.02,
    "modalities": ["rgb", "thermal", "event"]
  },
  "training": {
    "best_epoch": 15,
    "best_loss": 0.1027,
    "total_time_hours": 3.44
  },
  "model": {
    "total_params": 60014945,
    "total_params_M": 60.01
  },
  "evaluation": {
    "metrics": {
      "mAP": 0.xxxx,
      "mAP_50": 0.xxxx,
      "mAP_75": 0.xxxx,
      "mAP_small": 0.xxxx,
      "mAP_medium": 0.xxxx,
      "mAP_large": 0.xxxx
    },
    "eval_time_sec": xxx,
    "num_test_images": 1950
  }
}
```

## Evaluation Metrics

The script computes:

- **mAP** - Mean Average Precision (COCO style, averaged over IoU thresholds 0.5:0.95)
- **mAP@50** - Average Precision at IoU threshold 0.5
- **mAP@75** - Average Precision at IoU threshold 0.75
- **mAP_small** - mAP for small objects (area < 32²)
- **mAP_medium** - mAP for medium objects (32² < area < 96²)
- **mAP_large** - mAP for large objects (area > 96²)

## Expected Runtime

Evaluation time depends on the test set size and model complexity:

- **mit_b0**: ~1-2 minutes
- **mit_b1**: ~2-3 minutes
- **mit_b2**: ~3-5 minutes

**Total time:** ~10 minutes for all three models

## Comparison with Ablation Studies

These baseline results will be used to compare against:

1. **GAFF Ablations** (17 experiments)
   - Best GAFF: ~83.78% mAP
   - Adds gated attention fusion at various stages

2. **CSSA Ablations** (13 experiments)
   - Best CSSA: ~84.07% mAP
   - Adds channel switching and spatial attention

3. **Modality Ablations** (3 experiments)
   - RGB+Thermal: 83.42% mAP (2-modality baseline)
   - These are partial baselines with only 2 modalities

The key question: **How much do the ablation fusion methods improve over the full 3-modality baseline?**

## Troubleshooting

### CUDA Out of Memory

If you encounter OOM errors, reduce batch size in the script:

```python
self.test_loader = DataLoader(
    self.test_dataset,
    batch_size=1,  # Already set to 1 for safety
    ...
)
```

### Checkpoint Not Found

Verify checkpoint paths exist:

```bash
ls -lh training_logs/run_20251107_102948/*/checkpoints/best_model.pth
```

### Missing Data

Ensure data directory has the correct structure:

```
../RGBX_Semantic_Segmentation/data/
├── images/    (*.npy files)
└── labels/    (*.txt files)
```

## Next Steps After Evaluation

1. **Update Comprehensive Report**
   - Add 15-epoch baseline results to `COMPREHENSIVE_EXPERIMENTAL_RESULTS.md`
   - Compare with ablation studies

2. **Analysis**
   - Compare 3-modality baseline vs 2-modality baselines
   - Evaluate if ablation methods justify the added complexity
   - Determine best overall configuration

3. **Documentation**
   - Document final baseline performance
   - Create comparison charts
   - Update executive summary

## Notes

- These are the **TRUE baselines** with all 3 modalities
- Previous 1-epoch evaluations showed ~66% mAP (under-trained)
- 15-epoch training should show significant improvement
- Results will be directly comparable to GAFF and CSSA ablations which also trained for 15 epochs

---

**Script:** `evaluate_15epoch_baselines.py`
**Created:** 2025-11-10
**Purpose:** Establish proper baseline performance for comparison with ablation studies
