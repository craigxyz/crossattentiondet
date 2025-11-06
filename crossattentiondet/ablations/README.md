# CSSA Fusion Ablation Experiments

## Overview

This directory contains the implementation of CSSA (Channel Switching and Spatial Attention) fusion for ablation experiments. The goal is to compare CSSA fusion against the baseline FRM+FFM fusion at Stage 4 of the encoder.

## What's Implemented

### 1. Fusion Modules (`fusion/`)
- **`base.py`**: Abstract `FusionBlock` interface
- **`cssa.py`**: Complete CSSA implementation with:
  - `ECABlock`: Efficient Channel Attention
  - `ChannelSwitching`: Channel-wise swapping between modalities
  - `SpatialAttention`: Spatial attention with avg+max pooling
  - `CSSABlock`: Complete fusion module

### 2. CSSA-Enabled Encoder (`encoder_cssa.py`)
- `RGBXTransformerCSSA`: Modified encoder with CSSA at Stage 4
- Stages 1-3: Original FRM+FFM (unchanged)
- Stage 4: CSSA fusion (the ablation point)
- All backbone variants: `mit_b0_cssa`, `mit_b1_cssa`, `mit_b2_cssa`, `mit_b4_cssa`, `mit_b5_cssa`

### 3. Training Script (`scripts/train_cssa.py`)
- Modified trainer that uses CSSA-enabled encoder
- Identical training loop to baseline
- Additional CLI args for CSSA hyperparameters

## Setup Environment

Before running experiments, set up your environment:

```bash
# Activate your conda/virtualenv with PyTorch
# Make sure you have: torch, torchvision, numpy, pillow

# Verify PyTorch is available with CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

## Running Experiments

### Experiment 1: 5-Epoch Sanity Check

Quick test to verify everything works:

```bash
python crossattentiondet/ablations/scripts/train_cssa.py \
    --data data/images \
    --labels data/labels \
    --epochs 5 \
    --batch-size 2 \
    --backbone mit_b1 \
    --model checkpoints/cssa_stage4_sanity.pth \
    --results-dir test_results/cssa_sanity
```

**Expected outcome:**
- Training runs without errors
- Loss decreases over 5 epochs
- Checkpoint saved to `checkpoints/cssa_stage4_sanity.pth`
- Evaluation metrics printed at end

### Experiment 2: Full 25-Epoch Training

Full training run for comparison with baseline:

```bash
python crossattentiondet/ablations/scripts/train_cssa.py \
    --data data/images \
    --labels data/labels \
    --epochs 25 \
    --batch-size 2 \
    --backbone mit_b1 \
    --lr 0.005 \
    --model checkpoints/cssa_stage4_25epoch.pth \
    --results-dir test_results/cssa_25epoch
```

**Expected outcome:**
- Training for 25 epochs (~6-12 hours depending on GPU)
- Checkpoint saved every 5 epochs
- Final evaluation with mAP metrics

### Experiment 3: Hyperparameter Sweep (Optional)

Test different CSSA thresholds:

```bash
# Threshold 0.3 (more switching)
python crossattentiondet/ablations/scripts/train_cssa.py \
    --data data/images \
    --labels data/labels \
    --epochs 25 \
    --batch-size 2 \
    --backbone mit_b1 \
    --cssa-thresh 0.3 \
    --model checkpoints/cssa_thresh03.pth

# Threshold 0.7 (less switching)
python crossattentiondet/ablations/scripts/train_cssa.py \
    --data data/images \
    --labels data/labels \
    --epochs 25 \
    --batch-size 2 \
    --backbone mit_b1 \
    --cssa-thresh 0.7 \
    --model checkpoints/cssa_thresh07.pth
```

## Comparing with Baseline

### Load Baseline Results

Your existing baseline checkpoint is at:
```
checkpoints/crossattentiondet_mit_b1.pth
```

### Evaluation Comparison

```bash
# Evaluate baseline (existing script)
python scripts/test.py \
    --data data/images \
    --labels data/labels \
    --model checkpoints/crossattentiondet_mit_b1.pth \
    --backbone mit_b1

# Evaluate CSSA (note: need to modify test.py or create test_cssa.py)
# For now, evaluation happens automatically at end of training
```

### Expected Metrics to Compare

| Metric | Baseline FRM+FFM | CSSA (thresh=0.5) |
|--------|------------------|-------------------|
| mAP@[.5:.95] | ? | ? |
| mAP@.50 | ? | ? |
| mAP@.75 | ? | ? |
| Params | ? | ? (similar, <10K diff) |
| Training time | ? | ? (similar) |

## CSSA Hyperparameters

### `--cssa-thresh` (default: 0.5)
Channel switching threshold. When channel attention weight is below this threshold, swap channel from alternative modality.
- Lower (0.2-0.4): More aggressive switching
- Medium (0.5): Balanced
- Higher (0.6-0.8): Conservative switching

### `--cssa-kernel` (default: 3)
Kernel size for ECA 1D convolution (must be odd).
- 3: Smaller receptive field, faster
- 5: Larger receptive field, may capture longer-range channel dependencies

## Testing the Module

To verify CSSA implementation before training:

```bash
# Run unit tests (requires PyTorch)
cd crossattentiondet/ablations/fusion
python test_cssa.py
```

**Expected output:**
```
============================================================
CSSA Module Unit Tests
============================================================
Testing CSSA module with various input shapes...

Test case: B=2, C=64, H=32, W=32
  ✓ Output shape: torch.Size([2, 64, 32, 32])
  ✓ Output range: [-0.5234, 0.7821]
  ✓ Output mean: 0.0123, std: 0.1234

... (more test cases)

✅ All shape tests passed!
✅ Gradient flow test passed!
✅ Parameter count test passed!

============================================================
All tests passed! ✅
============================================================
```

## File Structure

```
crossattentiondet/ablations/
├── __init__.py
├── README.md                          # This file
├── encoder_cssa.py                    # CSSA-enabled encoder
├── fusion/
│   ├── __init__.py
│   ├── base.py                        # FusionBlock interface
│   ├── cssa.py                        # CSSA implementation
│   └── test_cssa.py                   # Unit tests
└── scripts/
    └── train_cssa.py                  # Training script
```

## Troubleshooting

### Issue: ModuleNotFoundError for torch
**Solution:** Activate your conda/virtualenv with PyTorch installed

### Issue: CUDA out of memory
**Solution:** Reduce `--batch-size` to 1, or use smaller backbone (`mit_b0`)

### Issue: Can't find data files
**Solution:** Verify paths to `--data` and `--labels` are correct

### Issue: Training loss not decreasing
**Solution:**
- Check data loading is working (print first batch)
- Verify targets format is correct
- Try lower learning rate (`--lr 0.001`)

## Next Steps (After Successful CSSA Run)

1. ✅ **CSSA works** → Implement GAFF fusion
2. ✅ **CSSA works** → Test CSSA at multiple stages (1,2,3,4 combinations)
3. ✅ **CSSA works** → Implement DetGate (detection-driven fusion)
4. ✅ **CSSA works** → Implement ProbEn (detection-level ensembling)
5. ✅ **All methods work** → Run full ablation grid

## Citation

If this CSSA implementation is used in publications:

```bibtex
@inproceedings{cao2023cssa,
  title={Multimodal Object Detection by Channel Switching and Spatial Attention},
  author={Cao, Yue and Bin, Junchi and Hamari, Jozsef and Blasch, Erik and Liu, Zheng},
  booktitle={CVPR Workshops},
  year={2023}
}
```

## Contact

For issues with this ablation code, check:
1. Original CSSA repo: https://github.com/artrela/mulitmodal-cssa
2. Your baseline training script: `scripts/train.py`
3. Your baseline config: `crossattentiondet/config.py`
