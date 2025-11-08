# GAFF (Guided Attentive Feature Fusion) Ablation Study

Quick start guide for implementing and testing GAFF fusion in multispectral object detection.

## Overview

GAFF (Guided Attentive Feature Fusion) is a fusion method from WACV 2021 that uses:
- **Intra-modality attention**: SE-style channel attention within each modality
- **Inter-modality attention**: Cross-attention between RGB and auxiliary modalities
- **Guided fusion**: Weighted combination with learned gates

**Paper**: "Guided Attentive Feature Fusion for Multispectral Pedestrian Detection"
**Authors**: Heng Zhang, Elisa Fromont, Sébastien Lefèvre, Bruno Avignon
**GitHub**: https://github.com/zhanghengdev/GAFF

## Quick Start (CPU Testing Workflow)

### Step 1: Module Verification (5-10 mins)

Test the GAFF fusion module:

```bash
# Unit tests
python -m crossattentiondet.ablations.fusion.test_gaff

# Detailed verification
python -m crossattentiondet.ablations.fusion.verify_gaff
```

Expected output: All tests pass, showing parameter counts and timing benchmarks.

### Step 2: Encoder Verification (10-15 mins)

Test the GAFF-enabled encoder:

```bash
python -m crossattentiondet.ablations.scripts.verify_gaff_encoder
```

Expected output: All backbones (mit_b0-b5) and stage configurations pass.

### Step 3: Dry-Run Tests (15-20 mins)

Test the complete training pipeline with synthetic data:

```bash
python -m crossattentiondet.ablations.scripts.dry_run_gaff
```

Expected output: Training loop works for all configurations.

### Step 4: Pre-GPU Checklist (30-40 mins)

Run the complete verification suite:

```bash
python -m crossattentiondet.ablations.scripts.pre_gpu_checklist
```

Expected output: All checks pass ✓

**If all tests pass, you're ready for GPU deployment!**

## GAFF Configuration Options

### Hyperparameters

| Parameter | Options | Description |
|-----------|---------|-------------|
| `gaff_stages` | `[1]`, `[2]`, `[3]`, `[4]`, `[2,3]`, etc. | Which stages use GAFF (others use FRM+FFM) |
| `gaff_se_reduction` | `4`, `8` | SE block reduction ratio |
| `gaff_inter_shared` | `True`, `False` | Use shared inter-modality conv |
| `gaff_merge_bottleneck` | `True`, `False` | Use bottleneck merge pathway |

### Example Usage

```python
from crossattentiondet.ablations.encoder_gaff_flexible import get_gaff_encoder

# GAFF at stage 4 only (default config)
encoder = get_gaff_encoder(
    backbone='mit_b1',
    gaff_stages=[4],
    gaff_se_reduction=4,
    gaff_inter_shared=False,
    gaff_merge_bottleneck=False
)

# GAFF at all stages with custom config
encoder = get_gaff_encoder(
    backbone='mit_b1',
    gaff_stages=[1, 2, 3, 4],
    gaff_se_reduction=8,
    gaff_inter_shared=True,
    gaff_merge_bottleneck=True
)
```

## GPU Training (After CPU Tests Pass)

### Single Experiment

```bash
python -m crossattentiondet.ablations.scripts.train_gaff_ablation \
    --backbone mit_b1 \
    --gaff-stages 4 \
    --gaff-se-reduction 4 \
    --gaff-inter-shared false \
    --gaff-merge-bottleneck false \
    --epochs 25 \
    --batch-size 8 \
    --lr 0.0001
```

### Full Ablation Suite

```bash
python -m crossattentiondet.ablations.scripts.run_gaff_ablations \
    --backbone mit_b1 \
    --epochs 25 \
    --batch-size 8
```

This will run all stage configurations and hyperparameter variants.

## File Structure

```
crossattentiondet/ablations/
├── fusion/
│   ├── gaff.py                    # GAFF fusion module
│   ├── test_gaff.py               # Unit tests
│   └── verify_gaff.py             # Module verification
│
├── encoder_gaff_flexible.py       # GAFF-enabled encoder
│
├── scripts/
│   ├── verify_gaff_encoder.py     # Encoder verification
│   ├── dry_run_gaff.py            # Dry-run testing
│   ├── pre_gpu_checklist.py       # Complete verification
│   ├── train_gaff_ablation.py     # Training script (TBD)
│   └── run_gaff_ablations.py      # Experiment runner (TBD)
│
└── gaff/
    ├── README.md                  # This file
    └── ABLATION_GUIDE.md          # Comprehensive guide
```

## Troubleshooting

### All tests failing
- Check Python path: Ensure `crossattentiondet` is in your path
- Check dependencies: Run `pip install -r requirements.txt`

### Import errors
- Verify file structure matches above
- Check that `__init__.py` files exist

### Memory errors on CPU
- Reduce batch size in dry-run tests
- Use smaller backbone (mit_b0 instead of mit_b1)

### Slow CPU tests
- This is expected! CPU is 10-100x slower than GPU
- Tests should still complete in < 1 hour total

## Next Steps

After passing all CPU tests:

1. **Transfer to GPU system**
2. **Set up dataset** (KAIST, MFNet, etc.)
3. **Run single training test** (5 epochs)
4. **Run full ablation suite** (25 epochs per experiment)
5. **Analyze results**

See `ABLATION_GUIDE.md` for comprehensive documentation.

## Support

For issues or questions:
- Check `ABLATION_GUIDE.md` for detailed documentation
- Review test output for specific error messages
- Compare with working CSSA implementation in `ablations/fusion/cssa.py`
