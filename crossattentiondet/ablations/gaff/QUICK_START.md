# GAFF Quick Start Guide

## CPU Verification (30-45 minutes)

Run these commands in order to verify the implementation:

```bash
# Navigate to project root
cd /mmfs1/project/pva23/csi3/cmx-object-detection

# Test 1: Module unit tests (5-10 min)
python -m crossattentiondet.ablations.fusion.test_gaff

# Test 2: Module verification (5-10 min)
python -m crossattentiondet.ablations.fusion.verify_gaff

# Test 3: Encoder verification (10-15 min)
python -m crossattentiondet.ablations.scripts.verify_gaff_encoder

# Test 4: Dry-run training tests (15-20 min)
python -m crossattentiondet.ablations.scripts.dry_run_gaff

# Or run all tests at once (30-45 min)
python -m crossattentiondet.ablations.scripts.pre_gpu_checklist
```

**Expected**: All tests should pass ✓

## GPU Training (When Ready)

### Prerequisites
- ✅ All CPU tests passed
- ✅ Dataset downloaded and prepared (KAIST, MFNet, etc.)
- ✅ GPU environment with PyTorch

### Create Training Scripts

You'll need to adapt the CSSA training scripts:

```bash
# Copy CSSA training scripts as templates
cp ablations/scripts/train_cssa_ablation.py ablations/scripts/train_gaff_ablation.py
cp ablations/scripts/run_cssa_ablations.py ablations/scripts/run_gaff_ablations.py

# Edit the files to:
# 1. Change imports from encoder_cssa_flexible to encoder_gaff_flexible
# 2. Update CLI arguments (--gaff-se-reduction, --gaff-inter-shared, --gaff-merge-bottleneck)
# 3. Update config logic to pass GAFF parameters
```

### Single Experiment

```bash
python -m crossattentiondet.ablations.scripts.train_gaff_ablation \
    --data-root /path/to/dataset \
    --backbone mit_b1 \
    --gaff-stages 4 \
    --gaff-se-reduction 4 \
    --gaff-inter-shared false \
    --gaff-merge-bottleneck false \
    --epochs 25 \
    --batch-size 8 \
    --device cuda
```

### Full Ablation Suite

```bash
python -m crossattentiondet.ablations.scripts.run_gaff_ablations \
    --data-root /path/to/dataset \
    --backbone mit_b1 \
    --epochs 25 \
    --batch-size 8 \
    --output-dir ./results/gaff_ablations
```

## File Structure

```
crossattentiondet/ablations/
├── fusion/
│   ├── gaff.py              ← GAFF module implementation
│   ├── test_gaff.py         ← Unit tests
│   └── verify_gaff.py       ← Module verification
│
├── encoder_gaff_flexible.py ← GAFF encoder
│
├── scripts/
│   ├── verify_gaff_encoder.py    ← Encoder tests
│   ├── dry_run_gaff.py           ← Training loop tests
│   ├── pre_gpu_checklist.py      ← Master verification
│   ├── train_gaff_ablation.py    ← TODO: Create from CSSA
│   └── run_gaff_ablations.py     ← TODO: Create from CSSA
│
└── gaff/
    ├── QUICK_START.md         ← This file
    ├── README.md              ← Full quick start
    ├── ABLATION_GUIDE.md      ← Comprehensive guide
    ├── CPU_TESTING_INSTRUCTIONS.md
    └── IMPLEMENTATION_SUMMARY.md
```

## Configuration Cheat Sheet

### Stage Selection
```python
gaff_stages=[4]        # GAFF at stage 4 only (recommended starting point)
gaff_stages=[3,4]      # GAFF at stages 3 and 4
gaff_stages=[1,2,3,4]  # GAFF at all stages
```

### Hyperparameters
```python
gaff_se_reduction=4           # SE bottleneck ratio (4 or 8)
gaff_inter_shared=False       # Separate inter-modality convs (True for shared)
gaff_merge_bottleneck=False   # Direct merge (True for bottleneck)
```

### Example Configurations

**Default (recommended starting point)**:
```python
gaff_stages=[4]
gaff_se_reduction=4
gaff_inter_shared=False
gaff_merge_bottleneck=False
```

**All stages**:
```python
gaff_stages=[1, 2, 3, 4]
gaff_se_reduction=4
gaff_inter_shared=False
gaff_merge_bottleneck=False
```

**Memory-efficient**:
```python
gaff_stages=[4]
gaff_se_reduction=8  # Larger reduction = fewer params
gaff_inter_shared=True
gaff_merge_bottleneck=False
```

## Quick Python Test

Test GAFF import and basic functionality:

```python
import torch
from crossattentiondet.ablations.fusion.gaff import GAFFBlock
from crossattentiondet.ablations.encoder_gaff_flexible import get_gaff_encoder

# Test module
gaff = GAFFBlock(in_channels=64, out_channels=64)
x_rgb = torch.randn(2, 64, 16, 16)
x_aux = torch.randn(2, 64, 16, 16)
out = gaff(x_rgb, x_aux)
print(f"GAFF output shape: {out.shape}")  # Should be (2, 64, 16, 16)

# Test encoder
encoder = get_gaff_encoder(backbone='mit_b1', gaff_stages=[4])
rgb = torch.randn(1, 3, 224, 224)
aux = torch.randn(1, 1, 224, 224)
features = encoder(rgb, aux)
print(f"Encoder outputs: {len(features)} stages")  # Should be 4
print(f"Stage shapes: {[f.shape for f in features]}")
```

## Troubleshooting

### Import errors
```bash
export PYTHONPATH=/mmfs1/project/pva23/csi3/cmx-object-detection:$PYTHONPATH
```

### No torch
```bash
pip install torch torchvision
```

### Tests slow
- Normal on CPU! Expect 30-60 minutes total
- GPU will be 10-100x faster

## Next Steps

1. ✅ Run CPU verification tests
2. → Transfer to GPU system
3. → Set up dataset
4. → Create training scripts (adapt from CSSA)
5. → Run pilot experiment (5 epochs)
6. → Run full ablation (32 experiments)
7. → Analyze results

## Documentation

- **Quick overview**: `README.md`
- **Comprehensive guide**: `ABLATION_GUIDE.md`
- **CPU testing**: `CPU_TESTING_INSTRUCTIONS.md`
- **Implementation details**: `IMPLEMENTATION_SUMMARY.md`
- **This file**: Quick commands and configs

Good luck with your experiments!
