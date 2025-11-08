# CPU Testing Instructions for GAFF Implementation

## Prerequisites

Before running any tests, ensure you have PyTorch installed:

```bash
# CPU-only PyTorch (for testing)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Or GPU-enabled PyTorch (if available)
pip install torch torchvision
```

## Testing Workflow

Follow these steps in order to verify the GAFF implementation on CPU:

### Step 1: Module Unit Tests (5-10 minutes)

Test the core GAFF fusion module:

```bash
cd /mmfs1/project/pva23/csi3/cmx-object-detection
python -m crossattentiondet.ablations.fusion.test_gaff
```

**Expected output**:
- 30+ tests should all pass
- No NaN or Inf values
- Final success rate: 100%

**If tests fail**:
- Check error messages for specific failures
- Verify torch is installed correctly
- Ensure all dependencies are available

### Step 2: Module Verification (5-10 minutes)

Run comprehensive module verification:

```bash
python -m crossattentiondet.ablations.fusion.verify_gaff
```

**Expected output**:
- Module structure analysis
- All hyperparameter combinations pass
- CPU timing benchmarks
- Memory usage profiling

### Step 3: Encoder Verification (10-15 minutes)

Test the GAFF-enabled encoder:

```bash
python -m crossattentiondet.ablations.scripts.verify_gaff_encoder
```

**Expected output**:
- All backbones (mit_b0 through mit_b5) pass
- All stage configurations work
- Gradient flow verified
- Parameter counts displayed

### Step 4: Dry-Run Tests (15-20 minutes)

Test the complete training pipeline with synthetic data:

```bash
python -m crossattentiondet.ablations.scripts.dry_run_gaff
```

**Expected output**:
- Training loop completes for all configurations
- All stage configs pass
- Checkpoint save/load works
- Different batch sizes work

### Step 5: Complete Pre-GPU Checklist (30-40 minutes)

Run all tests automatically:

```bash
python -m crossattentiondet.ablations.scripts.pre_gpu_checklist
```

**Expected output**:
```
================================================================================
✓ ✓ ✓  ALL CHECKS PASSED  ✓ ✓ ✓
================================================================================

Your GAFF implementation is ready for GPU deployment!
```

## Troubleshooting

### ModuleNotFoundError: No module named 'torch'

**Solution**: Install PyTorch
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### ModuleNotFoundError: No module named 'crossattentiondet'

**Solution**: Run from project root or add to PYTHONPATH
```bash
cd /mmfs1/project/pva23/csi3/cmx-object-detection
export PYTHONPATH=/mmfs1/project/pva23/csi3/cmx-object-detection:$PYTHONPATH
```

### Tests are very slow on CPU

**Solution**: This is normal! CPU is 10-100x slower than GPU. Tests may take 30-60 minutes total. This is expected.

### Import errors for baseline modules

**Error**: `ModuleNotFoundError: No module named 'crossattentiondet.models'`

**Solution**: Ensure you have the full project structure. The baseline FRM/FFM modules should be in:
- `crossattentiondet/models/fusion.py`
- `crossattentiondet/models/transformer.py`

### Memory errors

**Solution**:
- Close other applications
- Reduce batch size in dry_run tests (edit the script)
- Use a smaller backbone (mit_b0 instead of mit_b1)

## Quick Verification (If Time-Limited)

If you don't have time for the full test suite, run this minimal test:

```bash
# Quick test (1-2 minutes)
python -c "
import sys
sys.path.insert(0, '/mmfs1/project/pva23/csi3/cmx-object-detection')

import torch
from crossattentiondet.ablations.fusion.gaff import GAFFBlock

print('Creating GAFF block...')
gaff = GAFFBlock(in_channels=64, out_channels=64)
print('✓ GAFF block created')

print('Testing forward pass...')
x_rgb = torch.randn(2, 64, 16, 16)
x_aux = torch.randn(2, 64, 16, 16)
out = gaff(x_rgb, x_aux)
print(f'✓ Forward pass successful: {out.shape}')

print('Testing backward pass...')
loss = out.sum()
loss.backward()
print('✓ Backward pass successful')

print()
print('='*50)
print('✓ Basic GAFF functionality verified!')
print('='*50)
"
```

## After CPU Tests Pass

Once all CPU tests pass successfully:

1. ✓ Code is verified and error-free
2. ✓ Ready to transfer to GPU system
3. → Set up dataset (KAIST, MFNet, etc.)
4. → Run actual training experiments
5. → Analyze results

## Files Created

All GAFF implementation files are in:

```
/mmfs1/project/pva23/csi3/cmx-object-detection/crossattentiondet/ablations/

Core Implementation:
├── fusion/gaff.py                      # GAFF module (~200 lines)
├── fusion/__init__.py                  # Updated with GAFF export
├── encoder_gaff_flexible.py            # Flexible encoder (~350 lines)

Testing Suite:
├── fusion/test_gaff.py                 # Unit tests (~150 lines)
├── fusion/verify_gaff.py               # Module verification (~200 lines)
├── scripts/verify_gaff_encoder.py      # Encoder tests (~350 lines)
├── scripts/dry_run_gaff.py             # Training tests (~400 lines)
├── scripts/pre_gpu_checklist.py        # Master checklist (~150 lines)

Documentation:
├── gaff/README.md                      # Quick start guide
├── gaff/ABLATION_GUIDE.md              # Comprehensive guide
└── gaff/CPU_TESTING_INSTRUCTIONS.md    # This file
```

## Estimated Testing Time

| Test | Duration | Cumulative |
|------|----------|------------|
| Module unit tests | 5-10 min | 5-10 min |
| Module verification | 5-10 min | 10-20 min |
| Encoder verification | 10-15 min | 20-35 min |
| Dry-run tests | 15-20 min | 35-55 min |
| **Total (if run individually)** | **35-55 min** | - |
| **Total (pre-GPU checklist)** | **30-45 min** | - |

The pre-GPU checklist runs all tests automatically, which is faster than running each individually.

## Next Steps After Verification

See `README.md` and `ABLATION_GUIDE.md` for:
- GPU training instructions
- Experiment design details
- Results analysis procedures
- Full documentation

Good luck with testing!
