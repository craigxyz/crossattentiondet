# Implementation Details

**Code-Level Implementation with Snippets and Technical Specifications**

[← Back to Index](00_INDEX.md) | [← Previous: Training](05_TRAINING_AND_HYPERPARAMETERS.md) | [Next: Experimental Results →](07_EXPERIMENTAL_RESULTS.md)

---

## Key Implementation Files

### Fusion Mechanisms

**CSSA Implementation:**
- File: `crossattentiondet/ablations/fusion/cssa.py` (173 lines)
- Components: ECABlock, ChannelSwitching, SpatialAttention, CSSABlock
- Parameters: ~3,298 per stage (C=320)

**GAFF Implementation:**
- File: `crossattentiondet/ablations/fusion/gaff.py` (249 lines)
- Components: SEBlock, InterModalityAttention, GAFFBlock
- Parameters: ~717,440 per stage (C=320, r=4)

### Stage-Wise Integration

**CSSA Flexible Encoder:**
```python
# crossattentiondet/ablations/encoder_cssa_flexible.py
class RGBXTransformerCSSAFlexible(nn.Module):
    def __init__(self, ..., cssa_stages=[3], cssa_threshold=0.5):
        self.cssa_stages = cssa_stages
        
        # Create both baseline and CSSA fusion modules
        self.FRMs = nn.ModuleList([...])  # Baseline
        self.FFMs = nn.ModuleList([...])
        self.CSSAs = nn.ModuleList([CSSABlock(...) for _ in range(4)])
    
    def _fuse_stage(self, x_rgb, x_x, stage_idx):
        if (stage_idx + 1) in self.cssa_stages:
            return self.CSSAs[stage_idx](x_rgb, x_x)
        else:
            x_rgb, x_x = self.FRMs[stage_idx](x_rgb, x_x)
            return self.FFMs[stage_idx](x_rgb, x_x)
```

**GAFF Flexible Encoder:**
```python
# Similar structure for GAFF
class RGBXTransformerGAFFFlexible(nn.Module):
    def __init__(self, ..., gaff_stages=[3], gaff_se_reduction=4, ...):
        # Pluggable GAFF at specified stages
```

---

## Parameter Calculations

### CSSA (C=320, k=5)
- ECA_RGB: 320 × 5 = 1,600
- ECA_Aux: 320 × 5 = 1,600  
- ChannelSwitching: 0
- SpatialAttention: 2 × 7 × 7 = 98
- **Total: 3,298 params**

### GAFF (C=320, r=4, default config)
- SE_RGB: 2 × 320² / 4 = 51,200
- SE_Aux: 51,200
- InterModality: 4 × 320² = 409,600
- Merge: 2 × 320² = 205,440
- **Total: 717,440 params**

**Ratio: GAFF/CSSA = 217×** (for same channel count)

---

## Training Scripts

### Single Experiment Scripts

**CSSA Ablation:**
```bash
python crossattentiondet/ablations/scripts/train_cssa_ablation.py \
    --data ../RGBX_Semantic_Segmentation/data/images \
    --labels ../RGBX_Semantic_Segmentation/data/labels \
    --output-dir results/cssa_test \
    --cssa-stages "3" \
    --cssa-threshold 0.5 \
    --epochs 25 \
    --batch-size 2
```

**GAFF Ablation:**
```bash
python crossattentiondet/ablations/scripts/train_gaff_ablation.py \
    --data ... \
    --gaff-stages "3" \
    --gaff-se-reduction 4 \
    --gaff-inter-shared false \
    --gaff-merge-bottleneck false \
    --epochs 25 \
    --batch-size 8
```

### Master Runners

**Run All CSSA Experiments:**
```bash
python crossattentiondet/ablations/scripts/run_cssa_ablations.py \
    --data ... \
    --output-dir results/cssa_ablations \
    --epochs 25
```

**Run All GAFF Experiments:**
```bash
python crossattentiondet/ablations/scripts/run_gaff_ablations.py \
    --data ... \
    --output-base results/gaff_ablations_full \
    --epochs 25
```

---

For complete code details, see:
- Architecture Deep Dive (section 02)
- Source files in `crossattentiondet/`

[← Back to Index](00_INDEX.md) | [← Previous: Training](05_TRAINING_AND_HYPERPARAMETERS.md) | [Next: Experimental Results →](07_EXPERIMENTAL_RESULTS.md)
