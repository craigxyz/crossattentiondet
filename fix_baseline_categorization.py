#!/usr/bin/env python3
"""Fix the categorization of baseline vs ablation experiments."""

with open("COMPREHENSIVE_EXPERIMENTAL_RESULTS.md", "r") as f:
    content = f.read()

# Replace the modality ablations section title and description
old_section = """## 4. Modality Ablation Study

**Objective:** Evaluate model performance with different modality combinations"""

new_section = """## 4. Baseline Models (15 epochs) - Modality Combinations

**Architecture:** Standard CMX with FRM+FFM (baseline_FRM_FFM)

**Training:** 15 epochs, batch_size=16, lr=0.02, backbone=mit_b1

**Objective:** Establish baseline performance with standard fusion across different modality pairs

**⚠️ IMPORTANT:** These are BASELINE experiments, not ablations. They use the standard CMX architecture."""

content = content.replace(old_section, new_section)

# Update the best modality combination line
content = content.replace(
    "**Best Modality Combination:** RGB + THERMAL with mAP=0.8342",
    "**Best Baseline (15 epochs):** RGB + THERMAL with mAP=0.8342 (83.42%)\n\n**This is the primary baseline for comparison with GAFF and CSSA ablations.**"
)

# Update section 7 to add critical comparison
insights_section = """### Insights

#### GAFF Ablations:"""

new_insights = """### Critical Findings: Baseline vs Ablations

**Baseline Performance (15 epochs, RGB+Thermal):** 83.42% mAP

**Ablation Study Results:**
- Best GAFF: 83.78% mAP (+0.36% vs baseline)
- Best CSSA: 84.07% mAP (+0.65% vs baseline)
- Best CSSA fast: 83.44% mAP (+0.02% vs baseline)

**Key Insight:** The fusion ablation methods (GAFF, CSSA) provide **marginal improvements (~0.5%)** over the standard FRM+FFM baseline when trained under identical conditions (15 epochs, same hyperparameters).

### Insights

#### GAFF Ablations:"""

content = content.replace(insights_section, new_insights)

# Update the modality combinations insights
old_mod_insight = """#### Modality Combinations:
- Total combinations tested: 3
- mAP range: 0.6632 to 0.8342
- Average mAP: 0.7487"""

new_mod_insight = """#### Baseline Modality Combinations:
- Total combinations tested: 3
- mAP range: 0.6632 to 0.8342
- Average mAP: 0.7487
- **Best baseline: RGB+Thermal at 83.42% mAP**

**Modality Importance:**
- RGB alone: Not tested separately
- RGB + Thermal: 83.42% mAP (best combination)
- RGB + Event: 66.32% mAP (poor performance)
- Thermal + Event: 74.86% mAP (moderate)
- **Conclusion:** RGB+Thermal is the optimal modality pair"""

content = content.replace(old_mod_insight, new_mod_insight)

# Update section 6 title
content = content.replace(
    "## 6. Baseline Models (CMX with FRM+FFM)",
    "## 6. Baseline Models (1 epoch) - Initial Validation"
)

# Update the note in section 6
old_note = """**Important Note:** These baseline models were trained for only 1 epoch for initial validation.
For fair comparison with ablation studies (15 epochs), these would need to be retrained."""

new_note = """**Important Note:** These baseline models were trained for only 1 epoch for initial validation.
For fair comparison, see Section 4 which contains the 15-epoch baseline results."""

content = content.replace(old_note, new_note)

# Add comparison note to section 6
perf_comparison = """### Performance vs Ablation Studies

**⚠️ Training Epoch Mismatch:**
- Baseline models: **1 epoch** → mAP ~0.66 (66%)
- GAFF ablations: **15 epochs** → mAP ~0.84 (84%)
- CSSA ablations: **15 epochs** → mAP ~0.84 (84%)
- Modality ablations: **15 epochs** → mAP ~0.83 (83%)

**Fair comparison requires:** Retraining baseline models with 15 epochs to match ablation study protocols."""

new_perf_comparison = """### Performance vs Ablation Studies

**⚠️ See Section 4 for fair comparison:** The 15-epoch baseline (RGB+Thermal) achieves 83.42% mAP.

**1-epoch results (this section):**
- Best 1-epoch baseline: **66.24% mAP** (mit_b1)

**15-epoch results (Section 4):**
- Best 15-epoch baseline: **83.42% mAP** (RGB+Thermal)
- Best GAFF ablation: **83.78% mAP** (+0.36%)
- Best CSSA ablation: **84.07% mAP** (+0.65%)"""

content = content.replace(perf_comparison, new_perf_comparison)

with open("COMPREHENSIVE_EXPERIMENTAL_RESULTS.md", "w") as f:
    f.write(content)

print("✓ Fixed baseline categorization")
print("  - Renamed 'Modality Ablations' to 'Baseline Models (15 epochs)'")
print("  - Added critical comparison: ablations only +0.5% vs baseline")
print("  - Clarified that RGB+Thermal baseline (83.42%) is the primary comparison point")
