# Technical Specifications Reference

**Repository Structure and Reproducibility**

[← Back to Index](00_INDEX.md) | [← Previous: CVPR Paper Guidance](08_CVPR_PAPER_GUIDANCE.md)

---

## Repository Structure

```
cmx-object-detection/
├── crossattentiondet/           # Main package
│   ├── models/                  # Architecture
│   │   ├── backbone.py         # FPN wrapper
│   │   ├── encoder.py          # RGBXTransformer (267 lines)
│   │   ├── fusion.py           # Baseline FRM+FFM (229 lines)
│   │   └── transformer.py      # MiT components
│   ├── data/                   # Data loading
│   │   ├── dataset.py          # NpyYoloDataset (126 lines)
│   │   └── transforms.py       # YOLO→COCO conversion
│   ├── training/               # Training logic
│   │   ├── trainer.py          # Trainer class (151 lines)
│   │   └── evaluator.py        # COCO evaluation
│   ├── ablations/              # Ablation infrastructure
│   │   ├── fusion/             # Fusion modules
│   │   │   ├── cssa.py        # CSSA (173 lines)
│   │   │   ├── gaff.py        # GAFF (249 lines)
│   │   │   ├── test_cssa.py   # Unit tests
│   │   │   └── verify_gaff.py # Verification
│   │   ├── encoder_cssa_flexible.py  # CSSA encoder (303 lines)
│   │   ├── encoder_gaff_flexible.py  # GAFF encoder (318 lines)
│   │   ├── backbone_modality.py      # Modality ablations
│   │   └── scripts/            # Training scripts
│   │       ├── run_cssa_ablations.py
│   │       ├── run_gaff_ablations.py
│   │       └── train_*_ablation.py
│   ├── config.py               # Configuration
│   └── utils/                  # Utilities
│
├── scripts/                    # Entry points
│   ├── train.py               # Baseline training
│   ├── test.py                # Evaluation
│   └── train_all_backbones.py
│
├── results/                   # Experiment results
│   ├── cssa_ablations/        # 11 CSSA experiments
│   └── gaff_ablations_full/   # 32 GAFF experiments
│
├── docs/                      # Documentation (774+ lines each)
│   ├── EXECUTIVE_SUMMARY.md
│   ├── FUSION_MECHANISMS_COMPARISON.md
│   ├── GAFF_ABLATION_GUIDE.md
│   └── CSSA_ABLATION_GUIDE.md
│
├── cvpr_docs/                 # CVPR paper documentation
│   ├── 00_INDEX.md
│   ├── 01_EXECUTIVE_SUMMARY.md
│   ├── 02_ARCHITECTURE_DEEP_DIVE.md
│   ├── 03_DATASET_AND_MODALITIES.md
│   ├── 04_ABLATION_STUDIES.md
│   ├── 05_TRAINING_AND_HYPERPARAMETERS.md
│   ├── 06_IMPLEMENTATION_DETAILS.md
│   ├── 07_EXPERIMENTAL_RESULTS.md
│   ├── 08_CVPR_PAPER_GUIDANCE.md
│   └── 09_TECHNICAL_SPECIFICATIONS.md (this file)
│
└── README.md
```

---

## Key Files by Function

**Architecture:**
- `models/encoder.py`: RGBXTransformer with MiT variants
- `models/fusion.py`: Baseline FRM+FFM modules
- `models/backbone.py`: FPN integration

**Fusion Mechanisms:**
- `ablations/fusion/cssa.py`: CSSA implementation
- `ablations/fusion/gaff.py`: GAFF implementation
- `ablations/encoder_*_flexible.py`: Stage-wise integration

**Training:**
- `training/trainer.py`: Training loop
- `data/dataset.py`: Dataset class
- `config.py`: Hyperparameters

**Ablation Framework:**
- `ablations/scripts/run_*_ablations.py`: Master runners
- `ablations/scripts/train_*_ablation.py`: Single experiment

---

## Reproducibility Checklist

### 1. Environment Setup
```bash
git clone <repo>
cd cmx-object-detection
pip install -r requirements.txt
```

### 2. Data Preparation
- Ensure `../RGBX_Semantic_Segmentation/data/images/` contains .npy files
- Ensure `../RGBX_Semantic_Segmentation/data/labels/` contains .txt files

### 3. Run Baseline
```bash
python scripts/train.py --data ../RGBX_Semantic_Segmentation/data \
                        --backbone mit_b1 --epochs 15 --batch-size 2
```

### 4. Run CSSA Ablations
```bash
python crossattentiondet/ablations/scripts/run_cssa_ablations.py \
    --data ../RGBX_Semantic_Segmentation/data/images \
    --labels ../RGBX_Semantic_Segmentation/data/labels \
    --output-dir results/cssa_ablations --epochs 25
```

### 5. Run GAFF Ablations
```bash
python crossattentiondet/ablations/scripts/run_gaff_ablations.py \
    --data ../RGBX_Semantic_Segmentation/data/images \
    --labels ../RGBX_Semantic_Segmentation/data/labels \
    --output-base results/gaff_ablations_full --epochs 25
```

### 6. Check Results
```bash
# CSSA
cat results/cssa_ablations/exp_*/final_results.json

# GAFF  
cat results/gaff_ablations_full/phase1_stage_selection/exp_*/final_results.json
```

---

## Git History

**Recent Commits:**
```
1eaf843 - Optimize ablation training for A100 GPU
f98a962 - Add CSSA ablation system with comprehensive documentation
cecc343 - Add CSSA fusion ablation infrastructure
ab23a42 - Add batch testing script for all backbone variants
54abb1c - Initial commit: CrossAttentionDet framework
```

**Development Timeline:**
1. Initial framework (baseline FRM+FFM)
2. Backbone testing infrastructure
3. CSSA integration
4. GAFF integration
5. A100 optimization
6. Modality ablations

---

## Hardware Requirements

**GPU:** NVIDIA A100 (79.25 GiB) or equivalent
- mit_b0-b2: Works on most GPUs (12+ GB)
- mit_b4-b5: Requires optimization (gradient checkpointing, FP16)

**Storage:** ~500 GB for all experiments

**Compute Time:** ~6 days continuous A100 time for all 48 experiments

---

## Configuration Examples

**CSSA Experiment Config:**
```json
{
  "experiment_id": "exp_003_s3_t0.5",
  "cssa_stages": [3],
  "cssa_threshold": 0.5,
  "backbone": "mit_b1",
  "epochs": 25,
  "batch_size": 2,
  "learning_rate": 0.005
}
```

**GAFF Experiment Config:**
```json
{
  "experiment_id": "exp_003_s3_r4_is0_mb0",
  "gaff_stages": [3],
  "gaff_se_reduction": 4,
  "gaff_inter_shared": false,
  "gaff_merge_bottleneck": false,
  "backbone": "mit_b1",
  "epochs": 15,
  "batch_size": 8,
  "learning_rate": 0.02
}
```

---

[← Back to Index](00_INDEX.md) | [← Previous: CVPR Paper Guidance](08_CVPR_PAPER_GUIDANCE.md)

**Documentation suite complete! Ready for CVPR paper writing. 🚀**
