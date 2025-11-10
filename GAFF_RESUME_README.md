# GAFF Ablation Resume Scripts

## Overview

The GAFF ablation study was interrupted at experiment 010. This creates 3 scripts to resume and complete the remaining 20 experiments across two computers in parallel.

## Files Created

1. **train_gaff_ablation_resume.py** - Enhanced training script with checkpoint resume capability
2. **resume_gaff_part1.py** - Resume script for Computer 1 (experiments 010-019)
3. **resume_gaff_part2.py** - Resume script for Computer 2 (experiments 020-029)

## Experiment Status

### Completed (9 experiments)
- exp_001 through exp_009 ✓

### Part 1 - Computer 1 (10 experiments: 010-019)
- **exp_010**: Resume from epoch 11 checkpoint (s4, inter_shared=True)
- **exp_011-015**: s4 stage config variants (5 fresh experiments)
- **exp_016-019**: s3 stage config variants (first 4 of 7)

### Part 2 - Computer 2 (10 experiments: 020-029)
- **exp_020-022**: s3 stage config variants (last 3 of 7)
- **exp_023-029**: s234 stage config variants (all 7)

## Usage

### On Computer 1
```bash
cd /mmfs1/project/pva23/csi3/cmx-object-detection
python resume_gaff_part1.py
```

### On Computer 2
```bash
cd /mmfs1/project/pva23/csi3/cmx-object-detection
python resume_gaff_part2.py
```

## What Each Script Does

### Part 1 Script
- Automatically detects that exp_010 has a checkpoint and resumes from epoch 11
- Runs experiments 011-019 from scratch
- Logs to `results/gaff_ablations_full/master_log_part1.txt`
- Updates shared `summary_all_experiments.csv`

### Part 2 Script
- Runs experiments 020-029 from scratch
- Logs to `results/gaff_ablations_full/master_log_part2.txt`
- Updates shared `summary_all_experiments.csv`

## Important Notes

1. **Both computers must have access to the same shared filesystem** for the CSV updates to merge properly
2. Each part takes approximately **40-50 hours** to complete (10 experiments × 4 hours each)
3. The scripts can be run simultaneously on different computers
4. Progress is logged independently but results are merged into the same CSV

## Experiment Configuration

All experiments use:
- Backbone: mit_b1
- Epochs: 15
- Batch size: 16
- Learning rate: 0.02
- Data: `../RGBX_Semantic_Segmentation/data/images`
- Labels: `../RGBX_Semantic_Segmentation/data/labels`

## Output Structure

```
results/gaff_ablations_full/
├── master_log.txt                    # Original master log (stopped at exp_010)
├── master_log_part1.txt              # Computer 1 log
├── master_log_part2.txt              # Computer 2 log
├── summary_all_experiments.csv       # Shared results CSV (updated by both)
├── phase1_stage_selection/
│   └── exp_001-008/                  # Completed
└── phase2_hyperparameter_tuning/
    ├── exp_009/                      # Completed
    ├── exp_010/                      # Will be resumed (has checkpoint)
    └── exp_011-029/                  # Will be created
```

## Monitoring Progress

### Check Part 1 Progress
```bash
tail -f results/gaff_ablations_full/master_log_part1.txt
```

### Check Part 2 Progress
```bash
tail -f results/gaff_ablations_full/master_log_part2.txt
```

### Check Overall Results
```bash
cat results/gaff_ablations_full/summary_all_experiments.csv
```

## Resume Capability Details

The enhanced training script (`train_gaff_ablation_resume.py`) supports:
- Resuming from saved checkpoints (`checkpoint.pth`)
- Continuing from the last completed epoch
- Appending to existing log files
- Preserving training metrics across resume

For exp_010 specifically:
- It completed 11 out of 15 epochs
- Checkpoint saved at epoch 10 (last multiple of 5)
- Will resume and complete epochs 12-15
- Total additional time: ~1.3 hours

## Troubleshooting

### If a script crashes mid-experiment
The training script saves checkpoints every 5 epochs. You can manually resume by:
```bash
python train_gaff_ablation_resume.py \
    --data ../RGBX_Semantic_Segmentation/data/images \
    --labels ../RGBX_Semantic_Segmentation/data/labels \
    --output-dir results/gaff_ablations_full/phase2_hyperparameter_tuning/exp_XXX \
    --backbone mit_b1 \
    --epochs 15 \
    --batch-size 16 \
    --lr 0.02 \
    --gaff-stages <stages> \
    --gaff-se-reduction <value> \
    --gaff-inter-shared <true/false> \
    --gaff-merge-bottleneck <true/false> \
    --resume-from-checkpoint true
```

### CSV Conflicts
If both computers update the CSV simultaneously, you may need to manually merge. Each part logs independently, so you can reconstruct from the part logs if needed.

## Expected Completion Time

- Part 1: ~40-50 hours (10 experiments)
- Part 2: ~40-50 hours (10 experiments)
- **Total if run in parallel: ~40-50 hours**
- **Total if run sequentially: ~80-100 hours**

## After Completion

Once both parts complete, you can analyze all results in:
```bash
results/gaff_ablations_full/summary_all_experiments.csv
```

The CSV will contain all 29 experiments with metrics including:
- mAP, mAP@50, mAP@75
- Training loss
- Model parameters
- Training time
