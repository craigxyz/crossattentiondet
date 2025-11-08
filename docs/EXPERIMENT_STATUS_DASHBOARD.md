# Experiment Status Dashboard

**Project:** CrossAttentionDet Multi-Modal Object Detection
**Last Updated:** 2025-11-07
**Overall Progress:** 9/48 experiments complete (18.75%)

---

## Executive Summary

This dashboard tracks the progress of all experimental work in the CrossAttentionDet project, including baseline training, CSSA ablations, and GAFF ablations.

**Current Status:**
- **Total Configured Experiments:** 48
- **Completed:** 9 (18.75%)
- **In Progress:** 1 (2.08%)
- **Failed:** 2 (4.17%)
- **Pending:** 36 (75.00%)

**Resource Usage:**
- **Total GPU Hours:** ~6.1 hours (completed experiments only)
- **Estimated Remaining:** ~200-250 hours (for pending 36 experiments)
- **Total Data Processed:** ~15 training runs

---

## 1. Overall Progress Tracker

### 1.1 Experiment Categories

```
CrossAttentionDet Experiments (48 total)
│
├── [40%] Baseline Backbones (2/5 complete)
│   ├── ✅ mit_b0
│   ├── ✅ mit_b1
│   ├── 🔄 mit_b2 (in progress)
│   ├── ❌ mit_b4 (OOM)
│   └── ❌ mit_b5 (OOM)
│
├── [27%] CSSA Ablations (3/11 complete)
│   ├── [43%] Phase 1: Stage Selection (3/7)
│   └── [0%] Phase 2: Threshold Tuning (0/4)
│
└── [13%] GAFF Ablations (4/32 complete)
    ├── [50%] Phase 1: Stage Selection (4/8)
    └── [0%] Phase 2: Hyperparameter Tuning (0/24)
```

### 1.2 Progress by Status

| Status | Count | Percentage | Description |
|--------|-------|------------|-------------|
| ✅ Complete | 9 | 18.75% | Training finished, results available |
| 🔄 In Progress | 1 | 2.08% | Currently training |
| ❌ Failed | 2 | 4.17% | OOM or other errors |
| ⏸️ Pending | 36 | 75.00% | Not yet started |
| **Total** | **48** | **100%** | All configured experiments |

---

## 2. Baseline Training Status

### 2.1 Backbone Variants Progress

**Goal:** Train 5 backbone variants (mit_b0 through mit_b5) without fusion to establish baseline performance.

| Backbone | Params (M) | Status | Epochs | Best Loss | Time (hrs) | Notes |
|----------|------------|--------|--------|-----------|------------|-------|
| mit_b0 | 55.7 | ✅ Complete | 15/15 | 0.1057 | 2.68 | Success, good baseline |
| mit_b1 | 69.5 | ✅ Complete | 15/15 | 0.1027 | 3.44 | Success, lower loss than b0 |
| mit_b2 | 82.1 | 🔄 Training | 1/15 | 0.3787 | - | Restarted w/ grad accum, batch=4 |
| mit_b4 | 155.4 | ❌ Failed | 0/15 | - | - | CUDA OOM: 79.25 GiB exhausted |
| mit_b5 | 196.6 | ❌ Failed | 0/15 | - | - | CUDA OOM: 79.25 GiB exhausted |

**Training Configuration (Original):**
```python
batch_size = 16
learning_rate = 0.02
epochs = 15
optimizer = SGD(momentum=0.9, weight_decay=5e-4)
lr_scheduler = StepLR(step_size=15, gamma=0.1)
```

**Training Configuration (mit_b2 Restart - Memory Optimized):**
```python
batch_size = 4  # Reduced from 16
gradient_accumulation_steps = 4  # Effective batch size = 16
learning_rate = 0.02
epochs = 15
optimizer = SGD(momentum=0.9, weight_decay=5e-4)
```

### 2.2 Performance Trends

**Completed Baselines:**
| Metric | mit_b0 | mit_b1 | Trend |
|--------|--------|--------|-------|
| Parameters | 55.7M | 69.5M | +24.8% |
| Best Loss | 0.1057 | 0.1027 | -2.8% (improvement) |
| Time per Epoch | 10.7 min | 13.7 min | +28% |
| Total Time | 2.68 hrs | 3.44 hrs | +28% |

**Insights:**
- Larger backbone (mit_b1) achieves lower loss but requires more time
- Training time scales roughly linearly with parameter count
- Loss improvement (~3%) is modest for 25% more parameters

### 2.3 Memory Issues & Solutions

**OOM Failures (mit_b4, mit_b5):**
```
Error: CUDA out of memory. Tried to allocate 79.25 GiB
Available memory: ~40 GB on A100
```

**Attempted Solutions:**
1. **Gradient Accumulation:** Reduce batch size, maintain effective batch size
   - Status: Applied to mit_b2, testing in progress
2. **Mixed Precision Training:** Use torch.cuda.amp for FP16
   - Status: Not yet implemented
3. **Gradient Checkpointing:** Trade compute for memory
   - Status: Available but not enabled in current runs

**Next Steps for mit_b4, mit_b5:**
- Reduce batch size to 2 with gradient accumulation of 8
- Enable mixed precision training (FP16)
- Consider gradient checkpointing
- If still OOM: May need multi-GPU training or smaller input resolution

---

## 3. CSSA Ablation Status

### 3.1 Phase 1: Stage Selection (3/7 complete - 43%)

**Objective:** Identify which encoder stages benefit most from CSSA fusion

**Fixed Parameters:** threshold=0.5, kernel_size=3
**Variable:** Stage configurations

| Exp ID | Code | Stages | Status | Best mAP | Best mAP@50 | Time (hrs) | Result Location |
|--------|------|--------|--------|----------|-------------|------------|-----------------|
| exp_001 | s1_t0.5 | [1] | ✅ Complete | TBD | TBD | ~3.5 | results/cssa_ablations/exp_001_*/ |
| exp_002 | s2_t0.5 | [2] | ✅ Complete | TBD | TBD | ~3.5 | results/cssa_ablations/exp_002_*/ |
| exp_003 | s3_t0.5 | [3] | ✅ Complete | TBD | TBD | ~3.5 | results/cssa_ablations/exp_003_*/ |
| exp_004 | s4_t0.5 | [4] | ⏸️ Pending | - | - | - | - |
| exp_005 | s23_t0.5 | [2,3] | ⏸️ Pending | - | - | - | - |
| exp_006 | s34_t0.5 | [3,4] | ⏸️ Pending | - | - | - | - |
| exp_007 | s1234_t0.5 | [1,2,3,4] | ⏸️ Pending | - | - | - | - |

**Progress:** 3/7 complete (42.9%)

**Estimated Time to Complete Phase 1:**
- Remaining: 4 experiments
- Time per experiment: ~3.5 hours
- Total: ~14 hours

### 3.2 Phase 2: Threshold Tuning (0/4 complete - 0%)

**Objective:** Fine-tune channel switching threshold on top 2 stage configurations

**Status:** Blocked until Phase 1 completes

**Planned Experiments:**
- Will test thresholds {0.3, 0.7} on top 2 configs from Phase 1
- Expected to start after exp_007 completes
- Estimated time: 4 experiments × 3.5 hrs = 14 hours

### 3.3 CSSA Parameter Overhead

**Per-Stage CSSA Parameters (mit_b1):**
| Stage | Channels | CSSA Params | % of Backbone |
|-------|----------|-------------|---------------|
| 1 | 64 | ~580 | 0.0008% |
| 2 | 128 | ~860 | 0.0012% |
| 3 | 320 | ~2,019 | 0.0029% |
| 4 | 512 | ~3,170 | 0.0046% |

**Multi-Stage Overhead:**
- Stages [2,3]: ~2,879 params (+0.004%)
- Stages [1,2,3,4]: ~6,629 params (+0.010%)

**Key Insight:** CSSA adds negligible parameter overhead (<0.01% even for all stages)

---

## 4. GAFF Ablation Status

### 4.1 Phase 1: Stage Selection (4/8 complete - 50%)

**Objective:** Identify which encoder stages benefit most from GAFF fusion

**Fixed Parameters:** SE_reduction=4, inter_shared=False, merge_bottleneck=False
**Variable:** Stage configurations

| Exp ID | Code | Stages | Status | Best mAP | Best mAP@50 | Time (hrs) | Result Location |
|--------|------|--------|--------|----------|-------------|------------|-----------------|
| exp_001 | s1_r4_is0_mb0 | [1] | ✅ Complete | TBD | TBD | ~3.4 | results/gaff_ablations_full/phase1_*/exp_001_*/ |
| exp_002 | s2_r4_is0_mb0 | [2] | ✅ Complete | TBD | TBD | ~3.4 | results/gaff_ablations_full/phase1_*/exp_002_*/ |
| exp_003 | s3_r4_is0_mb0 | [3] | ✅ Complete | TBD | TBD | ~3.4 | results/gaff_ablations_full/phase1_*/exp_003_*/ |
| exp_004 | s4_r4_is0_mb0 | [4] | ✅ Complete | TBD | TBD | ~3.4 | results/gaff_ablations_full/phase1_*/exp_004_*/ |
| exp_005 | s23_r4_is0_mb0 | [2,3] | ⏸️ Pending | - | - | - | - |
| exp_006 | s34_r4_is0_mb0 | [3,4] | ⏸️ Pending | - | - | - | - |
| exp_007 | s234_r4_is0_mb0 | [2,3,4] | ⏸️ Pending | - | - | - | - |
| exp_008 | s1234_r4_is0_mb0 | [1,2,3,4] | ⏸️ Pending | - | - | - | - |

**Progress:** 4/8 complete (50%)

**Estimated Time to Complete Phase 1:**
- Remaining: 4 experiments
- Time per experiment: ~3.5 hours (may be longer for multi-stage)
- Total: ~14-18 hours

### 4.2 Phase 2: Hyperparameter Tuning (0/24 complete - 0%)

**Objective:** Test all hyperparameter combinations on top 3 stage configurations

**Status:** Blocked until Phase 1 completes

**Hyperparameter Grid:**
- SE_reduction: {4, 8}
- inter_shared: {False, True}
- merge_bottleneck: {False, True}
- Total combinations: 2 × 2 × 2 = 8 per stage config

**Planned Experiments:**
- Top 3 stage configs from Phase 1
- 8 combinations per config
- Minus 3 defaults (already tested in Phase 1)
- **Total: 3 × 7 = 21 new experiments**

**Estimated Time:**
- 21 experiments × 3.5-4 hrs = 73.5-84 hours (~3-3.5 days on single GPU)

### 4.3 GAFF Parameter Overhead

**Per-Stage GAFF Parameters (mit_b1, default config):**
| Stage | Channels | GAFF Params | % of Backbone | vs CSSA |
|-------|----------|-------------|---------------|---------|
| 1 | 64 | ~28,800 | 0.041% | 50× |
| 2 | 128 | ~115,200 | 0.166% | 134× |
| 3 | 320 | ~717,440 | 1.032% | 355× |
| 4 | 512 | ~1,835,008 | 2.640% | 579× |

**Multi-Stage Overhead:**
- Stages [2,3]: ~832K params (+1.20%)
- Stages [2,3,4]: ~2.67M params (+3.84%)
- Stages [1,2,3,4]: ~2.70M params (+3.88%)

**Key Insight:** GAFF adds significant overhead, especially at late stages with high channel counts.

---

## 5. Resource Usage Analysis

### 5.1 Completed Experiments

**Total GPU Time (Completed):**
```
Baseline:
- mit_b0: 2.68 hrs
- mit_b1: 3.44 hrs

CSSA Phase 1:
- exp_001, exp_002, exp_003: ~3.5 hrs each ≈ 10.5 hrs (estimated)

GAFF Phase 1:
- exp_001-004: ~3.4 hrs each ≈ 13.6 hrs (estimated)

Total: ~30 hours
```

**Note:** Actual CSSA/GAFF times are estimates. Precise timing data to be extracted from logs.

### 5.2 Resource Projections

**Remaining Work:**

| Category | Experiments | Time per Exp | Total Time |
|----------|-------------|--------------|------------|
| Baseline (mit_b2) | 1 | 3.5 hrs | 3.5 hrs |
| CSSA Phase 1 | 4 | 3.5 hrs | 14 hrs |
| CSSA Phase 2 | 4 | 3.5 hrs | 14 hrs |
| GAFF Phase 1 | 4 | 4 hrs | 16 hrs |
| GAFF Phase 2 | 21 | 4 hrs | 84 hrs |
| **Total** | **34** | - | **131.5 hrs** |

**Additional (if attempting mit_b4, mit_b5):**
- mit_b4: ~5-6 hours (if memory solutions work)
- mit_b5: ~6-8 hours (if memory solutions work)

**Grand Total Estimate:** ~145-175 GPU hours remaining

**Timeline Estimate:**
- Single A100 GPU: ~6-7 days of continuous training
- 2 GPUs in parallel: ~3-4 days
- 4 GPUs in parallel: ~2 days

### 5.3 Cost Analysis (Assuming Cloud GPU)

**A100 40GB GPU Pricing (approximate):**
- On-demand: ~$3-4/hour
- Spot instances: ~$1-2/hour

**Completed Work Cost:**
- 30 hours × $2/hr = ~$60 (spot pricing)

**Remaining Work Cost:**
- 145 hours × $2/hr = ~$290 (spot pricing)
- Total project: ~$350-400

---

## 6. Timeline & Milestones

### 6.1 Completed Milestones

| Date | Milestone | Status |
|------|-----------|--------|
| 2025-11-05 | Baseline mit_b0 training | ✅ Complete |
| 2025-11-06 | Baseline mit_b1 training | ✅ Complete |
| 2025-11-07 | CSSA Phase 1: exp_001-003 | ✅ Complete |
| 2025-11-07 | GAFF Phase 1: exp_001-004 | ✅ Complete |
| 2025-11-07 | mit_b2 restart (memory optimized) | 🔄 In Progress |

### 6.2 Upcoming Milestones

| Target Date | Milestone | Dependencies |
|-------------|-----------|--------------|
| 2025-11-08 | Complete mit_b2 training | Gradient accumulation working |
| 2025-11-09 | Complete CSSA Phase 1 (exp_004-007) | GPU availability |
| 2025-11-10 | Complete GAFF Phase 1 (exp_005-008) | GPU availability |
| 2025-11-11 | Analyze Phase 1 results, select top configs | All Phase 1 complete |
| 2025-11-12 | Start CSSA Phase 2 | Top 2 configs identified |
| 2025-11-12 | Start GAFF Phase 2 | Top 3 configs identified |
| 2025-11-15 | Complete CSSA Phase 2 | ~4 experiments |
| 2025-11-18 | Complete GAFF Phase 2 | ~21 experiments |
| 2025-11-19 | Final analysis and comparison | All experiments complete |
| 2025-11-20 | Documentation and publication | Results analyzed |

**Critical Path:** GAFF Phase 2 (longest duration, ~21 experiments)

### 6.3 Risk Factors

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| GPU unavailability | Medium | High | Queue jobs, use spot instances |
| Additional OOM errors | Medium | Medium | Preemptively reduce batch sizes |
| Training divergence | Low | Medium | Monitor losses, implement early stopping |
| Phase 1 inconclusive results | Low | High | Run all configs, analyze trends |
| Hardware failure | Low | High | Regular checkpointing, cloud backup |

---

## 7. Next Steps & Priorities

### 7.1 Immediate Actions (Next 24-48 hours)

1. **Monitor mit_b2 training**
   - Check if gradient accumulation resolves OOM
   - Validate loss convergence
   - If successful: Apply same strategy to mit_b4, mit_b5

2. **Launch CSSA Phase 1 remaining experiments**
   - Priority order: exp_004 (s4), exp_005 (s23), exp_006 (s34), exp_007 (s1234)
   - Run in parallel if multiple GPUs available

3. **Launch GAFF Phase 1 remaining experiments**
   - Priority order: exp_005 (s23), exp_006 (s34), exp_007 (s234), exp_008 (s1234)
   - Note: Multi-stage configs may be slower

4. **Extract and analyze completed results**
   - Parse CSSA exp_001-003 logs for mAP, mAP@50, training curves
   - Parse GAFF exp_001-004 logs similarly
   - Create preliminary comparison plots

### 7.2 Short-term Goals (Next 1 week)

1. **Complete all Phase 1 experiments**
   - CSSA: 4 remaining
   - GAFF: 4 remaining
   - Total: 8 experiments

2. **Analyze Phase 1 results**
   - Rank stage configurations by mAP
   - Identify top 2 for CSSA, top 3 for GAFF
   - Generate visualizations (bar charts, training curves)

3. **Design Phase 2 experiments**
   - CSSA: Select top 2 configs, plan threshold sweep
   - GAFF: Select top 3 configs, plan hyperparameter grid

4. **Document Phase 1 findings**
   - Update CSSA_ABLATION_GUIDE.md with results
   - Update GAFF_ABLATION_GUIDE.md with results
   - Create comparison figures

### 7.3 Medium-term Goals (Next 2-3 weeks)

1. **Execute Phase 2 ablations**
   - CSSA: 4 experiments (~14 hours)
   - GAFF: 21 experiments (~84 hours)
   - Parallelize where possible

2. **Comprehensive analysis**
   - Compare CSSA vs GAFF head-to-head
   - Analyze accuracy vs efficiency trade-offs
   - Identify best overall configuration

3. **Resolve baseline OOM issues**
   - Attempt mit_b4, mit_b5 with memory optimizations
   - Document memory requirements and solutions

4. **Publication preparation**
   - Finalize all figures and tables
   - Write experimental results section
   - Prepare for submission or technical report

---

## 8. Results Summary (Preliminary)

### 8.1 Baseline Results

**Available Results:**
| Backbone | Params (M) | Final Loss | Best Loss | Convergence | Notes |
|----------|------------|------------|-----------|-------------|-------|
| mit_b0 | 55.7 | TBD | 0.1057 | Good | Stable training |
| mit_b1 | 69.5 | TBD | 0.1027 | Good | ~3% better loss than b0 |
| mit_b2 | 82.1 | - | 0.3787 (epoch 1) | TBD | Training in progress |

**Key Observations:**
- Larger backbones achieve lower training loss
- Training is stable without fusion
- Memory is the limiting factor for b4, b5

### 8.2 CSSA Results (Preliminary)

**Phase 1 Completed:**
- exp_001 (s1): Results to be extracted
- exp_002 (s2): Results to be extracted
- exp_003 (s3): Results to be extracted

**Expected Patterns:**
- Stage 3 likely performs best (high-level semantic fusion)
- Stage 1 may underperform (too low-level)
- Multi-stage configs (s23, s34) may not improve much over best single stage

### 8.3 GAFF Results (Preliminary)

**Phase 1 Completed:**
- exp_001 (s1): Results to be extracted
- exp_002 (s2): Results to be extracted
- exp_003 (s3): Results to be extracted
- exp_004 (s4): Results to be extracted

**Expected Patterns:**
- Similar to CSSA: Stage 3 or 4 likely best
- GAFF may benefit more from multi-stage fusion due to richer attention
- Parameter overhead increases significantly for late stages

---

## 9. Known Issues & Workarounds

### 9.1 Memory Issues

**Issue:** mit_b4, mit_b5 cause CUDA OOM with batch_size=16

**Workaround Applied:**
```python
# Original config
batch_size = 16
gradient_accumulation_steps = 1

# Memory-optimized config
batch_size = 4  # or 2 for mit_b5
gradient_accumulation_steps = 4  # or 8 for mit_b5
# Maintains effective batch size = 16
```

**Status:** Testing on mit_b2, will apply to b4/b5 if successful

**Alternative Workarounds (not yet implemented):**
- Mixed precision training (FP16)
- Gradient checkpointing
- Reduce input resolution
- Multi-GPU data parallelism

### 9.2 Training Divergence

**Issue:** None observed yet, but potential risk for complex configs

**Prevention:**
- Monitor loss curves closely
- Implement gradient clipping if needed
- Early stopping if loss increases for 3+ epochs

### 9.3 Long Experiment Queue

**Issue:** 36 pending experiments, single GPU bottleneck

**Solutions:**
- Parallelize on multiple GPUs if available
- Use spot instances for cost savings
- Prioritize most informative experiments (e.g., single-stage before multi-stage)

---

## 10. Data & Artifact Locations

### 10.1 Results Directories

```
results/
├── baseline_backbones/
│   ├── mit_b0_15epochs/  # ✅ Complete
│   ├── mit_b1_15epochs/  # ✅ Complete
│   └── mit_b2_15epochs_restart/  # 🔄 In Progress
│
├── cssa_ablations/
│   ├── exp_001_s1_t0.5/  # ✅ Complete
│   ├── exp_002_s2_t0.5/  # ✅ Complete
│   ├── exp_003_s3_t0.5/  # ✅ Complete
│   └── exp_004-011/  # ⏸️ Pending
│
└── gaff_ablations_full/
    ├── phase1_stage_selection/
    │   ├── exp_001_s1_r4_is0_mb0/  # ✅ Complete
    │   ├── exp_002_s2_r4_is0_mb0/  # ✅ Complete
    │   ├── exp_003_s3_r4_is0_mb0/  # ✅ Complete
    │   ├── exp_004_s4_r4_is0_mb0/  # ✅ Complete
    │   └── exp_005-008/  # ⏸️ Pending
    ├── phase2_hyperparameter_tuning/  # ⏸️ Pending (21 experiments)
    ├── master_log.txt
    └── summary_all_experiments.csv
```

### 10.2 Key Files per Experiment

**Each experiment directory contains:**
```
exp_XXX_*/
├── checkpoints/
│   └── best_model.pth  # Best model by validation mAP
├── training.log  # Full training log
├── config.json  # Experiment configuration
├── model_info.json  # Model architecture details
├── metrics_per_epoch.csv  # Epoch-level metrics
├── metrics_per_batch.csv  # Batch-level metrics (if enabled)
└── final_results.json  # Final COCO evaluation results
```

### 10.3 Aggregated Summaries

**Master logs:**
- `results/gaff_ablations_full/master_log.txt` - Overall GAFF progress
- `results/cssa_ablations/summary_phase1.json` - CSSA Phase 1 summary (when complete)

**Summary CSVs:**
- `results/gaff_ablations_full/summary_all_experiments.csv` - All GAFF results
- Similar for CSSA (to be created)

---

## 11. Quick Commands

### 11.1 Check Experiment Status

```bash
# List all result directories
ls -lh results/*/

# Check GAFF experiment status
cat results/gaff_ablations_full/master_log.txt | grep "EXPERIMENT"

# Check latest training progress
tail -f results/gaff_ablations_full/phase1_*/exp_00X_*/training.log

# View summary CSV
cat results/gaff_ablations_full/summary_all_experiments.csv
```

### 11.2 Monitor Training

```bash
# Check GPU usage
nvidia-smi

# Monitor specific experiment
tail -f results/cssa_ablations/exp_003_*/training.log

# Check if experiment finished
ls results/gaff_ablations_full/phase1_*/exp_004_*/final_results.json
```

### 11.3 Extract Results

```bash
# Get final mAP for an experiment
cat results/gaff_ablations_full/phase1_*/exp_001_*/final_results.json | grep "mAP"

# List all completed experiments
find results/ -name "final_results.json" -type f

# Count completed experiments
find results/ -name "final_results.json" -type f | wc -l
```

---

## 12. Progress Visualization

### 12.1 Completion Status (Text-based)

```
Overall Progress: [██████░░░░░░░░░░░░░░░░░░░░░░░░░░] 18.75% (9/48)

Baseline:     [████░░░░░░] 40% (2/5)  ✅ mit_b0, mit_b1  🔄 mit_b2  ❌ b4, b5
CSSA Phase 1: [████░░░░░░] 43% (3/7)  ✅ s1, s2, s3  ⏸️ s4, s23, s34, s1234
CSSA Phase 2: [░░░░░░░░░░] 0% (0/4)   ⏸️ Blocked on Phase 1
GAFF Phase 1: [█████░░░░░] 50% (4/8)  ✅ s1, s2, s3, s4  ⏸️ s23, s34, s234, s1234
GAFF Phase 2: [░░░░░░░░░░] 0% (0/24)  ⏸️ Blocked on Phase 1

Legend: ██ Complete  ░░ Pending  🔄 In Progress  ❌ Failed
```

### 12.2 Time Investment

```
Time Spent:     [30 hours] ████░░░░░░░░░░░░░░░░░░░░░░
Time Remaining: [145 hours] ░░░░░░░░░░░░░░░░░░░░░░░░░░

Total Estimate: 175 GPU hours = 7.3 days (single GPU, 24/7)
```

---

## Appendix: Experiment Reference

### All Experiments at a Glance

| ID | Category | Name | Status | Priority |
|----|----------|------|--------|----------|
| 1 | Baseline | mit_b0 | ✅ Complete | - |
| 2 | Baseline | mit_b1 | ✅ Complete | - |
| 3 | Baseline | mit_b2 | 🔄 Training | High |
| 4 | Baseline | mit_b4 | ❌ Failed (OOM) | Medium |
| 5 | Baseline | mit_b5 | ❌ Failed (OOM) | Low |
| 6 | CSSA-P1 | exp_001 (s1) | ✅ Complete | - |
| 7 | CSSA-P1 | exp_002 (s2) | ✅ Complete | - |
| 8 | CSSA-P1 | exp_003 (s3) | ✅ Complete | - |
| 9 | CSSA-P1 | exp_004 (s4) | ⏸️ Pending | High |
| 10 | CSSA-P1 | exp_005 (s23) | ⏸️ Pending | High |
| 11 | CSSA-P1 | exp_006 (s34) | ⏸️ Pending | Medium |
| 12 | CSSA-P1 | exp_007 (s1234) | ⏸️ Pending | Medium |
| 13-16 | CSSA-P2 | Threshold tuning | ⏸️ Pending | Medium |
| 17 | GAFF-P1 | exp_001 (s1) | ✅ Complete | - |
| 18 | GAFF-P1 | exp_002 (s2) | ✅ Complete | - |
| 19 | GAFF-P1 | exp_003 (s3) | ✅ Complete | - |
| 20 | GAFF-P1 | exp_004 (s4) | ✅ Complete | - |
| 21 | GAFF-P1 | exp_005 (s23) | ⏸️ Pending | High |
| 22 | GAFF-P1 | exp_006 (s34) | ⏸️ Pending | High |
| 23 | GAFF-P1 | exp_007 (s234) | ⏸️ Pending | Medium |
| 24 | GAFF-P1 | exp_008 (s1234) | ⏸️ Pending | Medium |
| 25-48 | GAFF-P2 | Hyperparam grid | ⏸️ Pending | Medium |

**Priority Legend:**
- High: Critical for Phase 1 completion or immediate next step
- Medium: Important but can wait for Phase 1 analysis
- Low: Nice to have, attempt if time/resources allow

---

**Dashboard Last Updated:** 2025-11-07 18:37:00 UTC
**Next Update:** After Phase 1 completion (estimated 2025-11-10)

**For questions or issues, refer to:**
- `docs/CSSA_ABLATION_GUIDE.md` - CSSA methodology and analysis
- `docs/GAFF_ABLATION_GUIDE.md` - GAFF methodology and analysis
- `docs/FUSION_MECHANISMS_COMPARISON.md` - CSSA vs GAFF comparison
- `docs/EXPERIMENTAL_MATRIX.md` - Full experimental design
