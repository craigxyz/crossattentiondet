blations/fusion_mechanisms_plan.md

**Purpose.** Implement and evaluate *fusion mechanism & placement* ablations on top of the CMX-FPN Faster R-CNN baseline in `crossattentiondet`. The goal is to (i) benchmark your FRM/FFM hybrid against strong, recognizable baselines from the literature; (ii) measure where fusion should happen in the hierarchy; and (iii) produce camera-ready tables/figures that address CVPR reviewer critiques.

**External methods to reproduce as drop-in modules (feature-level unless noted):**

* **CSSA** — Channel Switching + Spatial Attention (CVPR’23 PBVS Wksp). Lightweight dual-backbone fusion designed for RGBT detection. ([CVF Open Access][1])
* **GAFF** — Guided Attentive Feature Fusion (WACV’21). Inter- and intra-modality attention for multispectral detection. ([CVF Open Access][2])
* **DetFusion-style gating** — Detection-driven fusion: use detector cues (objectness/semantics) to guide fusion (our adaptation at feature level; original is image-level). ([ACM Digital Library][3])
* **ProbEn** — Probabilistic Ensembling (detection-level). Fuse outputs from single-modal detectors via Bayesian marginalization; use as a *strong non-learned* baseline and as a “rescue” under missing modalities. ([arXiv][4])

---

## 0) Repo prerequisites

* **Branch:** create `feature/fusion-ablate` off `main`.
* **Env:** pin versions (`torch`, `torchvision`, `torchmetrics`, `timm`). Add `fvcore` or `ptflops` for MACs; `wandb` or TensorBoard for logs.
* **Data:** ensure 5-ch NPY loader is working; keep a *fixed* scene-wise split (train/val/test). Save split JSONs for reproducibility.
* **Metrics:** COCO mAP@[.5:.95], AP50/75, APs/m/l. Also log *condition-wise* AP (day/night, low-light, high-motion) via eval tags.

Artifacts folder structure:

```
crossattentiondet/
  ablations/
    fusion_mechanisms_plan.md   # (this file)
    configs/
      fusion/{cmx_frmffm, cssa, gaff, detgate}.yaml
      where/{early,mid,late,progressive}.yaml
    scripts/
      run_grid.sh
      export_predictions.py      # for ProbEn
      prob_en.py                 # detection-level fuse
    modules/
      fusion/
        cssa.py
        gaff.py
        det_gate.py
        adapters.py              # interface shims
  results/
    fusion/ ... csv/png
```

---

## 1) Interfaces & integration points

Your baseline backbone is `CMX_FPN_Backbone` that returns an **OrderedDict** of FPN features (`'0','1','2','3','pool'`). Internally, CMX emits 4 fused maps via FRM/FFM across stages 1–4.

We will inject alternative fusion blocks **at the CMX stage outputs** (before FPN) using a common interface:

```python
class FusionBlock(nn.Module):
    def forward(self, x_rgb: Tensor, x_aux: Tensor) -> Tensor: 
        """Return fused feature map of shape (B, C, H, W)."""
```

* **Baseline:** `FRM` + `FFM` (already implemented).
* **Drop-in:** `CSSABlock`, `GAFFBlock`, `DetGateBlock`.

Add a dispatcher in `modules/fusion/adapters.py`:

```python
def build_fusion_block(kind: str, dim: int, **kwargs) -> nn.Module:
    if kind == "frm_ffm": return FRMFFM(dim, **kwargs)
    if kind == "cssa":    return CSSABlock(dim, **kwargs)
    if kind == "gaff":    return GAFFBlock(dim, **kwargs)
    if kind == "detgate": return DetGateBlock(dim, **kwargs)
    raise ValueError
```

Wire a CLI/config flag `MODEL.FUSION.KIND` and a list `MODEL.FUSION.STAGES = [1,2,3,4]` to control *where* to fuse.

---

## 2) Implement alternative fusion blocks

### 2.1 CSSA (Channel Switching + Spatial Attention)

**Paper intent.** Swap channels between modalities to expose complementary responses, and modulate with spatial attention. Designed to be lightweight and memory-friendly for RGBT detection. ([CVF Open Access][1])

**Implementation sketch** (`modules/fusion/cssa.py`):

* Inputs: `(x_rgb, x_aux)` at same spatial size and `C` channels.
* **Channel switching:** split each into `C1,C2` chunks (e.g., half); concatenate cross-chunks:
  `rgb' = [rgb_C1, aux_C2]`, `aux' = [aux_C1, rgb_C2]`. (Make split ratio a hyperparam.)
* **Spatial attention:** `SA(·)` = `Conv1x1(C) → ReLU → Conv1x1(1) → Sigmoid`.
  `w_rgb = SA([x_rgb, x_aux])`, `w_aux = SA([x_rgb, x_aux])` (share or separate weights; start shared).
* **Fuse:** `y = BN(Conv1x1([w_rgb*rgb', w_aux*aux']))` (concat then 1x1 bottleneck).
* Return fused `y` (B,C,H,W).

**Config knobs:**

* `CHANNEL_SPLIT: 0.5|0.25|0.75`
* `ATTN_SHARE: true|false`
* `NORM: BN|GN`

**Notes:** Be faithful to the *lightweight* spirit; avoid self-attention. Cite when reporting memory/params. ([CVF Open Access][1])

---

### 2.2 GAFF (Guided Attentive Feature Fusion)

**Paper intent.** Learn inter- and intra-modality attention to guide fusion; widely cited for multispectral detection. ([CVF Open Access][2])

**Implementation sketch** (`modules/fusion/gaff.py`):

* **Intra-modality attention:** SE-style channel weights per stream (`SE(x)=GAP→MLP→σ`).
* **Inter-modality attention:** compute cross-attention *weights* as 1×1 convs over `concat(x_rgb, x_aux)`, producing `w_rgb←aux`, `w_aux←rgb`.
* Apply: `x̂_rgb = SE_rgb(x_rgb) + w_rgb←aux * x_aux`; `x̂_aux = SE_aux(x_aux) + w_aux←rgb * x_rgb`.
* **Merge:** concat `[x̂_rgb, x̂_aux] → 1×1 conv → BN` to fused `y`.

**Config knobs:**

* `SE_REDUCTION: 4|8`
* `INTER_SHARED: true|false`
* `MERGE_DIM: C|2C→C` (choose bottleneck width)

**Reference:** Ensure naming aligns with GAFF figures/terminology in WACV’21. ([CVF Open Access][2])

---

### 2.3 DetFusion-style detection-driven gating (feature-level adaptation)

**Paper intent.** Use detection cues to inform fusion (“task-driven fusion”). Original fuses images with detection signals in-loop; we adapt the *principle* to feature maps by injecting RPN objectness maps as spatial priors into fusion. ([ACM Digital Library][3])

**Implementation sketch** (`modules/fusion/det_gate.py`):

* **Upstream signal:** from the previous epoch (or previous iteration for teacher-forcing), cache per-level **RPN objectness heatmaps** resized to the current fusion stage resolution.
* Build **gates**: `g = σ(Conv1x1([x_rgb, x_aux, objness]))`.
* Reweight per-stream: `x̃_rgb = g * x_rgb + (1-g) * x_aux`; `x̃_aux = g * x_aux + (1-g) * x_rgb` (or two gates).
* **Merge:** concat → 1×1 conv → BN to `y`.

**Training protocol:**

* **Warm-up** 1–2 epochs with vanilla FRM/FFM (or CSSA/GAFF) to stabilize objectness.
* Then enable `det_gate` (read objectness from EMA of the last epoch).

**Config knobs:**

* `OBJ_SOURCE: rpn|roi_objness` (start with RPN)
* `GATE_MODE: shared|dual`
* `WARMUP_EPOCHS: 2`

**Caveat:** Avoid label leakage (no GT in gates). Cite DetFusion as *detection-driven* philosophy; clarify this is a feature-space adaptation. ([ACM Digital Library][3])

---

### 2.4 ProbEn (detection-level, non-learned)

**Paper intent.** Fuse *detections* (not features) from single-modal experts via probabilistic marginalization; handles missing modality and misalignment gracefully; widely used baseline. ([arXiv][4])

**Implementation (`scripts/prob_en.py`):**

* Train three experts with *identical heads*: RGB-only, Thermal-only, Event-only.
* Export predictions (`export_predictions.py`) → JSON per image: boxes, class posteriors, scores.
* **ProbEn**: For each class, compute fused posterior `p(y|x_rgb,x_ir,x_evt) ∝ ∏_m p(y|x_m)` under conditional independence; implement box association via IoU-based bipartite matching + log-prob sum.
* NMS post-processing identical to baseline.

**Use cases:**

* Report **ProbEn** alone (tri-expert) and **Hybrid**: `(CMX detector) ⊕ (ProbEn of experts)` by concatenating detections then NMS.
* Include memory trade-off discussion (ProbEn doubles memory at inference vs feature-fusion). ([CVF Open Access][1])

**Reference code:** seed from public ECCV’22 implementations if needed. ([GitHub][5])

---

## 3) “Where to fuse?” — placement matrix

Add `MODEL.FUSION.STAGES` to choose the CMX stages that use the fusion block. Evaluate:

* **Early-only:** `{1}`
* **Mid-only:** `{2}` and `{3}`
* **Late-only:** `{4}`
* **Two-stage:** `{2,3}` and `{3,4}`
* **All-stages (progressive):** `{1,2,3,4}` (baseline CMX behavior)

Run this grid for **FRM/FFM (baseline), CSSA, GAFF**. Expect mid/progressive to be best; include compute trade-offs.

---

## 4) Experimental design

### 4.1 Fixed settings

* **Head:** Faster R-CNN (your current config).
* **Input side:** 800 px (keep constant).
* **Epochs:** 25 (ablations) with StepLR at ⅔/5⁶; 50 for final main table.
* **Batch:** 2; **Seeds:** {0,1,2} (report mean±std).
* **Logging:** wandb/TensorBoard scalars (`losses`, `mAP`, `objness` stats), histograms for channel gates, images with GT/preds (hard cases).

### 4.2 Grid (minimum for CVPR rebuttal strength)

**A. Mechanism swap @ progressive fusion (`{1,2,3,4}`):**

* FRM+FFM (baseline)
* CSSA (default config)
* GAFF (default config)
* DetGate (after 2-epoch warm-up)

**B. Placement study (mechanism = *best of A*):**

* `{1}`, `{2}`, `{3}`, `{4}`, `{2,3}`, `{3,4}`, `{1,2,3,4}`

**C. Lightweight vs heavy:**

* For CSSA & GAFF, vary bottleneck width by ×0.5 and ×1.5; record **Params/MACs/Latency**.

**D. ProbEn (detection-level):**

* RGB, IR, Event experts → ProbEn fuse.
* Hybrid: `(best feature-fusion) ⊕ ProbEn`.

---

## 5) Evaluation & reporting

### 5.1 Metrics

* **Overall:** COCO mAP@[.5:.95], AP50/75, APs/m/l.
* **Condition-wise:** day/night, low-light, high-motion (bins from metadata or heuristics).
* **Runtime:** Desktop + Jetson Orin (bs=1) latency; Params/MACs via `fvcore/ptflops`.

### 5.2 Tables (CSV schemas)

* `results/fusion/main_table.csv`
  `method, fusion_kind, stages, params_m, macs_g, latency_ms_orin, mAP, AP50, AP_night, AP_lowlight`
* `results/fusion/placement.csv`
  `fusion_kind, stages, mAP, latency_ms`
* `results/fusion/compute_tradeoff.csv`
  `fusion_kind, width_scale, params_m, macs_g, fps_orin, mAP`
* `results/proben/proben.csv`
  `setting, mAP, AP50, AP_night, AP_lowlight`

### 5.3 Figures

* **PR curves** per class (overall + night subset) for top-2 methods.
* **Qual grids**: 3×3 panels (RGB/IR/Event) with GT(green)/Pred(red), hard cases (glare, low-light, motion).
* **Ablation bar plot**: mAP vs fusion stage; include error bars across seeds.

---

## 6) Implementation details (step-by-step)

1. **Create fusion API**

   * `modules/fusion/adapters.py` with `build_fusion_block`.
   * Refactor CMX to call `build_fusion_block(KIND, dim)` at each stage if that stage ∈ `MODEL.FUSION.STAGES`, else **passthrough** (no fusion).

2. **Wire configs & CLI**

   * Add `--fusion.kind [frm_ffm|cssa|gaff|detgate]`
   * Add `--fusion.stages "1,2,3,4"` (parse to list of ints)
   * Add `--fusion.width_scale` (for CSSA/GAFF bottlenecks)
   * Add `--detgate.warmup_epochs`, `--detgate.obj_source`

3. **Implement CSSA (`cssa.py`)**

   * Utility: `ChannelSplit(dim, ratio)` to split along **channel**.
   * Spatial attention head with shared weights by default.
   * Unit tests: check shapes; run on random tensors (B=2,C=64,H=W=64).

4. **Implement GAFF (`gaff.py`)**

   * SE blocks per stream (`reduction=4` default).
   * Inter-modality attention via 1×1 conv on `concat(x_rgb, x_aux)`.
   * Unit tests as above.

5. **Implement DetGate (`det_gate.py`)**

   * Hook into training loop to **cache** per-level RPN objectness maps (detach, store EMA).
   * During forward, read nearest-resolution objectness (bilinear resize).
   * Warm-up logic in trainer: if `epoch < warmup_epochs`, bypass to base fusion.

6. **ProbEn (`scripts/prob_en.py`)**

   * Implement IoU-based association and log-prob sum fusion (per class).
   * CLI to fuse saved predictions from RGB/IR/Event experts.

7. **Placement control**

   * In CMX forward, after each stage’s per-stream transformers, check if `stage_id ∈ STAGES`:
     `x_rgb, x_aux = fusion_block(x_rgb, x_aux)` else leave as is.
     (For FRM/FFM baseline, respect original flow; for CSSA/GAFF, replace FRM+FFM at that stage.)

8. **Compute measurement**

   * Add `utils/prof.py` to compute Params/MACs and profile latency on a dummy input `(B=1,5,H=800,W=1280)` (or your default).
   * Save to `results/fusion/compute_tradeoff.csv`.

9. **Logging & artifacts**

   * Standardize run names:
     `fusion-{kind}_stages-{stages}_w-{width}_seed-{s}`
   * Save CSVs per run; aggregate with a notebook/script to produce final tables.

---

## 7) Sanity checks & failure modes

* **Shape mismatches** when swapping blocks → assert `(B,C,H,W)` invariants after each fusion.
* **Numerical blow-ups** with attention/gates → clamp gate logits; use `eps` in normalizations.
* **DetGate leakage** → verify gates are derived from *predicted* objectness only (no GT).
* **ProbEn duplication** → ensure identical box is not double-counted; use NMS with class-wise IoU threshold.

---

## 8) Deliverables (CVPR-ready)

* **Table 1 (Main):** FRM/FFM vs **CSSA** vs **GAFF** vs **DetGate** (progressive), + **ProbEn**; report mAP overall + night/low-light; include Params/MACs/latency.
  *(CSSA/GAFF/ProbEn are recognizable baselines to reviewers.)* ([CVF Open Access][1])
* **Table 2 (Placement):** mAP vs fusion stage(s) for the best mechanism; highlight “mid/late vs early” findings.
* **Table 3 (Compute trade-off):** accuracy vs bottleneck width; include Jetson Orin latency.
* **Figure 1:** PR curves (night subset) for top-2 methods.
* **Figure 2:** Qualitative grid (hard cases).
* **Appendix:** Implementation details + config dumps.

---

## 9) References

* **CSSA:** Cao *et al.*, *Multimodal Object Detection by Channel Switching and Spatial Attention*, CVPR 2023 PBVS Workshop. ([CVF Open Access][1])
* **GAFF:** Zhang *et al.*, *Guided Attentive Feature Fusion for Multispectral Pedestrian Detection*, WACV 2021. ([CVF Open Access][2])
* **DetFusion:** Sun *et al.*, *DetFusion: A Detection-driven Infrared and Visible Image Fusion Network*, ACM MM 2022 (official repo: mmdetection-based). ([ACM Digital Library][3])
* **ProbEn:** Chen *et al.*, *Multimodal Object Detection via Probabilistic Ensembling*, ECCV 2022 (arXiv/ECCV version). ([arXiv][4])

---

## 10) Execution checklist (agent-friendly)

* [ ] Create branch `feature/fusion-ablate`, folders, and config templates.
* [ ] Implement `FusionBlock` API + `build_fusion_block`.
* [ ] Implement `CSSABlock` (+ unit test).
* [ ] Implement `GAFFBlock` (+ unit test).
* [ ] Implement `DetGateBlock` + trainer warm-up & objectness cache.
* [ ] Wire `MODEL.FUSION.{KIND,STAGES,WIDTH}` flags.
* [ ] Implement `prob_en.py` and `export_predictions.py`.
* [ ] Add `utils/prof.py` for Params/MACs/latency; log to CSV.
* [ ] Run Grid A (mechanism swap), Grid B (placement), Grid C (compute), Grid D (ProbEn).
* [ ] Aggregate CSVs → main_table/placement/compute/proben; generate plots.
* [ ] Commit results to `results/` and update paper tables/figures.

**Success criterion:** at least one alternative fusion (CSSA/GAFF/DetGate) outperforms FRM/FFM on **night** and **low-light** AP, while matching or improving overall mAP and giving a clear narrative about *where* to fuse. ProbEn should be competitive as a non-learned baseline and complementary in hybrid mode.

[1]: https://openaccess.thecvf.com/content/CVPR2023W/PBVS/papers/Cao_Multimodal_Object_Detection_by_Channel_Switching_and_Spatial_Attention_CVPRW_2023_paper.pdf "Multimodal Object Detection by Channel Switching and ..."
[2]: https://openaccess.thecvf.com/content/WACV2021/papers/Zhang_Guided_Attentive_Feature_Fusion_for_Multispectral_Pedestrian_Detection_WACV_2021_paper.pdf "Guided Attentive Feature Fusion for Multispectral ..."
[3]: https://dl.acm.org/doi/10.1145/3503161.3547902 "DetFusion: A Detection-driven Infrared and Visible Image ..."
[4]: https://arxiv.org/abs/2104.02904 "Multimodal Object Detection via Probabilistic Ensembling"
[5]: https://github.com/Jamie725/Multimodal-Object-Detection-via-Probabilistic-Ensembling "Jamie725/Multimodal-Object-Detection-via-Probabilistic- ..."

