# Fusion Architectures Visual Guide

**Document Version:** 1.0
**Last Updated:** 2025-11-07

---

## Overview

This document provides visual representations of the CSSA and GAFF fusion architectures using ASCII art diagrams, flowcharts, and decision trees. These visualizations help understand the multi-modal fusion mechanisms and guide configuration selection.

---

## Table of Contents

1. [CSSA Architecture Flowcharts](#1-cssa-architecture-flowcharts)
2. [GAFF Architecture Flowcharts](#2-gaff-architecture-flowcharts)
3. [Multi-Stage Encoder Integration](#3-multi-stage-encoder-integration)
4. [Configuration Decision Trees](#4-configuration-decision-trees)
5. [Component Comparison Diagrams](#5-component-comparison-diagrams)
6. [Data Flow Visualizations](#6-data-flow-visualizations)

---

## 1. CSSA Architecture Flowcharts

### 1.1 Complete CSSA Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         CSSA FUSION BLOCK                               │
│                  Channel Switching + Spatial Attention                  │
└─────────────────────────────────────────────────────────────────────────┘

INPUT:
┌──────────────┐                        ┌──────────────┐
│  RGB Features│                        │ Aux Features │
│   (B,C,H,W)  │                        │   (B,C,H,W)  │
└──────┬───────┘                        └──────┬───────┘
       │                                       │
       │                                       │
─────────────────────────────────────────────────────────────────────────
STEP 1: CHANNEL ATTENTION (ECA)
─────────────────────────────────────────────────────────────────────────
       │                                       │
       │                                       │
       v                                       v
┌──────────────┐                        ┌──────────────┐
│  ECABlock    │                        │  ECABlock    │
│              │                        │              │
│  1. GAP      │                        │  1. GAP      │
│  2. Conv1d   │                        │  2. Conv1d   │
│  3. Sigmoid  │                        │  3. Sigmoid  │
└──────┬───────┘                        └──────┬───────┘
       │                                       │
       v                                       v
  attn_rgb                                attn_aux
  (B,C,1,1)                              (B,C,1,1)
       │                                       │
       │                                       │
─────────────────────────────────────────────────────────────────────────
STEP 2: CHANNEL SWITCHING (Hard Threshold)
─────────────────────────────────────────────────────────────────────────
       │                                       │
       └───────────────┬───────────────────────┘
                       │
                       v
            ┌──────────────────────┐
            │  Channel Switching   │
            │                      │
            │  For each channel c: │
            │  ──────────────────  │
            │  if attn_rgb[c] > k: │
            │    keep RGB[c]       │
            │  elif attn_aux[c]>k: │
            │    RGB[c]←Aux[c]     │
            │    Aux[c]←RGB[c]     │
            │  else:               │
            │    keep original     │
            └──────────┬───────────┘
                       │
           ┌───────────┴───────────┐
           v                       v
    rgb_switched              aux_switched
     (B,C,H,W)                 (B,C,H,W)
           │                       │
           │                       │
─────────────────────────────────────────────────────────────────────────
STEP 3: SPATIAL ATTENTION
─────────────────────────────────────────────────────────────────────────
           │                       │
           └───────────┬───────────┘
                       │
                       v
                ┌──────────────┐
                │ Concatenate  │
                │  (B,2C,H,W)  │
                └──────┬───────┘
                       │
                       v
            ┌──────────────────────┐
            │  Channel Pooling     │
            │  ─────────────────   │
            │  - Avg Pool → w_avg  │
            │  - Max Pool → w_max  │
            └──────────┬───────────┘
                       │
                       v
              ┌────────────────┐
              │   Conv 7×7     │
              │   + Sigmoid    │
              └────────┬───────┘
                       │
                       v
                 Spatial weights
                   (B,1,H,W)
                       │
                       v
            ┌──────────────────────┐
            │  Weighted Fusion     │
            │  ──────────────────  │
            │  fused = w * rgb +   │
            │        (1-w) * aux   │
            └──────────┬───────────┘
                       │
                       v

OUTPUT:
              ┌────────────────┐
              │ Fused Features │
              │   (B,C,H,W)    │
              └────────────────┘

PARAMETERS:
  ECA (2 blocks): ~2 × C × k           (k=3-7, typically 3)
  Channel Switching: 0                 (threshold is fixed hyperparameter)
  Spatial Attention: 2 × 1 × 7 × 7 + 1 = 99
  ──────────────────────────────────────────────────
  TOTAL (C=320): ~2,019 parameters     (0.003% of 69.5M backbone)
```

### 1.2 Channel Switching Logic Detail

```
CHANNEL SWITCHING DECISION TREE
(For a single channel c)

                        ┌─────────────────┐
                        │  Input Channel  │
                        │  RGB[c], Aux[c] │
                        └────────┬────────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
                    v                         v
        ┌─────────────────────┐   ┌─────────────────────┐
        │  attn_rgb[c]        │   │  attn_aux[c]        │
        │  (RGB confidence)   │   │  (Aux confidence)   │
        └──────────┬──────────┘   └──────────┬──────────┘
                   │                         │
                   └───────────┬─────────────┘
                               │
                               v
                  ┌────────────────────────┐
                  │ Compare with threshold │
                  │    (default: 0.5)      │
                  └────────────┬───────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              v                v                v
     ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
     │attn_rgb > k  │  │attn_aux > k  │  │ Both ≤ k     │
     │              │  │ (and rgb≤k)  │  │              │
     └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
            │                 │                 │
            v                 v                 v
     ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
     │ RGB confident│  │ Aux confident│  │ Both weak    │
     │              │  │              │  │              │
     │ Keep RGB[c]  │  │ Swap:        │  │ Keep RGB[c]  │
     │              │  │ RGB[c]←Aux[c]│  │ (default)    │
     └──────────────┘  └──────────────┘  └──────────────┘


THRESHOLD EFFECTS:

threshold = 0.3 (Low - Aggressive Switching):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Low bar for "confident"
  → More channels satisfy threshold
  → More frequent swapping
  → Greater cross-modal interaction
  → Risk: May disrupt good RGB features

threshold = 0.5 (Medium - Balanced):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Moderate bar for "confident"
  → Moderate swapping frequency
  → Balanced RGB/Aux contribution
  → Default choice

threshold = 0.7 (High - Conservative Switching):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  High bar for "confident"
  → Fewer channels satisfy threshold
  → Less frequent swapping
  → Mostly preserves RGB
  → Risk: Aux modality underutilized
```

---

## 2. GAFF Architecture Flowcharts

### 2.1 Complete GAFF Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         GAFF FUSION BLOCK                               │
│               Guided Attentive Feature Fusion (WACV 2021)               │
└─────────────────────────────────────────────────────────────────────────┘

INPUT:
┌──────────────┐                        ┌──────────────┐
│  RGB Features│                        │ Aux Features │
│   (B,C,H,W)  │                        │   (B,C,H,W)  │
└──────┬───────┘                        └──────┬───────┘
       │                                       │
       │                                       │
─────────────────────────────────────────────────────────────────────────
STEP 1: INTRA-MODALITY ATTENTION (SE Blocks)
─────────────────────────────────────────────────────────────────────────
       │                                       │
       v                                       v
┌──────────────┐                        ┌──────────────┐
│   SEBlock    │                        │   SEBlock    │
│              │                        │              │
│  1. GAP      │                        │  1. GAP      │
│  2. FC(C→r)  │                        │  2. FC(C→r)  │
│  3. ReLU     │                        │  3. ReLU     │
│  4. FC(r→C)  │                        │  4. FC(r→C)  │
│  5. Sigmoid  │                        │  5. Sigmoid  │
│  6. Scale    │                        │  6. Scale    │
└──────┬───────┘                        └──────┬───────┘
       │                                       │
       v                                       v
   x_rgb_se                                x_aux_se
   (B,C,H,W)                              (B,C,H,W)
       │                                       │
       │              (original features)      │
       │              ┌───────────────────┐    │
       ├──────────────┤   x_rgb, x_aux    ├────┤
       │              └─────────┬─────────┘    │
       │                        │              │
─────────────────────────────────────────────────────────────────────────
STEP 2: INTER-MODALITY ATTENTION (Cross-Attention)
─────────────────────────────────────────────────────────────────────────
       │                        │              │
       │                        v              │
       │                 ┌──────────────┐      │
       │                 │ Concatenate  │      │
       │                 │  [rgb, aux]  │      │
       │                 │  (B,2C,H,W)  │      │
       │                 └──────┬───────┘      │
       │                        │              │
       │                        v              │
       │          ┌─────────────────────────┐  │
       │          │ Inter-Modality Attention│  │
       │          │ ──────────────────────  │  │
       │          │ Option A (shared=False):│  │
       │          │   Conv_rgb(2C→C)        │  │
       │          │   Conv_aux(2C→C)        │  │
       │          │                         │  │
       │          │ Option B (shared=True): │  │
       │          │   Conv(2C→2C) → Split  │  │
       │          └────┬──────────────┬─────┘  │
       │               │              │        │
       │               v              v        │
       │         w_rgb←aux        w_aux←rgb   │
       │         (B,C,H,W)       (B,C,H,W)    │
       │               │              │        │
       │               │              │        │
─────────────────────────────────────────────────────────────────────────
STEP 3: GUIDED FUSION (Residual-Style)
─────────────────────────────────────────────────────────────────────────
       │               │              │        │
       v               v              v        v
   x_rgb_se      w_rgb←aux  ×  x_aux    x_aux_se   w_aux←rgb × x_rgb
       │               │              │        │
       └───────────────┴──────────────┘        │
                       │                       │
                       v                       v
                ┌──────────────┐        ┌──────────────┐
                │ x_rgb_guided │        │ x_aux_guided │
                │ = x_rgb_se + │        │ = x_aux_se + │
                │   w × x_aux  │        │   w × x_rgb  │
                └──────┬───────┘        └──────┬───────┘
                       │                       │
                       │                       │
─────────────────────────────────────────────────────────────────────────
STEP 4: MERGE
─────────────────────────────────────────────────────────────────────────
                       │                       │
                       └───────────┬───────────┘
                                   │
                                   v
                        ┌──────────────────┐
                        │   Concatenate    │
                        │ [rgb_g, aux_g]   │
                        │   (B,2C,H,W)     │
                        └────────┬─────────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
                    v                         v
        ┌───────────────────────┐ ┌───────────────────────┐
        │ Direct Merge          │ │ Bottleneck Merge      │
        │ (merge_bottleneck=F)  │ │ (merge_bottleneck=T)  │
        │ ───────────────────   │ │ ───────────────────   │
        │ Conv(2C→C) + BN       │ │ Conv(2C→C) + BN       │
        │                       │ │    ↓                  │
        │                       │ │  ReLU                 │
        │                       │ │    ↓                  │
        │                       │ │ Conv(C→C) + BN        │
        └───────────┬───────────┘ └───────────┬───────────┘
                    │                         │
                    └────────────┬────────────┘
                                 │
                                 v

OUTPUT:
                        ┌────────────────┐
                        │ Fused Features │
                        │   (B,C,H,W)    │
                        └────────────────┘

PARAMETERS (C=320, reduction=4, inter_shared=False, merge_bottleneck=False):
  SE_RGB: 2 × C²/4 = 51,200
  SE_Aux: 2 × C²/4 = 51,200
  Inter_RGB: 2C × C = 204,800
  Inter_Aux: 2C × C = 204,800
  Merge: 2C × C = 204,800
  BN: 2C = 640
  ──────────────────────────────────────────
  TOTAL: 717,440 parameters (1.03% of 69.5M backbone)
```

### 2.2 SE Block Detail

```
SQUEEZE-AND-EXCITATION (SE) BLOCK
(Intra-Modality Channel Attention)

Input: x (B, C, H, W)

       ┌─────────────────┐
       │   Feature Map   │
       │   (B,C,H,W)     │
       └────────┬────────┘
                │
                v
       ┌─────────────────┐
       │ Global Avg Pool │
       │ (Squeeze step)  │
       └────────┬────────┘
                │
                v
          (B, C, 1, 1)
                │
                v  Flatten
          (B, C)
                │
                v
       ┌─────────────────┐
       │ FC: C → C/r     │  ← Bottleneck (reduction r)
       │ (Excitation)    │
       └────────┬────────┘
                │
                v
          (B, C/r)
                │
                v
       ┌─────────────────┐
       │     ReLU        │  ← Non-linearity
       └────────┬────────┘
                │
                v
       ┌─────────────────┐
       │ FC: C/r → C     │  ← Expansion back to C
       └────────┬────────┘
                │
                v
          (B, C)
                │
                v
       ┌─────────────────┐
       │    Sigmoid      │  ← Attention weights [0,1]
       └────────┬────────┘
                │
                v  Reshape
          (B, C, 1, 1)
                │
                v
       ┌─────────────────┐
       │  Multiply with  │  ← Scale input channels
       │  input x        │
       └────────┬────────┘
                │
                v
       ┌─────────────────┐
       │ Output: x_se    │
       │   (B,C,H,W)     │
       └─────────────────┘

REDUCTION RATIO EFFECTS:

reduction = 4:
  Bottleneck size = C/4
  Parameters: 2×C²/4 = C²/2
  More expressive, richer representation

reduction = 8:
  Bottleneck size = C/8
  Parameters: 2×C²/8 = C²/4
  Fewer params, stronger compression

reduction = 16:
  Bottleneck size = C/16
  Parameters: 2×C²/16 = C²/8
  Very compact, may lose information
```

---

## 3. Multi-Stage Encoder Integration

### 3.1 Four-Stage Encoder with Fusion Points

```
MULTI-MODAL ENCODER WITH FUSION STAGES
(SegFormer/MiT-style architecture)

                   INPUT
                     │
         ┌───────────┴───────────┐
         │                       │
         v                       v
    ┌────────┐              ┌────────┐
    │  RGB   │              │  Aux   │
    │ Image  │              │ Image  │
    │(3,H,W) │              │(2,H,W) │
    └────┬───┘              └────┬───┘
         │                       │
         │                       │
┌────────────────────────────────────────────────────────┐
│ STAGE 1: Early Features (H/4 × W/4)                   │
│ Channels: 64                                           │
└────────────────────────────────────────────────────────┘
         │                       │
         v                       v
    ┌─────────┐             ┌─────────┐
    │ Patch   │             │ Patch   │
    │ Embed   │             │ Embed   │
    │ + Trans │             │ + Trans │
    └────┬────┘             └────┬────┘
         │                       │
         │   Fusion Point #1     │
         │   (if 1 ∈ stages)     │
         └───────────┬───────────┘
                     │
                     v
              ┌──────────────┐
              │ CSSA or GAFF │  ← Optional fusion
              │  (C=64)      │
              └──────┬───────┘
         ┌───────────┴───────────┐
         │                       │
         v                       v
    rgb_1 (64)              aux_1 (64)
         │                       │
         │                       │
┌────────────────────────────────────────────────────────┐
│ STAGE 2: Mid-Early Features (H/8 × W/8)               │
│ Channels: 128                                          │
└────────────────────────────────────────────────────────┘
         │                       │
         v                       v
    ┌─────────┐             ┌─────────┐
    │ Patch   │             │ Patch   │
    │ Merge   │             │ Merge   │
    │ + Trans │             │ + Trans │
    └────┬────┘             └────┬────┘
         │                       │
         │   Fusion Point #2     │
         │   (if 2 ∈ stages)     │
         └───────────┬───────────┘
                     │
                     v
              ┌──────────────┐
              │ CSSA or GAFF │  ← Optional fusion
              │  (C=128)     │
              └──────┬───────┘
         ┌───────────┴───────────┐
         │                       │
         v                       v
    rgb_2 (128)             aux_2 (128)
         │                       │
         │                       │
┌────────────────────────────────────────────────────────┐
│ STAGE 3: Mid-Late Features (H/16 × W/16)              │
│ Channels: 320                                          │
│ ★ MOST IMPORTANT FOR OBJECT DETECTION ★               │
└────────────────────────────────────────────────────────┘
         │                       │
         v                       v
    ┌─────────┐             ┌─────────┐
    │ Patch   │             │ Patch   │
    │ Merge   │             │ Merge   │
    │ + Trans │             │ + Trans │
    └────┬────┘             └────┬────┘
         │                       │
         │   Fusion Point #3     │
         │   (if 3 ∈ stages)     │  ← RECOMMENDED
         └───────────┬───────────┘
                     │
                     v
              ┌──────────────┐
              │ CSSA or GAFF │  ← Recommended fusion point
              │  (C=320)     │
              └──────┬───────┘
         ┌───────────┴───────────┐
         │                       │
         v                       v
    rgb_3 (320)             aux_3 (320)
         │                       │
         │                       │
┌────────────────────────────────────────────────────────┐
│ STAGE 4: Late Features (H/32 × W/32)                  │
│ Channels: 512                                          │
└────────────────────────────────────────────────────────┘
         │                       │
         v                       v
    ┌─────────┐             ┌─────────┐
    │ Patch   │             │ Patch   │
    │ Merge   │             │ Merge   │
    │ + Trans │             │ + Trans │
    └────┬────┘             └────┬────┘
         │                       │
         │   Fusion Point #4     │
         │   (if 4 ∈ stages)     │
         └───────────┬───────────┘
                     │
                     v
              ┌──────────────┐
              │ CSSA or GAFF │  ← Optional fusion
              │  (C=512)     │
              └──────┬───────┘
         ┌───────────┴───────────┐
         │                       │
         v                       v
    rgb_4 (512)             aux_4 (512)
         │                       │
         │                       │
         └───────────┬───────────┘
                     │
                     v
              ┌──────────────┐
              │    OUTPUT    │
              │ Multi-scale  │
              │   Features   │
              │ [f1,f2,f3,f4]│
              └──────────────┘
                     │
                     v
              ┌──────────────┐
              │ Faster R-CNN │
              │   Detector   │
              └──────────────┘


FUSION STAGE SELECTION GUIDELINES:

Stage 1 [H/4, C=64]:
  Pros: Fine spatial detail, early fusion
  Cons: Low-level features, may not help detection
  Recommended: ⚠️  Only for specific use cases

Stage 2 [H/8, C=128]:
  Pros: Good spatial detail, moderate semantics
  Cons: Still relatively low-level
  Recommended: ✓  Good for multi-stage fusion

Stage 3 [H/16, C=320]:
  Pros: High semantics, good resolution balance
  Cons: None significant
  Recommended: ★★★  HIGHLY RECOMMENDED (single stage)

Stage 4 [H/32, C=512]:
  Pros: Highest semantics, object-level features
  Cons: Lower spatial resolution
  Recommended: ✓  Good, especially with Stage 3
```

### 3.2 Example Fusion Configurations

```
CONFIGURATION A: Single-Stage Fusion at Stage 3
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Stage 1: Identity (no fusion)
Stage 2: Identity (no fusion)
Stage 3: CSSA or GAFF ← FUSION HERE
Stage 4: Identity (no fusion)

Advantages:
  + Minimal parameter overhead
  + Fusion at optimal semantic level
  + Fast training and inference
  + Recommended starting point


CONFIGURATION B: Two-Stage Fusion at [2,3]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Stage 1: Identity (no fusion)
Stage 2: CSSA or GAFF ← FUSION HERE
Stage 3: CSSA or GAFF ← FUSION HERE
Stage 4: Identity (no fusion)

Advantages:
  + Multi-level fusion
  + Captures both detail and semantics
  + Good accuracy potential
Disadvantages:
  - 2× fusion overhead
  - Slightly slower


CONFIGURATION C: Late Fusion at [3,4]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Stage 1: Identity (no fusion)
Stage 2: Identity (no fusion)
Stage 3: CSSA or GAFF ← FUSION HERE
Stage 4: CSSA or GAFF ← FUSION HERE

Advantages:
  + High-level semantic fusion
  + Object-level multimodal interaction
Disadvantages:
  - Stage 4 fusion expensive (C=512)
  - May lose fine-grained detail


CONFIGURATION D: All-Stage Fusion [1,2,3,4]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Stage 1: CSSA or GAFF ← FUSION
Stage 2: CSSA or GAFF ← FUSION
Stage 3: CSSA or GAFF ← FUSION
Stage 4: CSSA or GAFF ← FUSION

Advantages:
  + Maximum multimodal interaction
  + May achieve best accuracy
Disadvantages:
  - Highest parameter overhead
  - Slowest training/inference
  - Risk of overfitting
  - Only recommended for CSSA (lightweight)
```

---

## 4. Configuration Decision Trees

### 4.1 Fusion Mechanism Selection

```
FUSION MECHANISM DECISION TREE

START: Need multi-modal fusion for object detection
   │
   v
┌────────────────────────────────────────────┐
│ Q1: What is your deployment environment?   │
└────────────────┬───────────────────────────┘
                 │
        ┌────────┴────────┐
        │                 │
        v                 v
   ┌─────────┐       ┌─────────┐
   │  Edge   │       │  Cloud  │
   │ Device  │       │   GPU   │
   └────┬────┘       └────┬────┘
        │                 │
        │                 v
        │          ┌────────────────────────────────────┐
        │          │ Q2: Is accuracy more important     │
        │          │     than efficiency?               │
        │          └────────────┬───────────────────────┘
        │                       │
        │              ┌────────┴────────┐
        │              │                 │
        │              v                 v
        │         ┌─────────┐       ┌─────────┐
        │         │   YES   │       │   NO    │
        │         └────┬────┘       └────┬────┘
        │              │                 │
        │              v                 │
        │         ┌─────────┐            │
        │         │  GAFF   │            │
        │         │ (richer │            │
        │         │accuracy)│            │
        │         └────┬────┘            │
        │              │                 │
        v              v                 v
   ┌──────────────────────────────────────────┐
   │              USE CSSA                    │
   │    (lightweight, fast, efficient)        │
   └──────────────────┬───────────────────────┘
                      │
                      v
              ┌───────────────┐
              │ Configuration │
              │   Selection   │
              └───────┬───────┘
                      │
                      v
       ┌──────────────────────────────┐
       │ Q3: Which stages to fuse?    │
       └──────────────┬───────────────┘
                      │
         ┌────────────┼────────────┐
         │            │            │
         v            v            v
   ┌─────────┐  ┌─────────┐  ┌─────────┐
   │ Simple  │  │Balanced │  │Maximum  │
   │  Fast   │  │Accuracy │  │Accuracy │
   └────┬────┘  └────┬────┘  └────┬────┘
        │            │            │
        v            v            v
   Stage [3]    Stages [2,3]  Stages [1,2,3,4]
                or [3,4]      (CSSA only)

RECOMMENDATIONS:
━━━━━━━━━━━━━━━━
1. CSSA @ Stage [3]
   → Best starting point for most use cases
   → Minimal overhead, good accuracy

2. GAFF @ Stage [3]
   → Cloud deployment with accuracy priority
   → Can afford ~700K params overhead

3. CSSA @ Stages [2,3] or [3,4]
   → If single-stage CSSA insufficient
   → Still lightweight, multi-level fusion

4. GAFF @ Stages [3,4]
   → Maximum accuracy on cloud GPU
   → Accept ~2.5M params overhead
```

### 4.2 Hyperparameter Selection Tree

```
CSSA HYPERPARAMETER SELECTION

START: Using CSSA fusion
   │
   v
┌────────────────────────────────────────────┐
│ Q: What is modality complementarity?      │
│    (How different are RGB and Aux?)       │
└────────────────┬───────────────────────────┘
                 │
        ┌────────┴────────┬────────────┐
        │                 │            │
        v                 v            v
   ┌─────────┐       ┌─────────┐  ┌─────────┐
   │  High   │       │ Medium  │  │   Low   │
   │(very    │       │(moderate│  │(similar)│
   │different)│      │ differ) │  │         │
   └────┬────┘       └────┬────┘  └────┬────┘
        │                 │            │
        v                 v            v
   threshold=0.3      threshold=0.5  threshold=0.7
   (aggressive)       (balanced)     (conservative)
        │                 │            │
        └─────────────────┴────────────┘
                         │
                         v
                  ┌──────────────┐
                  │ Test stages: │
                  │ [3] or [2,3] │
                  └──────────────┘


GAFF HYPERPARAMETER SELECTION

START: Using GAFF fusion
   │
   v
┌────────────────────────────────────────────┐
│ Q1: Parameter budget?                      │
└────────────────┬───────────────────────────┘
                 │
        ┌────────┴────────┐
        │                 │
        v                 v
   ┌─────────┐       ┌─────────┐
   │  Tight  │       │Flexible │
   │ Budget  │       │ Budget  │
   └────┬────┘       └────┬────┘
        │                 │
        v                 v
   SE_reduction=8    SE_reduction=4
   (fewer params)    (more params)
        │                 │
        └─────────┬───────┘
                  │
                  v
       ┌────────────────────────────┐
       │ Q2: Is RGB dominant?       │
       │ (Is RGB much better        │
       │  than Aux modality?)       │
       └──────────┬─────────────────┘
                  │
         ┌────────┴────────┐
         │                 │
         v                 v
    ┌─────────┐       ┌─────────┐
    │   YES   │       │   NO    │
    │(RGB>>Aux)│      │(balanced)│
    └────┬────┘       └────┬────┘
         │                 │
         v                 v
   inter_shared=True  inter_shared=False
   (symmetric)        (asymmetric)
         │                 │
         └─────────┬───────┘
                   │
                   v
        ┌────────────────────────────┐
        │ Q3: Need max accuracy?     │
        └──────────┬─────────────────┘
                   │
          ┌────────┴────────┐
          │                 │
          v                 v
     ┌─────────┐       ┌─────────┐
     │   YES   │       │   NO    │
     └────┬────┘       └────┬────┘
          │                 │
          v                 v
   merge_bottleneck=T  merge_bottleneck=F
   (non-linear)        (direct)

COMMON CONFIGURATIONS:
━━━━━━━━━━━━━━━━━━━━━━
1. Default (balanced):
   SE_r=4, inter_shared=False, merge_bn=False

2. Lightweight:
   SE_r=8, inter_shared=True, merge_bn=False

3. Maximum Accuracy:
   SE_r=4, inter_shared=False, merge_bn=True

4. Fast Training:
   SE_r=8, inter_shared=True, merge_bn=False
```

---

## 5. Component Comparison Diagrams

### 5.1 Attention Mechanism Comparison

```
CHANNEL ATTENTION: ECA (CSSA) vs SE (GAFF)

ECA (Efficient Channel Attention):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input: (B, C, H, W)
   │
   v
┌──────────────────┐
│ Global Avg Pool  │  Squeeze
└────────┬─────────┘
         │ (B, C, 1, 1)
         v
┌──────────────────┐
│ Transpose        │  Prepare for 1D conv
│ (B, C, 1, 1) →   │
│ (B, 1, C, 1)     │
└────────┬─────────┘
         │
         v
┌──────────────────┐
│ Conv1d(k)        │  Local channel interactions
└────────┬─────────┘  k = kernel size (3-7)
         │
         v
┌──────────────────┐
│ Transpose back   │
└────────┬─────────┘
         │ (B, C, 1, 1)
         v
┌──────────────────┐
│ Sigmoid          │  Activation [0,1]
└────────┬─────────┘
         │
         v
      Output
   (B, C, 1, 1)

Parameters: C × k  (e.g., 320 × 3 = 960)
Receptive field: Local (kernel k)
Complexity: Low


SE (Squeeze-and-Excitation):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input: (B, C, H, W)
   │
   v
┌──────────────────┐
│ Global Avg Pool  │  Squeeze
└────────┬─────────┘
         │ (B, C, 1, 1) → (B, C)
         v
┌──────────────────┐
│ FC: C → C/r      │  Bottleneck (reduction)
└────────┬─────────┘  r = reduction ratio
         │ (B, C/r)
         v
┌──────────────────┐
│ ReLU             │  Non-linearity
└────────┬─────────┘
         │
         v
┌──────────────────┐
│ FC: C/r → C      │  Excitation
└────────┬─────────┘
         │ (B, C)
         v
┌──────────────────┐
│ Sigmoid          │  Activation [0,1]
└────────┬─────────┘
         │ (B, C, 1, 1)
         v
      Output
   (B, C, 1, 1)

Parameters: 2×C²/r  (e.g., 2×320²/4 = 51,200)
Receptive field: Global (all channels)
Complexity: Medium

COMPARISON:
┌──────────────┬────────────┬──────────────┐
│  Aspect      │    ECA     │      SE      │
├──────────────┼────────────┼──────────────┤
│ Parameters   │   C × k    │   2×C²/r     │
│ (C=320)      │   ~960     │   ~51,200    │
├──────────────┼────────────┼──────────────┤
│ Receptive    │   Local    │   Global     │
│ Field        │ (k channels)│ (all channels)│
├──────────────┼────────────┼──────────────┤
│ Non-linearity│  Sigmoid   │ ReLU+Sigmoid │
│              │   only     │              │
├──────────────┼────────────┼──────────────┤
│ Bottleneck   │    No      │     Yes      │
├──────────────┼────────────┼──────────────┤
│ Expressiveness│   Low     │     High     │
├──────────────┼────────────┼──────────────┤
│ Speed        │ Very Fast  │     Fast     │
└──────────────┴────────────┴──────────────┘
```

### 5.2 Cross-Modal Interaction Comparison

```
CSSA: Channel Switching (Hard Threshold)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

For each channel c:

RGB_attn[c]     Aux_attn[c]      Decision
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  0.8              0.3           Keep RGB[c]
  0.3              0.7           Swap: RGB[c]←Aux[c]
  0.4              0.4           Keep RGB[c] (default)
  0.6              0.2           Keep RGB[c]
  0.2              0.8           Swap: RGB[c]←Aux[c]

threshold = 0.5

Characteristics:
  ✓ Hard decisions (discrete)
  ✓ Explicit channel replacement
  ✓ Easy to interpret
  ✗ Non-differentiable threshold
  ✗ May discard useful information


GAFF: Guided Fusion (Soft Weighting)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

For each spatial location (h,w):

RGB_se   w_rgb←aux  Aux        Result
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 [0.5]  *  [0.3]  * [0.7]  =  0.5 + 0.21 = 0.71
 [0.8]  *  [0.1]  * [0.4]  =  0.8 + 0.04 = 0.84
 [0.3]  *  [0.6]  * [0.9]  =  0.3 + 0.54 = 0.84

RGB_guided = RGB_se + w_rgb←aux × Aux

Characteristics:
  ✓ Soft decisions (continuous)
  ✓ Fully differentiable
  ✓ Retains information from both
  ✗ Less interpretable
  ✗ Requires learned weights

VISUALIZATION:

CSSA (Binary Switch):
━━━━━━━━━━━━━━━━━━━━━━
Channel 0: [RGB] ← RGB confident
Channel 1: [Aux] ← Aux confident, swapped
Channel 2: [RGB] ← Both weak, default RGB
Channel 3: [RGB] ← RGB confident
Channel 4: [Aux] ← Aux confident, swapped

GAFF (Weighted Combination):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Channel 0: [0.7×RGB + 0.3×Aux] ← Mostly RGB
Channel 1: [0.4×RGB + 0.6×Aux] ← Balanced
Channel 2: [0.5×RGB + 0.5×Aux] ← Equal
Channel 3: [0.9×RGB + 0.1×Aux] ← Strongly RGB
Channel 4: [0.2×RGB + 0.8×Aux] ← Mostly Aux
```

---

## 6. Data Flow Visualizations

### 6.1 Complete Training Pipeline

```
TRAINING PIPELINE: FROM DATA TO DETECTION

┌────────────────────────────────────────────────────────────────────────┐
│ 1. DATA LOADING                                                        │
└────────────────────────────────────────────────────────────────────────┘

  RGB Images              Thermal Images          Event Images
  (H, W, 3)               (H, W, 1)               (H, W, 1)
      │                        │                       │
      └────────────────────────┴───────────────────────┘
                               │
                               v
                       ┌───────────────┐
                       │  Concatenate  │
                       │   Aux Stack   │
                       │   (H, W, 2)   │
                       └───────┬───────┘
                               │
                               v
                    Multi-modal Input Pair
                    [RGB, Aux]

┌────────────────────────────────────────────────────────────────────────┐
│ 2. PREPROCESSING & NORMALIZATION                                      │
└────────────────────────────────────────────────────────────────────────┘

      RGB (H,W,3)                     Aux (H,W,2)
           │                               │
           v                               v
     ┌─────────────┐               ┌─────────────┐
     │ Normalize   │               │ Normalize   │
     │ ImageNet    │               │ [0.5, 0.5]  │
     │ Statistics  │               │ [0.5, 0.5]  │
     └──────┬──────┘               └──────┬──────┘
            │                             │
            v                             v
     ┌─────────────┐               ┌─────────────┐
     │  Resize to  │               │  Resize to  │
     │ (H, W)      │               │ (H, W)      │
     └──────┬──────┘               └──────┬──────┘
            │                             │
            v                             v
      (B,3,H,W)                      (B,2,H,W)

┌────────────────────────────────────────────────────────────────────────┐
│ 3. MULTI-MODAL ENCODER (with fusion)                                  │
└────────────────────────────────────────────────────────────────────────┘

(B,3,H,W)                            (B,2,H,W)
    │                                     │
    └─────────────┬───────────────────────┘
                  │
                  v
        ┌─────────────────────┐
        │  Dual-Stream Encoder│
        │  with Fusion Points │
        │                     │
        │  Stage 1 (64)       │
        │    ↓   [fusion?]    │
        │  Stage 2 (128)      │
        │    ↓   [fusion?]    │
        │  Stage 3 (320) ★    │  ← Typical fusion point
        │    ↓   [fusion?]    │
        │  Stage 4 (512)      │
        │    ↓   [fusion?]    │
        └─────────┬───────────┘
                  │
                  v
          Multi-scale Features
          [f1, f2, f3, f4]
          (64, 128, 320, 512 channels)

┌────────────────────────────────────────────────────────────────────────┐
│ 4. FASTER R-CNN DETECTOR                                              │
└────────────────────────────────────────────────────────────────────────┘

     [f1, f2, f3, f4]
            │
            v
    ┌───────────────┐
    │     FPN       │  Feature Pyramid Network
    │   (optional)  │
    └───────┬───────┘
            │
            v
    ┌───────────────┐
    │     RPN       │  Region Proposal Network
    │ (Anchor Gen)  │
    └───────┬───────┘
            │
            v
      Proposals (ROIs)
            │
            v
    ┌───────────────┐
    │ ROI Align +   │
    │ ROI Heads     │
    └───────┬───────┘
            │
            ├───────────────┬──────────────┐
            v               v              v
    ┌─────────────┐  ┌──────────┐  ┌──────────┐
    │ Bounding    │  │  Class   │  │ Conf.    │
    │ Boxes       │  │  Labels  │  │ Scores   │
    └─────────────┘  └──────────┘  └──────────┘

┌────────────────────────────────────────────────────────────────────────┐
│ 5. LOSS COMPUTATION & BACKPROPAGATION                                 │
└────────────────────────────────────────────────────────────────────────┘

Predictions            Ground Truth
     │                      │
     └──────────┬───────────┘
                │
                v
        ┌───────────────┐
        │ Multi-task    │
        │ Loss Function │
        │               │
        │ - RPN Loss    │
        │ - Box Loss    │
        │ - Class Loss  │
        │ - Obj Loss    │
        └───────┬───────┘
                │
                v
          Total Loss
                │
                v
        ┌───────────────┐
        │  Backward()   │  Compute gradients
        └───────┬───────┘
                │
                v
        ┌───────────────┐
        │ Optimizer.    │  Update weights
        │   step()      │
        └───────────────┘
```

### 6.2 Inference Pipeline

```
INFERENCE PIPELINE: FROM IMAGE TO DETECTIONS

Input: RGB Image + Aux Image
   │
   v
┌────────────────┐
│ Preprocessing  │
│ - Resize       │
│ - Normalize    │
└───────┬────────┘
        │
        v
┌────────────────┐
│ Encoder with   │
│ Fusion         │
│ → Features     │
└───────┬────────┘
        │
        v
┌────────────────┐
│ Faster R-CNN   │
│ → Proposals    │
└───────┬────────┘
        │
        v
┌────────────────┐
│ NMS Filtering  │
│ → Final Boxes  │
└───────┬────────┘
        │
        v
┌────────────────┐
│ Detections:    │
│ - Boxes        │
│ - Classes      │
│ - Scores       │
└────────────────┘

SPEED COMPARISON (Inference Time):

Baseline (no fusion):          ████████████ 100%  (100ms)
+ CSSA @ Stage 3:              ████████████▒ 102%  (102ms)
+ CSSA @ Stages [2,3,4]:       █████████████ 105%  (105ms)
+ GAFF @ Stage 3:              ██████████████ 110%  (110ms)
+ GAFF @ Stages [3,4]:         ███████████████ 115%  (115ms)
```

---

## Appendix: Symbol Legend

```
SYMBOL GUIDE
━━━━━━━━━━━━

FLOWCHART SYMBOLS:
┌────────┐    Process / Operation
│        │
└────────┘

┌────────┐    Decision Point
│   ?    │
└────────┘

──────────    Data flow / connection

     │        Vertical flow
     v

═══════════   Important section
━━━━━━━━━━━   Header / separator

STATUS SYMBOLS:
✓  Recommended
★  Highly recommended
⚠️  Caution / warning
✗  Not recommended
✅ Complete
🔄 In progress
❌ Failed
⏸️  Pending

PARAMETER NOTATION:
B  = Batch size
C  = Number of channels
H  = Height
W  = Width
r  = Reduction ratio
k  = Kernel size
→  = Transformation
×  = Multiplication
+  = Addition

FUSION STAGES:
[1]      = Stage 1 only
[2,3]    = Stages 2 and 3
[1,2,3,4]= All stages
```

---

**Document End**

**Related Documentation:**
- `CSSA_ABLATION_GUIDE.md` - CSSA technical details
- `GAFF_ABLATION_GUIDE.md` - GAFF technical details
- `FUSION_MECHANISMS_COMPARISON.md` - Side-by-side comparison
- `EXPERIMENTAL_MATRIX.md` - All experiment configurations
- `EXPERIMENT_STATUS_DASHBOARD.md` - Current progress tracking
