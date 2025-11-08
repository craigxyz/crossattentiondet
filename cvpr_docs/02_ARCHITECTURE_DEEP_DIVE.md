# Architecture Deep Dive

**Complete Neural Network Architecture with Mathematical Formulations**

[← Back to Index](00_INDEX.md) | [← Previous: Executive Summary](01_EXECUTIVE_SUMMARY.md) | [Next: Dataset & Modalities →](03_DATASET_AND_MODALITIES.md)

---

## Table of Contents
1. [High-Level Architecture](#high-level-architecture)
2. [Dual-Stream Encoder: RGBXTransformer](#dual-stream-encoder-rgbxtransformer)
3. [Fusion Mechanisms](#fusion-mechanisms)
   - [Baseline: FRM + FFM](#baseline-frm--ffm)
   - [CSSA](#cssa-channel-switching-and-spatial-attention)
   - [GAFF](#gaff-guided-attentive-feature-fusion)
4. [Feature Pyramid Network (FPN)](#feature-pyramid-network-fpn)
5. [Detection Head: Faster R-CNN](#detection-head-faster-r-cnn)
6. [Backbone Variants](#backbone-variants)
7. [Data Flow & Dimensions](#data-flow--dimensions)

---

## High-Level Architecture

### Complete Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                   Input: 5-Channel Image                    │
│              (B, 5, H, W) - RGB + Thermal + Event           │
└───────────────────┬─────────────────────────────────────────┘
                    │
                    ├─────────────────────────────┐
                    ├──────────────┐              │
                    │              │              │
            RGB (B, 3, H, W)   Thermal (B, 1, H, W)  Event (B, 1, H, W)
                    │              └──────┬───────┘
                    │                     │
                    │              Aux (B, 2, H, W)
                    │                     │
                    ├─────────────────────┴─────────────────────┐
                    │     Dual-Stream Transformer Encoder       │
                    │         (RGBXTransformer - MiT)           │
                    │                                           │
                    │   ┌─────────────────────────────────┐    │
                    │   │ Stage 1: Patch Embed + Blocks   │    │
                    │   │   RGB Stream: (B,64,56,56)      │    │
                    │   │   Aux Stream: (B,64,56,56)      │    │
                    │   └──────────┬──────────────────────┘    │
                    │              ↓ [Optional Fusion]          │
                    │   ┌──────────────────────────────────┐   │
                    │   │ Stage 2: Patch Embed + Blocks    │   │
                    │   │   RGB Stream: (B,128,28,28)      │   │
                    │   │   Aux Stream: (B,128,28,28)      │   │
                    │   └──────────┬───────────────────────┘   │
                    │              ↓ [Optional Fusion]          │
                    │   ┌──────────────────────────────────┐   │
                    │   │ Stage 3: Patch Embed + Blocks    │   │
                    │   │   RGB Stream: (B,320,14,14)      │   │
                    │   │   Aux Stream: (B,320,14,14)      │   │
                    │   └──────────┬───────────────────────┘   │
                    │              ↓ [Optional Fusion]          │
                    │   ┌──────────────────────────────────┐   │
                    │   │ Stage 4: Patch Embed + Blocks    │   │
                    │   │   RGB Stream: (B,512,7,7)        │   │
                    │   │   Aux Stream: (B,512,7,7)        │   │
                    │   └──────────┬───────────────────────┘   │
                    │              ↓ [Optional Fusion]          │
                    └──────────────┴───────────────────────────┘
                                   │
                            4 Feature Maps
                    [f1, f2, f3, f4] from encoder
                                   │
                    ┌──────────────┴────────────────┐
                    │  Feature Pyramid Network      │
                    │  (FPN with lateral connections)│
                    └──────────────┬────────────────┘
                                   │
                         5 Pyramid Levels
                    {P0, P1, P2, P3, Pool}
                    All 256 channels, multi-scale
                                   │
                    ┌──────────────┴────────────────┐
                    │    Faster R-CNN Head          │
                    │                               │
                    │  ┌─────────────────────────┐  │
                    │  │ RPN (Region Proposals)  │  │
                    │  │ - Anchor Generation     │  │
                    │  │ - Objectness Scores     │  │
                    │  │ - Box Regression        │  │
                    │  └──────────┬──────────────┘  │
                    │             │ ~2000 proposals │
                    │  ┌──────────┴──────────────┐  │
                    │  │ RoI Pooling (7×7)       │  │
                    │  └──────────┬──────────────┘  │
                    │             │                 │
                    │  ┌──────────┴──────────────┐  │
                    │  │ Box Head                │  │
                    │  │ - Classification        │  │
                    │  │ - Box Refinement        │  │
                    │  └──────────┬──────────────┘  │
                    └─────────────┴─────────────────┘
                                  │
                    ┌─────────────┴──────────────┐
                    │  Detected Objects          │
                    │  - Bounding boxes          │
                    │  - Class labels            │
                    │  - Confidence scores       │
                    └────────────────────────────┘
```

**File:** `crossattentiondet/models/backbone.py`, `crossattentiondet/models/encoder.py`

---

## Dual-Stream Encoder: RGBXTransformer

**File:** `crossattentiondet/models/encoder.py` (267 lines)

### Design Philosophy
- **Dual-Stream:** Separate processing for RGB and auxiliary (Thermal+Event) modalities
- **Hierarchical:** 4 stages with progressively increasing semantics
- **Efficient:** Mix Transformer (MiT) with spatial reduction for computational efficiency
- **Flexible:** Support for stage-wise fusion insertion

### Architecture Components

#### 1. Patch Embedding

**Purpose:** Convert image to token sequence with spatial downsampling.

**Implementation:**
```python
class OverlapPatchEmbed(nn.Module):
    """
    Overlapping patch embedding using Conv2d.
    Unlike ViT (non-overlapping 16×16 patches), uses overlapping patches
    for better local context capture.
    """
    def __init__(self, patch_size, stride, in_chans, embed_dim):
        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=patch_size // 2
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, embed_dim, H', W')
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, H'*W', embed_dim)
        x = self.norm(x)
        return x, H, W
```

**Per-Stage Configuration (mit_b1):**

| Stage | Patch Size | Stride | Input Channels | Output Dim | Input Res | Output Res |
|-------|------------|--------|----------------|------------|-----------|------------|
| 1 | 7×7 | 4 | 3 (RGB) or 2 (Aux) | 64 | 224×224 | 56×56 |
| 2 | 3×3 | 2 | 64 | 128 | 56×56 | 28×28 |
| 3 | 3×3 | 2 | 128 | 320 | 28×28 | 14×14 |
| 4 | 3×3 | 2 | 320 | 512 | 14×14 | 7×7 |

**Rationale:** Overlapping patches (vs. ViT's non-overlapping) provide better local spatial context, critical for dense prediction tasks like object detection.

#### 2. Transformer Blocks

**Purpose:** Extract hierarchical features via efficient self-attention.

**Block Structure:**
```python
class Block(nn.Module):
    """
    Standard Transformer block with:
    1. Efficient Self-Attention (with spatial reduction)
    2. Feed-Forward Network (FFN)
    3. Residual connections
    4. Layer normalization
    """
    def __init__(self, dim, num_heads, mlp_ratio=4, sr_ratio=1, drop_path=0.):
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, sr_ratio)
        self.drop_path = DropPath(drop_path)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, hidden_features=dim * mlp_ratio)

    def forward(self, x, H, W):
        # Multi-head self-attention with residual
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        # Feed-forward network with residual
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
```

#### 3. Efficient Self-Attention (with Spatial Reduction)

**Innovation:** Reduces computational cost from O(N²) to O(N²/R²) where R is the reduction ratio.

**Mathematical Formulation:**
```
Standard Attention (O(N²)):
  Q = Linear_Q(X)           # (B, N, d)
  K = Linear_K(X)           # (B, N, d)
  V = Linear_V(X)           # (B, N, d)
  Attention = Softmax(QK^T / √d) · V

Efficient Attention (O(N²/R²)):
  Q = Linear_Q(X)           # (B, N, d)
  X_reduced = SpatialReduction(X, R)  # (B, N/R², d)
  K = Linear_K(X_reduced)   # (B, N/R², d)
  V = Linear_V(X_reduced)   # (B, N/R², d)
  Attention = Softmax(QK^T / √d) · V  # (B, N, d)
```

**Implementation:**
```python
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, sr_ratio=1):
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape

        # Query (no reduction)
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads)
        q = q.permute(0, 2, 1, 3)  # (B, heads, N, head_dim)

        # Key, Value (with spatial reduction if sr_ratio > 1)
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)  # (B, N/R², C)
            x_ = self.norm(x_)
        else:
            x_ = x

        kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads)
        kv = kv.permute(2, 0, 3, 1, 4)  # (2, B, heads, N/R², head_dim)
        k, v = kv[0], kv[1]

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, N, N/R²)
        attn = attn.softmax(dim=-1)

        # Output
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return x
```

**Spatial Reduction Ratios by Stage (mit_b1):**

| Stage | SR Ratio | Input Res | Reduced Res | Tokens (N) | Reduced Tokens (N/R²) | Complexity Reduction |
|-------|----------|-----------|-------------|------------|----------------------|----------------------|
| 1 | 8 | 56×56 | 7×7 | 3,136 | 49 | 64× |
| 2 | 4 | 28×28 | 7×7 | 784 | 49 | 16× |
| 3 | 2 | 14×14 | 7×7 | 196 | 49 | 4× |
| 4 | 1 | 7×7 | 7×7 | 49 | 49 | 1× (no reduction) |

**Rationale:** Early stages have high spatial resolution (many tokens) → aggressive reduction. Late stages have fewer tokens → less reduction needed.

#### 4. Feed-Forward Network (FFN)

**Standard MLP with expansion:**
```python
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)  # Depthwise conv for locality
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)  # Add spatial inductive bias
        x = self.act(x)
        x = self.fc2(x)
        return x
```

**Expansion ratio:** 4× (e.g., 320 → 1280 → 320 for stage 3)

#### 5. Dual-Stream Forward Pass

**File:** `crossattentiondet/models/encoder.py:110-266`

```python
class RGBXTransformer(nn.Module):
    def forward(self, rgb_images, x_images):
        """
        Args:
            rgb_images: (B, 3, H, W) - RGB input
            x_images: (B, 2, H, W) - Thermal + Event input

        Returns:
            outs: List[Tensor] - 4 fused feature maps
        """
        outs = []

        # Stage 1
        x_rgb, H, W = self.patch_embed1_rgb(rgb_images)  # (B, N1, 64)
        x_x, _, _ = self.patch_embed1_x(x_images)        # (B, N1, 64)

        for blk in self.block1:
            x_rgb = blk(x_rgb, H, W)
            x_x = blk(x_x, H, W)

        x_rgb = self.norm1(x_rgb).reshape(B, H, W, -1).permute(0, 3, 1, 2)  # (B, 64, 56, 56)
        x_x = self.norm1(x_x).reshape(B, H, W, -1).permute(0, 3, 1, 2)      # (B, 64, 56, 56)

        # Fusion (FRM + FFM)
        x_rgb, x_x = self.FRMs[0](x_rgb, x_x)
        x = self.FFMs[0](x_rgb, x_x)  # (B, 64, 56, 56)
        outs.append(x)

        # Stage 2
        x_rgb, H, W = self.patch_embed2_rgb(x_rgb)  # (B, N2, 128)
        x_x, _, _ = self.patch_embed2_x(x_x)        # (B, N2, 128)

        for blk in self.block2:
            x_rgb = blk(x_rgb, H, W)
            x_x = blk(x_x, H, W)

        x_rgb = self.norm2(x_rgb).reshape(B, H, W, -1).permute(0, 3, 1, 2)  # (B, 128, 28, 28)
        x_x = self.norm2(x_x).reshape(B, H, W, -1).permute(0, 3, 1, 2)      # (B, 128, 28, 28)

        x_rgb, x_x = self.FRMs[1](x_rgb, x_x)
        x = self.FFMs[1](x_rgb, x_x)  # (B, 128, 28, 28)
        outs.append(x)

        # Stage 3
        x_rgb, H, W = self.patch_embed3_rgb(x_rgb)
        x_x, _, _ = self.patch_embed3_x(x_x)

        for blk in self.block3:
            x_rgb = blk(x_rgb, H, W)
            x_x = blk(x_x, H, W)

        x_rgb = self.norm3(x_rgb).reshape(B, H, W, -1).permute(0, 3, 1, 2)  # (B, 320, 14, 14)
        x_x = self.norm3(x_x).reshape(B, H, W, -1).permute(0, 3, 1, 2)      # (B, 320, 14, 14)

        x_rgb, x_x = self.FRMs[2](x_rgb, x_x)
        x = self.FFMs[2](x_rgb, x_x)  # (B, 320, 14, 14)
        outs.append(x)

        # Stage 4
        x_rgb, H, W = self.patch_embed4_rgb(x_rgb)
        x_x, _, _ = self.patch_embed4_x(x_x)

        for blk in self.block4:
            x_rgb = blk(x_rgb, H, W)
            x_x = blk(x_x, H, W)

        x_rgb = self.norm4(x_rgb).reshape(B, H, W, -1).permute(0, 3, 1, 2)  # (B, 512, 7, 7)
        x_x = self.norm4(x_x).reshape(B, H, W, -1).permute(0, 3, 1, 2)      # (B, 512, 7, 7)

        x_rgb, x_x = self.FRMs[3](x_rgb, x_x)
        x = self.FFMs[3](x_rgb, x_x)  # (B, 512, 7, 7)
        outs.append(x)

        return outs  # [f1, f2, f3, f4]
```

---

## Fusion Mechanisms

### Baseline: FRM + FFM

**File:** `crossattentiondet/models/fusion.py` (229 lines)

**Design:** Two-stage fusion: (1) Feature Rectify Module (FRM) refines each modality using channel + spatial attention, (2) Feature Fusion Module (FFM) performs cross-attention fusion.

#### FRM: Feature Rectify Module

**Purpose:** Adaptively weight each modality's contribution before fusion.

**Architecture:**
```
x_rgb (B,C,H,W)     x_aux (B,C,H,W)
    │                   │
    ├───────────────────┤
    │                   │
    ▼                   ▼
┌───────────────────────────┐
│   ChannelWeights          │
│   (Global pooling + MLP)  │
└───────┬───────────────────┘
        │
    [w_rgb_c, w_aux_c]  # (2, B, C, 1, 1)
        │
┌───────┴───────────────────┐
│   SpatialWeights          │
│   (Conv on concat feats)  │
└───────┬───────────────────┘
        │
    [w_rgb_s, w_aux_s]  # (2, B, 1, H, W)
        │
        ▼
    Weighted rectification:
    x_rgb' = x_rgb + λ_c * w_aux_c * x_aux + λ_s * w_aux_s * x_aux
    x_aux' = x_aux + λ_c * w_rgb_c * x_rgb + λ_s * w_rgb_s * x_rgb
```

**Implementation:**
```python
class ChannelWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        hidden = max(dim // reduction, 16)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim * 2, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, dim * 2, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        """
        Returns: (2, B, C, 1, 1) channel attention weights
        """
        concat = torch.cat([x1, x2], dim=1)  # (B, 2C, H, W)

        avg = self.avg_pool(concat)  # (B, 2C, 1, 1)
        max_val = self.max_pool(concat)  # (B, 2C, 1, 1)

        weights = self.mlp(avg + max_val)  # (B, 2C, 1, 1)
        weights = self.sigmoid(weights)

        w1, w2 = torch.chunk(weights, 2, dim=1)  # Each (B, C, 1, 1)
        return torch.stack([w1, w2], dim=0)  # (2, B, C, 1, 1)

class SpatialWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        hidden = max(dim // reduction, 16)
        self.conv = nn.Sequential(
            nn.Conv2d(dim * 2, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 2, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        """
        Returns: (2, B, 1, H, W) spatial attention weights
        """
        concat = torch.cat([x1, x2], dim=1)  # (B, 2C, H, W)
        weights = self.conv(concat)  # (B, 2, H, W)
        w1, w2 = weights[:, 0:1], weights[:, 1:2]  # Each (B, 1, H, W)
        return torch.stack([w1, w2], dim=0)  # (2, B, 1, H, W)

class FRM(nn.Module):
    def __init__(self, dim, reduction=1):
        self.channel_weights = ChannelWeights(dim, reduction)
        self.spatial_weights = SpatialWeights(dim, reduction)
        self.lambda_c = 0.5  # Channel weight scale
        self.lambda_s = 0.5  # Spatial weight scale

    def forward(self, x_rgb, x_aux):
        # Get attention weights
        c_weights = self.channel_weights(x_rgb, x_aux)  # (2, B, C, 1, 1)
        s_weights = self.spatial_weights(x_rgb, x_aux)  # (2, B, 1, H, W)

        # Apply weighted rectification
        x_rgb_out = x_rgb + self.lambda_c * c_weights[1] * x_aux + self.lambda_s * s_weights[1] * x_aux
        x_aux_out = x_aux + self.lambda_c * c_weights[0] * x_rgb + self.lambda_s * s_weights[0] * x_rgb

        return x_rgb_out, x_aux_out
```

**Parameters (C=320, reduction=1):**
- ChannelWeights: 2C × hidden + hidden × 2C ≈ 4C²/reduction = 409,600
- SpatialWeights: 2C × hidden + hidden × 2 ≈ 2C²/reduction + 2×hidden = 204,832
- **Total: ~614K params per stage**

#### FFM: Feature Fusion Module

**Purpose:** Cross-attention-based fusion of rectified features.

**Architecture:**
```
x_rgb (B,C,H,W)     x_aux (B,C,H,W)
    │                   │
    │ Flatten           │ Flatten
    ▼                   ▼
(B,N,C)             (B,N,C)
    │                   │
    ├───────────────────┤
    │  CrossPath        │
    │  (Cross-Attn)     │
    └───────┬───────────┘
            │
        (B,N,2C)
            │
    ┌───────┴───────┐
    │ ChannelEmbed  │
    │ (DW Sep Conv) │
    └───────┬───────┘
            │
        (B,C,H,W) fused
```

**Implementation:**
```python
class CrossPath(nn.Module):
    """Dual-path cross-attention"""
    def __init__(self, dim, reduction=1):
        self.query1 = nn.Linear(dim, dim // reduction)
        self.key1 = nn.Linear(dim, dim // reduction)
        self.query2 = nn.Linear(dim, dim // reduction)
        self.key2 = nn.Linear(dim, dim // reduction)

    def forward(self, x1, x2):
        """
        x1, x2: (B, N, C)
        Returns: (x1_att, x2_att) after cross-attention
        """
        # Path 1: x1 attends to x2
        q1 = self.query1(x1)  # (B, N, C/r)
        k2 = self.key2(x2)    # (B, N, C/r)
        attn1 = F.softmax(q1 @ k2.transpose(-2, -1) / (q1.size(-1) ** 0.5), dim=-1)
        x1_att = x1 + attn1 @ x2  # Residual cross-attention

        # Path 2: x2 attends to x1
        q2 = self.query2(x2)
        k1 = self.key1(x1)
        attn2 = F.softmax(q2 @ k1.transpose(-2, -1) / (q2.size(-1) ** 0.5), dim=-1)
        x2_att = x2 + attn2 @ x1

        return x1_att, x2_att

class ChannelEmbed(nn.Module):
    """Depthwise separable conv for channel embedding"""
    def __init__(self, in_channels, out_channels):
        self.conv_dw = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels)
        self.conv_pw = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        x = self.conv_dw(x)
        x = self.conv_pw(x)
        return x

class FFM(nn.Module):
    def __init__(self, dim, reduction=1):
        self.cross_path = CrossPath(dim, reduction)
        self.channel_embed = ChannelEmbed(dim * 2, dim)

    def forward(self, x_rgb, x_aux):
        B, C, H, W = x_rgb.shape

        # Flatten spatial
        x_rgb_flat = x_rgb.flatten(2).permute(0, 2, 1)  # (B, N, C)
        x_aux_flat = x_aux.flatten(2).permute(0, 2, 1)  # (B, N, C)

        # Cross-attention
        x_rgb_att, x_aux_att = self.cross_path(x_rgb_flat, x_aux_flat)

        # Concatenate and embed
        merge = torch.cat([x_rgb_att, x_aux_att], dim=-1)  # (B, N, 2C)
        fused = self.channel_embed(merge, H, W)  # (B, C, H, W)

        return fused
```

**Parameters (C=320, reduction=1):**
- CrossPath: 4 × (C × C/r) = 4C²/r = 409,600
- ChannelEmbed: C×3×3 + 2C×C = 9C + 2C² = 207,680
- **Total: ~617K params per stage**

**FRM + FFM Total: ~1.23M params per stage** (for C=320)

---

### CSSA: Channel Switching and Spatial Attention

**Reference:** Cao et al., "Multimodal Object Detection by Channel Switching and Spatial Attention", CVPR 2023 PBVS Workshop

**File:** `crossattentiondet/ablations/fusion/cssa.py` (173 lines)

**Design Philosophy:** Ultra-lightweight fusion through hard channel switching + soft spatial attention.

**Architecture:**
```
x_rgb (B,C,H,W)            x_aux (B,C,H,W)
    │                           │
    ▼                           ▼
┌─────────┐               ┌─────────┐
│ ECABlock│               │ ECABlock│
└────┬────┘               └────┬────┘
     │                          │
  w_rgb (B,C,1,1)          w_aux (B,C,1,1)
     │                          │
     ├──────────────────────────┤
     │  ChannelSwitching        │
     │  (threshold-based)       │
     └──────────┬───────────────┘
                │
     rgb_switched, aux_switched
                │
     ┌──────────┴───────────┐
     │  SpatialAttention    │
     │  (avg+max pooling)   │
     └──────────┬───────────┘
                │
           fused (B,C,H,W)
```

#### Component 1: ECABlock (Efficient Channel Attention)

**Innovation:** 1D convolution on channel dimension (vs. SE's FC layers).

**Mathematical Formulation:**
```
Gap = GlobalAvgPool(X)  # (B, C, 1, 1)
Gap_1d = Gap.squeeze().transpose()  # (B, 1, C)
Attn = sigmoid(Conv1d_k(Gap_1d))  # (B, 1, C), k = adaptive kernel size
Attn = Attn.transpose().unsqueeze()  # (B, C, 1, 1)
Output = X * Attn
```

**Adaptive Kernel Size:**
```python
def get_kernel_size(channels):
    # k = |log2(C) / γ + b / γ|_odd
    # γ = 2, b = 1 (empirical)
    t = int(abs((math.log2(channels) + 1) / 2))
    k = t if t % 2 else t + 1  # Ensure odd
    return max(k, 3)  # Minimum 3
```

**Example:** C=320 → k=5

**Implementation:**
```python
class ECABlock(nn.Module):
    def __init__(self, kernel_size=3):
        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.f = nn.Conv1d(1, 1, kernel_size=kernel_size,
                          padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.shape

        # Global pooling
        gap = self.GAP(x)  # (B, C, 1, 1)

        # 1D conv on channel dimension
        gap = gap.squeeze(-1).transpose(-1, -2)  # (B, 1, C)
        attn = self.f(gap)  # (B, 1, C)
        attn = attn.transpose(-1, -2).unsqueeze(-1)  # (B, C, 1, 1)
        attn = self.sigmoid(attn)

        return attn  # Channel attention weights
```

**Parameters:** C × k = 320 × 5 = 1,600

#### Component 2: ChannelSwitching

**Purpose:** Hard threshold-based channel replacement.

**Algorithm:**
```python
def channel_switching(x, x_prime, w, threshold):
    """
    Args:
        x: (B, C, H, W) - primary modality
        x_prime: (B, C, H, W) - alternative modality
        w: (B, C, 1, 1) - channel attention for x
        threshold: float - switching threshold

    Returns:
        x_switched: (B, C, H, W)
    """
    mask = (w < threshold)  # (B, C, 1, 1) boolean
    x_switched = torch.where(mask, x_prime, x)
    return x_switched
```

**Threshold Interpretation:**
- **threshold = 0.3 (aggressive):** Switch if attention < 0.3 (swap ~70% of low-attention channels)
- **threshold = 0.5 (balanced):** Switch if attention < 0.5 (swap ~50% of channels)
- **threshold = 0.7 (conservative):** Switch if attention < 0.7 (swap ~30% of low-attention channels)

**Parameters:** 0 (threshold is hyperparameter, not learned)

#### Component 3: SpatialAttention

**Purpose:** Soft spatial weighting for fusion.

**Mathematical Formulation:**
```
Concat = [RGB_switched, Aux_switched]  # (B, 2C, H, W)

# Channel pooling
Avg = Mean(Concat, dim=1)  # (B, 1, H, W)
Max = Max(Concat, dim=1)   # (B, 1, H, W)

# 7×7 conv for spatial attention
SpatialAttn = sigmoid(Conv2d_7x7([Avg, Max]))  # (B, 1, H, W)

# Weighted fusion
Fused = SpatialAttn * RGB_switched + (1 - SpatialAttn) * Aux_switched
```

**Implementation:**
```python
class SpatialAttention(nn.Module):
    def __init__(self):
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, rgb_feats, ir_feats):
        B, C, H, W = rgb_feats.shape
        x_cat = torch.cat((rgb_feats, ir_feats), dim=1)  # (B, 2C, H, W)

        # Channel pooling
        avg_pool = torch.mean(x_cat, dim=1, keepdim=True)  # (B, 1, H, W)
        max_pool, _ = torch.max(x_cat, dim=1, keepdim=True)  # (B, 1, H, W)

        # Spatial attention
        spatial_attn = self.sigmoid(self.conv(torch.cat([avg_pool, max_pool], dim=1)))

        # Weighted fusion
        x_fused = spatial_attn * rgb_feats + (1 - spatial_attn) * ir_feats

        return x_fused
```

**Parameters:** 2 × 1 × 7 × 7 = 98

#### CSSA Complete Forward Pass

```python
class CSSABlock(nn.Module):
    def __init__(self, kernel_size=3, switching_thresh=0.5):
        self.eca_rgb = ECABlock(kernel_size)
        self.eca_aux = ECABlock(kernel_size)
        self.cs = ChannelSwitching(switching_thresh)
        self.sa = SpatialAttention()

    def forward(self, x_rgb, x_aux):
        # Step 1: Channel attention
        rgb_w = self.eca_rgb(x_rgb)  # (B, C, 1, 1)
        aux_w = self.eca_aux(x_aux)  # (B, C, 1, 1)

        # Step 2: Channel switching
        rgb_switched = self.cs(x_rgb, x_aux, rgb_w)
        aux_switched = self.cs(x_aux, x_rgb, aux_w)

        # Step 3: Spatial attention and fusion
        fused = self.sa(rgb_switched, aux_switched)

        return fused
```

**Total Parameters (C=320, k=5):**
- ECA_RGB: 1,600
- ECA_Aux: 1,600
- ChannelSwitching: 0
- SpatialAttention: 98
- **Total: 3,298 params (~0.003% of model!)**

**Parameter Efficiency vs. Baseline FRM+FFM:**
- CSSA: 3,298 params
- FRM+FFM: 1,231,000 params
- **CSSA is 373× lighter than baseline!**

---

### GAFF: Guided Attentive Feature Fusion

**Reference:** Zhang et al., "Guided Attentive Feature Fusion for Multispectral Pedestrian Detection", WACV 2021

**File:** `crossattentiondet/ablations/fusion/gaff.py` (249 lines)

**Design Philosophy:** Rich cross-modal interactions through multi-level attention (intra + inter).

**Architecture:**
```
x_rgb (B,C,H,W)        x_aux (B,C,H,W)
    │                       │
    ▼                       ▼
┌─────────┐           ┌─────────┐
│ SEBlock │           │ SEBlock │
└────┬────┘           └────┬────┘
     │                     │
  x̂_rgb               x̂_aux
     │                     │
     ├─────────────────────┤
     │ InterModalityAttn   │
     └──────┬──────────────┘
            │
     [w_rgb←aux, w_aux←rgb]
            │
     ┌──────┴──────┐
     │ Guided Fusion│
     │ (Residual)   │
     └──────┬──────┘
            │
     [x̂_rgb_guided, x̂_aux_guided]
            │
     ┌──────┴──────┐
     │ Merge Layer │
     │ (Conv+BN)   │
     └──────┬──────┘
            │
        fused (B,C,H,W)
```

#### Component 1: SEBlock (Squeeze-and-Excitation)

**Purpose:** Intra-modality channel recalibration.

**Mathematical Formulation:**
```
Squeeze: z = GlobalAvgPool(X)  # (B, C)
Excitation: s = sigmoid(W2 · ReLU(W1 · z))  # (B, C)
            W1: (C, C/r), W2: (C/r, C)
Scale: Output = X * s.unsqueeze(-1).unsqueeze(-1)
```

**Implementation:**
```python
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        self.gap = nn.AdaptiveAvgPool2d(1)
        reduced = max(channels // reduction, 1)

        self.fc1 = nn.Linear(channels, reduced, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(reduced, channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, _, _ = x.size()

        # Squeeze
        y = self.gap(x).view(B, C)

        # Excitation (bottleneck)
        y = self.fc1(y)  # (B, C/r)
        y = self.relu(y)
        y = self.fc2(y)  # (B, C)
        y = self.sigmoid(y)

        # Scale
        y = y.view(B, C, 1, 1)
        return x * y
```

**Parameters (C=320, r=4):**
- fc1: C × (C/r) = 320 × 80 = 25,600
- fc2: (C/r) × C = 80 × 320 = 25,600
- **Total: 51,200 params per SE block**
- **Two SE blocks (RGB + Aux): 102,400 params**

#### Component 2: InterModalityAttention

**Purpose:** Cross-modal attention weighting.

**Design Choice 1: Separate Convolutions (inter_shared=False)**
```python
concat = cat([x_rgb, x_aux], dim=1)  # (B, 2C, H, W)

w_rgb_from_aux = Conv2d(2C → C)(concat)  # (B, C, H, W)
w_aux_from_rgb = Conv2d(2C → C)(concat)  # (B, C, H, W)

w_rgb_from_aux = sigmoid(w_rgb_from_aux)
w_aux_from_rgb = sigmoid(w_aux_from_rgb)
```

**Parameters (separate):** 2 × (2C × C × 1 × 1) = 4C² = 409,600 (for C=320)

**Design Choice 2: Shared Convolution (inter_shared=True)**
```python
concat = cat([x_rgb, x_aux], dim=1)  # (B, 2C, H, W)

attn = Conv2d(2C → 2C)(concat)  # (B, 2C, H, W)
w_rgb_from_aux, w_aux_from_rgb = chunk(attn, 2, dim=1)  # Each (B, C, H, W)

w_rgb_from_aux = sigmoid(w_rgb_from_aux)
w_aux_from_rgb = sigmoid(w_aux_from_rgb)
```

**Parameters (shared):** 2C × 2C × 1 × 1 = 4C² = 409,600 (for C=320, same as separate!)

**Implementation:**
```python
class InterModalityAttention(nn.Module):
    def __init__(self, channels, shared=False):
        self.shared = shared

        if shared:
            self.conv = nn.Conv2d(channels * 2, channels * 2, kernel_size=1, bias=False)
        else:
            self.conv_rgb = nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False)
            self.conv_aux = nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x_rgb, x_aux):
        concat = torch.cat([x_rgb, x_aux], dim=1)  # (B, 2C, H, W)

        if self.shared:
            attn = self.conv(concat)  # (B, 2C, H, W)
            w_rgb_from_aux, w_aux_from_rgb = torch.chunk(attn, 2, dim=1)
        else:
            w_rgb_from_aux = self.conv_rgb(concat)  # (B, C, H, W)
            w_aux_from_rgb = self.conv_aux(concat)  # (B, C, H, W)

        return self.sigmoid(w_rgb_from_aux), self.sigmoid(w_aux_from_rgb)
```

**Parameters:** 409,600 (regardless of shared choice for C=320)

#### Component 3: Guided Fusion (Residual)

**Purpose:** Add cross-modal guidance to SE-enhanced features.

```python
def guided_fusion(x_rgb_se, x_aux_se, w_rgb_from_aux, w_aux_from_rgb):
    x_rgb_guided = x_rgb_se + w_rgb_from_aux * x_aux_se  # Residual
    x_aux_guided = x_aux_se + w_aux_from_rgb * x_rgb_se  # Residual
    return x_rgb_guided, x_aux_guided
```

**Parameters:** 0 (uses weights from InterModalityAttention)

#### Component 4: Merge Layer

**Design Choice 1: Direct Merge (merge_bottleneck=False)**
```python
concat = cat([x_rgb_guided, x_aux_guided], dim=1)  # (B, 2C, H, W)
out = Conv2d(2C → out_C, 1×1)(concat)
out = BatchNorm2d(out_C)(out)
```

**Parameters:** 2C × out_C + out_C = 2C² + 2C = 205,440 (for C=out_C=320)

**Design Choice 2: Bottleneck Merge (merge_bottleneck=True)**
```python
concat = cat([x_rgb_guided, x_aux_guided], dim=1)  # (B, 2C, H, W)
out = Conv2d(2C → C, 1×1)(concat)
out = BatchNorm2d(C)(out)
out = ReLU()(out)
out = Conv2d(C → out_C, 1×1)(out)
out = BatchNorm2d(out_C)(out)
```

**Parameters:** 2C×C + C + C×out_C + out_C = 3C² + 4C = 308,480 (for C=out_C=320)

**Implementation:**
```python
class GAFFBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, se_reduction=4,
                 inter_shared=False, merge_bottleneck=False):
        out_channels = out_channels or in_channels

        # Intra-modality SE
        self.se_rgb = SEBlock(in_channels, se_reduction)
        self.se_aux = SEBlock(in_channels, se_reduction)

        # Inter-modality attention
        self.inter_attn = InterModalityAttention(in_channels, inter_shared)

        # Merge layer
        if merge_bottleneck:
            self.merge_conv1 = nn.Conv2d(in_channels * 2, in_channels, 1, bias=False)
            self.merge_bn1 = nn.BatchNorm2d(in_channels)
            self.merge_conv2 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
            self.merge_bn2 = nn.BatchNorm2d(out_channels)
        else:
            self.merge_conv = nn.Conv2d(in_channels * 2, out_channels, 1, bias=False)
            self.merge_bn = nn.BatchNorm2d(out_channels)

        self.merge_bottleneck = merge_bottleneck

    def forward(self, x_rgb, x_aux):
        # Intra-modality SE
        x_rgb_se = self.se_rgb(x_rgb)
        x_aux_se = self.se_aux(x_aux)

        # Inter-modality attention
        w_rgb_from_aux, w_aux_from_rgb = self.inter_attn(x_rgb, x_aux)

        # Guided fusion (residual)
        x_rgb_guided = x_rgb_se + w_rgb_from_aux * x_aux
        x_aux_guided = x_aux_se + w_aux_from_rgb * x_rgb

        # Merge
        concat = torch.cat([x_rgb_guided, x_aux_guided], dim=1)

        if self.merge_bottleneck:
            out = self.merge_conv1(concat)
            out = self.merge_bn1(out)
            out = F.relu(out, inplace=True)
            out = self.merge_conv2(out)
            out = self.merge_bn2(out)
        else:
            out = self.merge_conv(concat)
            out = self.merge_bn(out)

        return out
```

#### GAFF Parameter Summary

**Configuration: C=320, se_reduction=4, inter_shared=False, merge_bottleneck=False**

| Component | Parameters |
|-----------|------------|
| SE_RGB | 51,200 |
| SE_Aux | 51,200 |
| InterModalityAttention | 409,600 |
| Guided Fusion | 0 |
| Merge (direct) | 205,440 |
| BatchNorm | ~640 |
| **TOTAL** | **~717,440** |

**Hyperparameter Variations:**
- **se_reduction=8:** SE params halve (25,600 × 2 = 51,200), total ≈ 666K
- **merge_bottleneck=True:** Merge params increase (+103K), total ≈ 820K

**CSSA vs. GAFF Comparison:**
- CSSA: 3,298 params
- GAFF (default): 717,440 params
- **GAFF is 217× heavier than CSSA** (for C=320, r=4)

---

## Feature Pyramid Network (FPN)

**File:** `crossattentiondet/models/backbone.py:20-73`

**Purpose:** Multi-scale feature representation for object detection across scales.

### Architecture

```
Encoder Outputs:              FPN Outputs:
f1 (B,64,56,56) ──────────────→ P0 (B,256,56,56)
                    │
f2 (B,128,28,28) ───┼──────────→ P1 (B,256,28,28)
                    │  │
f3 (B,320,14,14) ───┼──┼────────→ P2 (B,256,14,14)
                    │  │  │
f4 (B,512,7,7) ─────┼──┼──┼─────→ P3 (B,256,7,7)
                    │  │  │  │
                    └──┴──┴──┴───→ Pool (B,256,3,3)
```

**Top-Down Pathway with Lateral Connections:**
1. **Lateral convs:** 1×1 conv to reduce encoder feature channels to 256
2. **Top-down:** Upsample higher-level features (2× bilinear)
3. **Merge:** Element-wise addition of upsampled + lateral features

### Implementation

```python
class CrossAttentionBackbone(nn.Module):
    def __init__(self, encoder, fpn_out_channels=256):
        self.encoder = encoder

        # Lateral 1×1 convs
        encoder_channels = encoder.embed_dims  # [64, 128, 320, 512] for mit_b1
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(c, fpn_out_channels, kernel_size=1)
            for c in encoder_channels
        ])

        # Top-down 3×3 convs (reduce aliasing from upsampling)
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(fpn_out_channels, fpn_out_channels, kernel_size=3, padding=1)
            for _ in range(4)
        ])

    def forward(self, images_tensor):
        rgb_images = images_tensor[:, :3]
        x_images = images_tensor[:, 3:5]

        # Encoder forward
        features = self.encoder.forward_features(rgb_images, x_images)  # [f1, f2, f3, f4]

        # Build FPN bottom-up
        laterals = [lateral_conv(feat) for lateral_conv, feat in zip(self.lateral_convs, features)]

        # Top-down pathway
        fpn_features = []
        for i in range(len(laterals) - 1, -1, -1):
            if i == len(laterals) - 1:
                fpn_feat = laterals[i]  # Start with highest level
            else:
                upsampled = F.interpolate(fpn_features[0], scale_factor=2, mode='bilinear', align_corners=False)
                fpn_feat = laterals[i] + upsampled

            fpn_feat = self.fpn_convs[i](fpn_feat)
            fpn_features.insert(0, fpn_feat)

        # Add extra coarse level (MaxPool)
        pool_feat = F.max_pool2d(fpn_features[-1], kernel_size=1, stride=2, padding=0)

        # Return as dict for Faster R-CNN
        return {
            '0': fpn_features[0],  # P0: (B, 256, 56, 56)
            '1': fpn_features[1],  # P1: (B, 256, 28, 28)
            '2': fpn_features[2],  # P2: (B, 256, 14, 14)
            '3': fpn_features[3],  # P3: (B, 256, 7, 7)
            'pool': pool_feat      # Pool: (B, 256, 3, 3)
        }
```

**FPN Parameters:**
- Lateral convs: 4 × (C_encoder × 256) = 64×256 + 128×256 + 320×256 + 512×256 = 262,144
- FPN convs: 4 × (256 × 256 × 3 × 3) = 2,359,296
- **Total: ~2.62M params**

---

## Detection Head: Faster R-CNN

**File:** `crossattentiondet/training/trainer.py:92-99`

**Framework:** PyTorch's built-in `FasterRCNN` from `torchvision.models.detection`

### Components

#### 1. Anchor Generator

**Purpose:** Generate anchor boxes at each FPN location.

**Configuration:**
```python
anchor_generator = AnchorGenerator(
    sizes=((32,), (64,), (128,), (256,), (512,)),  # One size per FPN level
    aspect_ratios=((0.5, 1.0, 2.0),) * 5            # 3 ratios per level
)
```

**Total Anchors per Location:** 3 (aspect ratios) × 1 (size) = 3
**Total Anchors:** Sum over all FPN locations ≈ 20,000

**Anchor Dimensions (stage 2 example, size=64, ratios=[0.5, 1.0, 2.0]):**
- Ratio 0.5: w=64/√0.5≈90, h=64×√0.5≈45
- Ratio 1.0: w=64, h=64
- Ratio 2.0: w=64×√2≈90, h=64/√2≈45

#### 2. Region Proposal Network (RPN)

**Purpose:** Propose object regions from anchors.

**Architecture:**
```
FPN features (B,256,H,W)
    ↓
Conv2d(256,256,3×3)  # RPN head
    ↓
ReLU
    ├──────────────┬──────────────┐
    │              │              │
Objectness    Box Regression    │
Conv2d(256,3,1×1)  Conv2d(256,12,1×1)  # 3 anchors × 4 coords
    ↓              ↓              ↓
Objectness    Box Deltas     Apply to anchors
Scores         (dx,dy,dw,dh)     ↓
    │              │         Proposals (~2000)
    └──────────────┴──────────────┘
```

**Loss:**
- **Objectness:** Binary cross-entropy (object vs. background)
- **Box Regression:** Smooth L1 loss on (dx, dy, dw, dh)

**NMS (Non-Maximum Suppression):**
- Pre-NMS top-N: 2000
- Post-NMS top-N: 2000 (training), 1000 (inference)
- IoU threshold: 0.7

#### 3. RoI Pooling (MultiScaleRoIAlign)

**Purpose:** Extract fixed-size features from proposals across FPN levels.

**Configuration:**
```python
roi_pooler = MultiScaleRoIAlign(
    featmap_names=['0', '1', '2', '3', 'pool'],
    output_size=7,        # 7×7 output
    sampling_ratio=2      # 2×2 sampling grid per bin
)
```

**FPN Level Assignment (by proposal size):**
```
Proposal size (sqrt(w*h)):
  < 56: Use FPN level '0' (56×56)
  56-112: Use FPN level '1' (28×28)
  112-224: Use FPN level '2' (14×14)
  224-448: Use FPN level '3' (7×7)
  > 448: Use FPN level 'pool' (3×3)
```

**Output:** (N_proposals, 256, 7, 7) features

#### 4. Box Head (Classification + Regression)

**Architecture:**
```
RoI Features (N,256,7,7)
    ↓
Flatten → (N, 256×7×7) = (N, 12544)
    ↓
FC(12544, 1024)
    ↓
ReLU
    ↓
FC(1024, 1024)
    ↓
ReLU
    ├──────────────┬──────────────┐
    │              │              │
Classification  Box Refinement │
FC(1024, num_classes)  FC(1024, num_classes×4)
    ↓              ↓              ↓
Class Logits  Box Deltas    Final Boxes
(N, C)        (N, C×4)      + Class Labels
```

**Loss:**
- **Classification:** Cross-entropy over classes
- **Box Regression:** Smooth L1 loss on refined (dx, dy, dw, dh)

### Total Faster R-CNN Losses

```python
total_loss = loss_classifier + loss_box_reg + loss_objectness + loss_rpn_box_reg
```

**Typical Loss Breakdown (from training logs):**
- loss_classifier: ~0.03-0.05
- loss_box_reg: ~0.02-0.03
- loss_objectness: ~0.02-0.03
- loss_rpn_box_reg: ~0.01-0.02
- **Total: ~0.10-0.13**

---

## Backbone Variants

**File:** `crossattentiondet/models/encoder.py:267`

### MiT (Mix Transformer) Variants

| Variant | Total Params | Channels | Depths | Heads | Use Case | Status |
|---------|--------------|----------|--------|-------|----------|--------|
| **mit_b0** | 55.7M | [32,64,160,256] | [2,2,2,2] | [1,2,5,8] | Fast prototyping, edge | ✅ Complete (loss=0.1057) |
| **mit_b1** | 69.5M | [64,128,320,512] | [2,2,2,2] | [1,2,5,8] | **Default, best trade-off** | ✅ Complete (loss=0.1027) |
| **mit_b2** | 82.1M | [64,128,320,512] | [3,4,6,3] | [1,2,5,8] | Balanced accuracy/speed | 🔄 Training (1/15 epochs) |
| **mit_b3** | 117.2M | [64,128,320,512] | [3,4,18,3] | [1,2,5,8] | Higher accuracy | ⏸️ Not tested |
| **mit_b4** | 155.4M | [64,128,320,512] | [3,8,27,3] | [1,2,5,8] | High accuracy | ❌ CUDA OOM (needs optimization) |
| **mit_b5** | 196.6M | [64,128,320,512] | [3,6,40,3] | [1,2,5,8] | Maximum accuracy | ❌ CUDA OOM (needs optimization) |

### Detailed Specifications (mit_b1)

| Stage | Input Res | Output Res | Channels | Depth | Heads | SR Ratio | MLP Ratio | Params (approx) |
|-------|-----------|------------|----------|-------|-------|----------|-----------|-----------------|
| 1 | 224×224 | 56×56 | 64 | 2 | 1 | 8 | 4 | ~1.5M |
| 2 | 56×56 | 28×28 | 128 | 2 | 2 | 4 | 4 | ~3M |
| 3 | 28×28 | 14×14 | 320 | 2 | 5 | 2 | 4 | ~15M |
| 4 | 14×14 | 7×7 | 512 | 2 | 8 | 1 | 4 | ~25M |

**Total Encoder Params (mit_b1):** ~44.5M (dual-stream RGB + Aux doubles this to ~69.5M with FPN+detection head)

### Backbone Selection Guidelines

**For Accuracy:**
- mit_b2, mit_b3: Best if GPU memory allows
- mit_b1: Good default

**For Speed:**
- mit_b0: Fastest, acceptable accuracy
- mit_b1: Good balance

**For Edge Deployment:**
- mit_b0 + CSSA fusion: Minimal overhead

**For Cloud/GPU-Rich:**
- mit_b2/b3 + GAFF fusion: Maximum accuracy

---

## Data Flow & Dimensions

### Complete Forward Pass (mit_b1, batch_size=2, input=224×224×5)

```
Input:
  images_tensor: (2, 5, 224, 224)
    ├─ RGB: (2, 3, 224, 224)
    └─ Aux: (2, 2, 224, 224)  # Thermal + Event

Encoder Stage 1:
  Patch Embed: (2, 3, 224, 224) → (2, 3136, 64)  # RGB stream, 56×56 = 3136 tokens
               (2, 2, 224, 224) → (2, 3136, 64)  # Aux stream
  Transformer Blocks (2 blocks): (2, 3136, 64) → (2, 3136, 64)
  Reshape: (2, 3136, 64) → (2, 64, 56, 56)

  Fusion (FRM+FFM): (2, 64, 56, 56) × 2 → (2, 64, 56, 56)  # f1

Encoder Stage 2:
  Patch Embed: (2, 64, 56, 56) → (2, 784, 128)  # 28×28 = 784 tokens
  Transformer Blocks (2 blocks): (2, 784, 128) → (2, 784, 128)
  Reshape: (2, 784, 128) → (2, 128, 28, 28)

  Fusion: (2, 128, 28, 28) × 2 → (2, 128, 28, 28)  # f2

Encoder Stage 3:
  Patch Embed: (2, 128, 28, 28) → (2, 196, 320)  # 14×14 = 196 tokens
  Transformer Blocks (2 blocks): (2, 196, 320) → (2, 196, 320)
  Reshape: (2, 196, 320) → (2, 320, 14, 14)

  Fusion: (2, 320, 14, 14) × 2 → (2, 320, 14, 14)  # f3

Encoder Stage 4:
  Patch Embed: (2, 320, 14, 14) → (2, 49, 512)  # 7×7 = 49 tokens
  Transformer Blocks (2 blocks): (2, 49, 512) → (2, 49, 512)
  Reshape: (2, 49, 512) → (2, 512, 7, 7)

  Fusion: (2, 512, 7, 7) × 2 → (2, 512, 7, 7)  # f4

FPN:
  Lateral Convs:
    f1: (2, 64, 56, 56) → (2, 256, 56, 56)  # P0
    f2: (2, 128, 28, 28) → (2, 256, 28, 28)  # P1
    f3: (2, 320, 14, 14) → (2, 256, 14, 14)  # P2
    f4: (2, 512, 7, 7) → (2, 256, 7, 7)     # P3

  Top-Down Pathway: Merges with upsampling
  Extra Pool: (2, 256, 7, 7) → (2, 256, 3, 3)  # Pool

  Output: {'0': P0, '1': P1, '2': P2, '3': P3, 'pool': Pool}

RPN:
  Input: FPN features (5 levels, each 256 channels)
  Anchors: ~20,000 total across all levels
  Proposals: ~2000 after NMS

RoI Pooling:
  Input: 2000 proposals + FPN features
  Output: (2000, 256, 7, 7)

Box Head:
  FC Layers: (2000, 256×7×7) → (2000, 1024) → (2000, 1024)
  Classification: (2000, 1024) → (2000, num_classes)
  Box Regression: (2000, 1024) → (2000, num_classes×4)

Final Output:
  Detected boxes: (N, 4)  # N varies per image
  Class labels: (N,)
  Scores: (N,)
```

**Total FLOPs (approximate, mit_b1 + CSSA fusion):**
- Encoder: ~50 GFLOPs
- Fusion: ~0.1 GFLOPs (CSSA) or ~5 GFLOPs (GAFF)
- FPN: ~10 GFLOPs
- RPN + Detection Head: ~20 GFLOPs
- **Total: ~80 GFLOPs (CSSA) or ~85 GFLOPs (GAFF)**

---

[← Back to Index](00_INDEX.md) | [← Previous: Executive Summary](01_EXECUTIVE_SUMMARY.md) | [Next: Dataset & Modalities →](03_DATASET_AND_MODALITIES.md)
