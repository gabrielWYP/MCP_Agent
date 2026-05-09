"""
Cross-Modal Attention Fusion module.

Applied at each of the 4 backbone stages independently.
RGB features act as Query — they "ask" the NIR stream what is relevant.
NIR features act as Key and Value — they "answer" with their information.

The output is an enriched RGB feature map that has absorbed NIR context,
plus the original NIR features (passed through unchanged for the NIR FPN).

Why RGB as Query?
    At inference time the student only sees RGB. By making RGB the query,
    we train the RGB stream to actively seek NIR information, which means
    the RGB features become richer representations of what NIR would reveal.
    This is the core of cross-modal distillation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StageAttentionFusion(nn.Module):
    """
    Cross-attention fusion for a single backbone stage.

    Implements:
        Q = linear(rgb_features)
        K = linear(nir_features)
        V = linear(nir_features)
        fused = MultiheadAttention(Q, K, V) + rgb_features  (residual)

    The spatial dimensions are flattened for attention and restored after.

    Args:
        channels (int): Number of channels at this stage (96/192/384/768).
        num_heads (int): Number of attention heads. Must divide channels evenly.
        dropout (float): Dropout on attention weights (regularization).
    """

    def __init__(self, channels: int, num_heads: int = 8, dropout: float = 0.1, max_tokens_side: int = 20):
        super().__init__()

        assert channels % num_heads == 0, (
            f"channels ({channels}) must be divisible by num_heads ({num_heads})"
        )

        self.max_tokens_side = max_tokens_side  # pool spatial dims to at most this size

        # Project RGB and NIR to a common attention dimension
        # We keep embed_dim == channels to preserve spatial resolution info
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,   # (N, seq_len, embed_dim) — more intuitive
        )

        # Layer norm before attention (Pre-LN, more stable training)
        self.norm_rgb = nn.LayerNorm(channels)
        self.norm_nir = nn.LayerNorm(channels)

        # Feed-forward after attention (standard transformer block pattern)
        self.ffn = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 4, channels),
            nn.Dropout(dropout),
        )

    def forward(
        self, rgb_feat: torch.Tensor, nir_feat: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            rgb_feat: (N, C, H, W)
            nir_feat: (N, C, H, W)

        Returns:
            fused: (N, C, H, W) — RGB enriched with NIR context
        """
        N, C, H, W = rgb_feat.shape

        # --- Spatial pooling to reduce sequence length before attention ---
        # Early stages (160x160, 80x80) would produce 25k+ tokens — OOM on CPU.
        # We pool to a fixed token grid, apply attention, then upsample back.
        # This is the approach used in CMX and TokenFusion papers.
        pool_size = min(H, W, self.max_tokens_side)
        needs_pool = (H > pool_size) or (W > pool_size)

        if needs_pool:
            rgb_pooled = F.adaptive_avg_pool2d(rgb_feat, pool_size)  # (N, C, P, P)
            nir_pooled = F.adaptive_avg_pool2d(nir_feat, pool_size)
        else:
            rgb_pooled = rgb_feat
            nir_pooled = nir_feat

        # Flatten spatial dims: (N, C, P, P) → (N, P*P, C)
        rgb_seq = rgb_pooled.flatten(2).permute(0, 2, 1)
        nir_seq = nir_pooled.flatten(2).permute(0, 2, 1)

        # Pre-LN
        rgb_norm = self.norm_rgb(rgb_seq)
        nir_norm = self.norm_nir(nir_seq)

        # Cross-attention: Q from RGB, K/V from NIR
        attn_out, _ = self.attn(
            query=rgb_norm,
            key=nir_norm,
            value=nir_norm,
        )

        # Residual + FFN
        rgb_seq = rgb_seq + attn_out
        rgb_seq = rgb_seq + self.ffn(rgb_seq)

        # Restore to spatial map: (N, P*P, C) → (N, C, P, P)
        P = pool_size
        fused_pooled = rgb_seq.permute(0, 2, 1).reshape(N, C, P, P)

        # Upsample back to original resolution if we pooled
        if needs_pool:
            fused = F.interpolate(fused_pooled, size=(H, W), mode="bilinear", align_corners=False)
            # Residual from original full-res RGB (preserves fine-grained detail)
            fused = fused + rgb_feat
        else:
            fused = fused_pooled

        return fused


class CrossModalFusion(nn.Module):
    """
    Applies StageAttentionFusion at each of the 4 backbone stages.

    Args:
        stage_channels (list[int]): Channels per stage, e.g. [96, 192, 384, 768].
        num_heads_per_stage (list[int]): Attention heads per stage.
            Defaults to [4, 8, 8, 8] — fewer heads at early stages (lower channels).
        dropout (float): Attention dropout.

    Returns (forward):
        fused_features: list of 4 fused tensors [F1, F2, F3, F4]
        nir_features:   list of 4 original NIR tensors (passed to NIR FPN)
    """

    DEFAULT_HEADS = [4, 8, 8, 8]

    def __init__(
        self,
        stage_channels: list[int] = None,
        num_heads_per_stage: list[int] = None,
        dropout: float = 0.1,
        max_tokens_side: int = 20,
    ):
        super().__init__()

        if stage_channels is None:
            stage_channels = [96, 192, 384, 768]
        if num_heads_per_stage is None:
            num_heads_per_stage = self.DEFAULT_HEADS

        assert len(stage_channels) == len(num_heads_per_stage) == 4

        self.fusion_stages = nn.ModuleList([
            StageAttentionFusion(
                channels=ch,
                num_heads=heads,
                dropout=dropout,
                max_tokens_side=max_tokens_side,
            )
            for ch, heads in zip(stage_channels, num_heads_per_stage)
        ])

    def forward(
        self,
        rgb_features: list[torch.Tensor],
        nir_features: list[torch.Tensor],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        Args:
            rgb_features: [S1, S2, S3, S4] from RGB encoder
            nir_features: [S1, S2, S3, S4] from NIR encoder

        Returns:
            fused_features: [F1, F2, F3, F4] — RGB enriched with NIR
            nir_features:   [S1, S2, S3, S4] — original NIR (for NIR FPN)
        """
        fused_features = []
        for i, fusion in enumerate(self.fusion_stages):
            fused = fusion(rgb_features[i], nir_features[i])
            fused_features.append(fused)

        # NIR features are returned unchanged for the parallel NIR FPN
        return fused_features, nir_features
