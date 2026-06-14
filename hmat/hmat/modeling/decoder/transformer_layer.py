import torch
import torch.nn as nn
from .gated_attention import GatedMemoryAttention


class MemoryDecoderLayer(nn.Module):
    """
    Pre-LayerNorm 版本的 Memory-Augmented Decoder 层。

    结构（每层）：
    1. Pre-Norm Self-Attention   → residual +
    2. Pre-Norm Gated Cross-Attn → residual +
    3. Pre-Norm FFN (GELU)       → residual +

    Pre-Norm 优势：梯度通过 residual 直通，在深层 / backbone 解冻时更稳定。
    """

    def __init__(self, d_model=256, nhead=8, dim_feedforward=1024, dropout=0.1):
        super().__init__()

        # ── 1. Pre-Norm Self-Attention ──
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)

        # ── 2. Pre-Norm Gated Cross-Attention ──
        self.norm2 = nn.LayerNorm(d_model)
        self.gated_cross_attn = GatedMemoryAttention(
            d_model, nhead, dropout=dropout)

        # ── 3. Pre-Norm FFN (GELU) ──
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, src, memory_values=None,
                src_mask=None, src_pos=None, query_pos=None):
        """
        Args:
            tgt:  [B, Nq, D]  当前 queries（含 track+detect）
            src:  [B, Ns, D]  当前帧图像特征
            memory_values: [B, Nm, D] or None  历史记忆
            src_mask:  [B, Ns]        图像 padding mask (True=pad)
            src_pos:   [B, Ns, D]     图像位置编码
            query_pos: [B, Nq, D]     query 位置编码

        Returns:
            tgt: [B, Nq, D] 更新后的 queries
        """
        # ── A. Pre-Norm Self-Attention ──
        residual = tgt
        normed = self.norm1(tgt)

        # position 只加到 query/key 上（value 不加 —— 标准做法）
        q = k = normed + query_pos if query_pos is not None else normed
        attn_out, _ = self.self_attn(q, k, normed)
        tgt = residual + self.dropout1(attn_out)

        # ── B. Pre-Norm Gated Cross-Attention ──
        tgt = self.gated_cross_attn(
            tgt=tgt,
            query_pos=query_pos,
            src=src,
            src_pos=src_pos,
            memory_values=memory_values,
            src_mask=src_mask,
        )

        # ── C. Pre-Norm FFN (GELU) ──
        residual = tgt
        tgt = residual + self.dropout3(self.ffn(self.norm3(tgt)))

        return tgt
