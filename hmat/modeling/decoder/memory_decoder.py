import torch
import torch.nn as nn
from .transformer_layer import MemoryDecoderLayer


class MemoryAugmentedDecoder(nn.Module):
    """
    Pre-Norm Memory-Augmented Decoder（v2）。

    6 层 Pre-Norm Decoder Layer，每层顺序：
    1. Self-Attention（query-query 交互）
    2. Gated Cross-Attention（query ← image + memory 门控融合）
    3. FFN（GELU）

    query_pos 在每层只加一次（self-attn 后自动被 residual 传递），
    修复了旧版中 query_pos 被重复累加的 bug。
    """

    def __init__(self, d_model=256, nhead=8, num_layers=6,
                 dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        self.layers = nn.ModuleList([
            MemoryDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

    def forward(self,
                src_feats, src_mask, src_pos,
                detect_queries, track_queries,
                detect_pos=None, track_pos=None,
                memory_values=None,
                memory_ages=None,
                memory_mask=None):
        """
        Args:
            src_feats:      [B, N_img, D]  编码器输出图像特征
            src_mask:       [B, N_img]     图像 padding mask
            src_pos:        [B, N_img, D]  图像位置编码

            detect_queries: [B, N_det, D]  可学习检测 query（content）
            track_queries:  [B, N_trk, D]  记忆库 track query（content; None=空）

            detect_pos:     [B, N_det, D]  检测 query 位置编码
            track_pos:      [B, N_trk, D]  track query 位置编码（None=空）

            memory_values:  [B, N_mem, D]  历史记忆 key/value（None=空）
            memory_ages:    [B, N_mem]     各槽位年龄，用于时序偏置（None=不衰减）
            memory_mask:    [B, N_mem]     记忆 padding mask (True=pad/空槽)

        Returns:
            output_embeds: [B, N_trk + N_det, D]  最终解码嵌入
        """
        # ── 1. 拼合 queries 和其位置编码 ──
        device = detect_queries.device
        if track_queries is not None and track_queries.shape[1] > 0:
            track_queries = track_queries.to(device)
            if track_queries.device != detect_queries.device:
                import logging
                logging.warning(f"Device mismatch: track_queries@{track_queries.device} vs detect_queries@{detect_queries.device}")
            tgt = torch.cat([track_queries, detect_queries], dim=1)  # [B, Nq, D]
            if track_pos is not None and detect_pos is not None:
                track_pos = track_pos.to(device)
                query_pos = torch.cat([track_pos, detect_pos], dim=1)
            else:
                query_pos = detect_pos
        else:
            tgt = detect_queries
            query_pos = detect_pos

        # ── 2. 逐层处理 ──
        for layer in self.layers:
            tgt = layer(
                tgt=tgt,
                src=src_feats,
                memory_values=memory_values,
                memory_ages=memory_ages,
                memory_mask=memory_mask,
                src_mask=src_mask,
                src_pos=src_pos,
                query_pos=query_pos,
            )

        # ── 3. 最终 LayerNorm ──
        output = self.norm(tgt)

        return output
