import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedMemoryAttention(nn.Module):
    """
    Pre-Norm 版 Gated Memory Attention。

    工作流程（每层）：
    1. Pre-LayerNorm(query) → 加权 query_pos → image attn / memory attn
    2. 5 通道门控融合: [Query, ImgOut, MemOut, Divergence, Agreement]
    3. Residual: output = tgt + Dropout(gated_fusion)

    关键改进:
    - Pre-Norm: 梯度直通 residual，比原 post-norm 更稳定
    - 门控输入从 3 通道补全到 5 通道（对齐论文描述）
    - 残差由外部管理，避免原实现中 residual 嵌套混乱
    """

    def __init__(self, d_model=256, nhead=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        assert d_model % nhead == 0

        # Pre-norm
        self.norm_query = nn.LayerNorm(d_model)

        # Image cross-attention
        self.img_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True)

        # Memory cross-attention
        self.mem_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True)

        # 5-channel gating network:
        # [Query, ImgOut, MemOut, Divergence(Img-Mem), Agreement(Img*Mem)]
        self.gate_net = nn.Sequential(
            nn.Linear(d_model * 5, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.Sigmoid(),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, query_pos, src, src_pos,
                memory_values=None, memory_ages=None, memory_mask=None,
                src_mask=None):
        """
        Args:
            tgt:            [B, Nq, D]  queries（已经过 self-attn）
            query_pos:      [B, Nq, D]  query 位置编码 (or None)
            src:            [B, Ns, D]  图像特征 values
            src_pos:        [B, Ns, D]  图像位置编码 (or None)
            memory_values:  [B, Nm, D]  记忆 key/value (or None)
            memory_ages:    [B, Nm]     各槽位年龄，用于时序衰减偏置
            memory_mask:    [B, Nm]     记忆 padding mask (True=pad/空槽)
            src_mask:       [B, Ns]     图像 padding mask (True=pad)

        Returns:
            tgt: [B, Nq, D] 更新后的 queries（含 residual）
        """
        residual = tgt

        # Pre-Norm: norm before attention
        normed = self.norm_query(tgt)

        # Add position to query (only once — no double addition)
        q = normed + query_pos if query_pos is not None else normed

        # Image key with position
        k_img = src + src_pos if src_pos is not None else src

        # ── 1. Image Cross-Attention ──
        out_img, _ = self.img_attn(
            q, k_img, src,
            key_padding_mask=src_mask
        )

        # ── 2. Memory Cross-Attention (with temporal bias) ──
        if memory_values is not None and memory_values.shape[1] > 0:
            # Manual attention computation to inject temporal decay bias.
            # Standard: softmax(Q @ K^T / sqrt(d) + mask)
            # Enhanced: softmax(Q @ K^T / sqrt(d) + mask - alpha * age)
            mha = self.mem_attn

            # Compute Q, K, V projections
            q_proj = mha.in_proj_q(q) if hasattr(mha, 'in_proj_q') else \
                     F.linear(q, mha.in_proj_weight[:self.d_model], mha.in_proj_bias[:self.d_model] if mha.in_proj_bias is not None else None)
            k_proj = mha.in_proj_k(memory_values) if hasattr(mha, 'in_proj_k') else \
                     F.linear(memory_values, mha.in_proj_weight[self.d_model:2*self.d_model],
                              mha.in_proj_bias[self.d_model:2*self.d_model] if mha.in_proj_bias is not None else None)
            v_proj = mha.in_proj_v(memory_values) if hasattr(mha, 'in_proj_v') else \
                     F.linear(memory_values, mha.in_proj_weight[2*self.d_model:],
                              mha.in_proj_bias[2*self.d_model:] if mha.in_proj_bias is not None else None)

            # Reshape to multi-head: [B, N, D] → [B, H, N, d]
            B, Nq, D = q_proj.shape
            _, Nm, _ = k_proj.shape
            q_proj = q_proj.view(B, Nq, self.nhead, self.head_dim).transpose(1, 2)
            k_proj = k_proj.view(B, Nm, self.nhead, self.head_dim).transpose(1, 2)
            v_proj = v_proj.view(B, Nm, self.nhead, self.head_dim).transpose(1, 2)

            # Attention scores
            scale = self.head_dim ** -0.5
            attn_scores = torch.matmul(q_proj, k_proj.transpose(-2, -1)) * scale  # [B, H, Nq, Nm]

            # Apply memory padding mask
            if memory_mask is not None:
                attn_scores = attn_scores.masked_fill(
                    memory_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

            # Apply temporal decay bias: penalty for older memory slots
            if memory_ages is not None:
                # age: [B, Nm] → [B, 1, 1, Nm]
                age_bias = memory_ages.unsqueeze(1).unsqueeze(2).to(attn_scores.dtype)
                # Log-scale decay: older slots get lower attention
                temporal_penalty = torch.log1p(age_bias) * 2.0  # ~0 at age=0, grows with age
                attn_scores = attn_scores - temporal_penalty

            # Softmax + output
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = F.dropout(attn_weights, p=self.mem_attn.dropout if hasattr(self.mem_attn, 'dropout') else 0.0, training=self.training)
            out_mem = torch.matmul(attn_weights, v_proj)  # [B, H, Nq, d]
            out_mem = out_mem.transpose(1, 2).contiguous().view(B, Nq, D)

            # Apply output projection
            if hasattr(mha, 'out_proj'):
                out_mem = mha.out_proj(out_mem)

            # ── 3. 5-Channel Gating ──
            # 在 FP32 中进行门控计算（防止乘积溢出）
            q_fp32 = q.float()
            out_img_fp32 = out_img.float()
            out_mem_fp32 = out_mem.float()
            divergence = out_img_fp32 - out_mem_fp32
            agreement = out_img_fp32 * out_mem_fp32
            gate_input = torch.cat(
                [q_fp32, out_img_fp32, out_mem_fp32, divergence, agreement], dim=-1
            )  # [B, Nq, D*5]

            gate = self.gate_net(gate_input).to(tgt.dtype)  # 转回原始精度

            # Adaptive fusion: gate→1 trust image, gate→0 trust memory
            out_fused = gate * out_img + (1.0 - gate) * out_mem
        else:
            # No memory: use image-only
            out_fused = out_img

        # ── 4. Residual connection ──
        tgt = residual + self.dropout(out_fused)

        return tgt

    def get_gate_statistics(self, tgt, query_pos, src, src_pos,
                            memory_values, src_mask=None):
        """
        分析用：返回门控统计值。
        """
        if memory_values is None or memory_values.shape[1] == 0:
            return {'gate_mean': 1.0, 'gate_std': 0.0}

        normed = self.norm_query(tgt)
        q = normed + query_pos if query_pos is not None else normed
        k_img = src + src_pos if src_pos is not None else src

        out_img, _ = self.img_attn(q, k_img, src, key_padding_mask=src_mask)
        out_mem, _ = self.mem_attn(q, memory_values, memory_values)

        divergence = out_img - out_mem
        agreement = out_img * out_mem
        gate_input = torch.cat(
            [q, out_img, out_mem, divergence, agreement], dim=-1)
        gate = self.gate_net(gate_input)

        return {
            'gate_mean': gate.mean().item(),
            'gate_std': gate.std().item(),
        }
