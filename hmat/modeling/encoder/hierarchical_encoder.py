import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.ops import roi_align
from .rpn_lite import RPNLite


class PositionalEmbedding2D(nn.Module):
    """
    True 2D sinusoidal positional encoding.

    Splits d_model into two halves:
      - First half encodes Y-axis (row position)
      - Second half encodes X-axis (column position)

    This preserves 2D spatial topology — patches at (h, w) get unique
    encoding based on their true 2D coordinates, unlike a flattened 1D PE
    where row neighbours are spatially disconnected.
    """

    def __init__(self, d_model, max_h=100, max_w=100):
        super().__init__()
        self.d_model = d_model

        pe = torch.zeros(max_h, max_w, d_model)
        d_half = d_model // 2
        div_term = torch.exp(torch.arange(0, d_half, 2).float()
                             * (-math.log(10000.0) / d_half))

        # Y-axis (rows)
        pos_y = torch.arange(0, max_h).unsqueeze(1).float()
        pe[:, :, 0:d_half:2] = torch.sin(pos_y * div_term).unsqueeze(1)
        pe[:, :, 1:d_half:2] = torch.cos(pos_y * div_term).unsqueeze(1)

        # X-axis (columns)
        pos_x = torch.arange(0, max_w).unsqueeze(1).float()
        pe[:, :, d_half::2] = torch.sin(pos_x * div_term).unsqueeze(0)
        pe[:, :, d_half+1::2] = torch.cos(pos_x * div_term).unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x, H, W):
        """
        Get 2D positional encoding for a flattened grid.

        Args:
            x: [B, H*W, D] — used only for shape/batch reference
            H, W: grid height and width in patches

        Returns:
            pos: [B, H*W, D] positional encoding
        """
        pe = self.pe[:H, :W, :].reshape(H * W, self.d_model)
        return pe.unsqueeze(0).expand(x.shape[0], -1, -1)

    def get_pos_at_indices(self, B, indices, H, W):
        """
        Get PD at specific 2D positions given linear patch-grid indices.

        Args:
            B: batch size
            indices: [B, K] linear indices in [0, H*W)
            H, W: grid dimensions

        Returns:
            pos: [B, K, d_model]
        """
        cy = (indices // W).clamp(0, H - 1)      # [B, K]
        cx = (indices % W).clamp(0, W - 1)       # [B, K]
        pe_grid = self.pe[:H, :W, :]              # [H, W, D]
        pos = pe_grid[cy, cx]                     # [B, K, D]
        return pos


class HierarchicalVisualEncoder(nn.Module):
    """
    Hierarchical Visual Encoder (v3).

    改进:
    - 使用 DINOv3 多层特征（4 层）代替单层，获取多尺度信息
    - 添加可学习的 Type Embedding 区分全局 token 和 ROI token
    - 向量化 ROI 邻域提取（保持 v2 的优化）
    """

    def __init__(self, hidden_dim=256, num_levels=2, num_rois=10, patch_size=16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_levels = num_levels
        self.num_rois = num_rois
        self.patch_size = patch_size

        # Regional Proposal Network (Lite)
        self.rpn = RPNLite(in_channels=hidden_dim,
                           out_channels=hidden_dim, num_rois=num_rois)

        # Positional encoding
        self.pos_encoder = PositionalEmbedding2D(hidden_dim)

        # Type embeddings: 区分全局 token (type=0) 和 ROI token (type=1)
        self.type_embeddings = nn.Embedding(2, hidden_dim)
        nn.init.normal_(self.type_embeddings.weight, std=0.02)

        # ROI 邻域聚合: 将 ROI 中心 token 的邻域 pool 到单个向量
        self.roi_window = 5  # 5x5 window around ROI center
        self.roi_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def _extract_roi_features_vectorized(
        self, global_feats, roi_indices, h_patches, w_patches
    ):
        """
        RoIAlign 精确提取 ROI 区域特征（替代粗糙的 5×5 patch 均值池化）。

        将 1D patch token 序列恢复为 2D 空间特征图，使用双线性插值
        在连续坐标上采样，保留亚像素空间精度。

        Args:
            global_feats: [B, N, D]  全局 patch tokens
            roi_indices:  [B, K]     top-K token indices (linear)
            h_patches:    int        grid height in patches
            w_patches:    int        grid width in patches

        Returns:
            roi_feats: [B, K, D]  每个 ROI 的精确池化特征
        """
        B, N, D = global_feats.shape
        K = roi_indices.shape[1]
        device = global_feats.device

        # 1. 将 1D 序列恢复为 2D 特征图 [B, D, H, W]
        features_2d = global_feats.transpose(1, 2).reshape(
            B, D, h_patches, w_patches).contiguous()

        # 2. ROI 中心坐标（patch grid 坐标系）
        cy = roi_indices.float() // w_patches   # [B, K]
        cx = roi_indices.float() % w_patches    # [B, K]

        # 3. 构建 5×5 窗口 boxes [x1, y1, x2, y2]
        half = self.roi_window / 2.0
        # +0.5 偏移确保采样点位于 patch 中心（roi_align aligned=True）
        x1 = cx - half + 0.5
        y1 = cy - half + 0.5
        x2 = cx + half + 0.5
        y2 = cy + half + 0.5

        # 4. 每 batch 收集 boxes
        boxes_list = [
            torch.stack([x1[b], y1[b], x2[b], y2[b]], dim=-1)
            for b in range(B)
        ]  # list of [K, 4]

        # 5. RoIAlign: 在 5×5 窗口内用双线性插值采样
        roi_feats = roi_align(
            features_2d, boxes_list, output_size=(5, 5),
            spatial_scale=1.0, aligned=True,
        )  # [B*K, D, 5, 5]

        # 6. 空间池化 → [B, K, D]
        roi_feats = roi_feats.view(B, K, D, 5, 5).mean(dim=[-2, -1])

        return roi_feats

    def forward(self, backbone, images):
        """
        Forward pass — 多层 DINOv3 特征 + RPN + Type Embedding。

        Args:
            backbone: DINOv3Wrapper instance
            images:   [B, C, H, W] input images

        Returns:
            src_feats: [B, N_global + N_roi, D]  拼接特征（含 type embedding）
            src_mask:  [B, N_global + N_roi]     padding mask
            src_pos:   [B, N_global + N_roi, D]  位置编码
        """
        B, _, H, W = images.shape
        device = images.device

        # ── 1. 多层特征提取 (P1: multi-scale DINOv3) ──
        global_feats, global_mask = backbone.get_multiscale_features(
            images, num_levels=4
        )  # [B, N, D]
        N_global = global_feats.shape[1]

        # 计算 patch grid 尺寸 (从 backbone 动态获取 patch_size)
        h_patches = max(H // self.patch_size, 1)
        w_patches = max(W // self.patch_size, 1)

        # ── 2. RPN: 对每个 token 打分 选 top-K ROI ──
        roi_indices, roi_scales = self.rpn(global_feats)

        # ── 3. 向量化提取 ROI 邻域特征 ──
        roi_feats = self._extract_roi_features_vectorized(
            global_feats, roi_indices, h_patches, w_patches
        )  # [B, K, D]
        roi_feats = self.roi_proj(roi_feats)  # [B, K, D]

        K = roi_indices.shape[1]

        # ── 4. 拼接全局 + ROI 特征 ──
        src_feats = torch.cat([global_feats, roi_feats], dim=1)  # [B, N+K, D]

        # ── 5. Token Type Embedding (P1) ──
        # type=0 → 全局 token, type=1 → ROI token
        global_type_ids = torch.zeros(B, N_global, dtype=torch.long, device=device)
        roi_type_ids = torch.ones(B, K, dtype=torch.long, device=device)
        type_ids = torch.cat([global_type_ids, roi_type_ids], dim=1)  # [B, N+K]
        src_feats = src_feats + self.type_embeddings(type_ids)  # add type info

        # ── 6. Mask ──
        src_mask = torch.zeros((B, N_global + K),
                               dtype=torch.bool, device=device)
        src_mask[:, :N_global] = global_mask

        # ── 7. 位置编码 (True 2D PE) ──
        src_pos = torch.zeros(
            (B, N_global + K, self.hidden_dim), device=device)
        # 全局 token: 2D grid flatten → PE
        src_pos[:, :N_global, :] = self.pos_encoder(
            global_feats, h_patches, w_patches)
        # ROI token: 使用 ROI 中心在 grid 中的 2D 位置
        if K > 0:
            src_pos[:, N_global:, :] = self.pos_encoder.get_pos_at_indices(
                B, roi_indices, h_patches, w_patches)

        return src_feats, src_mask, src_pos
