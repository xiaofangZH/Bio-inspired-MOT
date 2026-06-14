import torch
import torch.nn as nn
import torch.nn.functional as F


class RPNLite(nn.Module):
    """
    Lightweight Region Proposal Network (v2).

    改进:
    - 每个线性层后添加 LayerNorm，防止 backbone 解冻时激活值爆炸
    - ReLU → GELU，更平滑的梯度
    - bbox_head 输出动态 ROI 尺度并通过 softplus 激活
    """

    def __init__(self, in_channels=256, out_channels=256, num_rois=10):
        super().__init__()
        self.in_channels = in_channels
        self.num_rois = num_rois

        # Score head: token → ROI 中心分数 [0, 1]
        self.score_head = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.LayerNorm(in_channels),
            nn.GELU(),
            nn.Linear(in_channels, 1),
            nn.Sigmoid(),
        )

        # Bbox head: token → (dw_log, dh_log) 动态 ROI 尺度
        self.bbox_head = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.LayerNorm(in_channels),
            nn.GELU(),
            nn.Linear(in_channels, 2),
        )

    def forward(self, x):
        """
        Args:
            x: [B, N, D]  backbone 特征 tokens

        Returns:
            top_indices: [B, K]       top-K ROI 中心 token 索引
            top_scales:  [B, K, 2]    (w_scale, h_scale) ≥ 0.5
        """
        B, N, D = x.shape

        # 1. 每个 token 的 ROI 中心分数
        scores = self.score_head(x).squeeze(-1)  # [B, N]

        # 2. 选 top-K
        K = min(self.num_rois, N)
        _, top_indices = torch.topk(scores, K, dim=1)  # [B, K]

        # 3. 动态 ROI 尺度
        bbox_raw = self.bbox_head(x)  # [B, N, 2]
        top_scales = torch.gather(
            bbox_raw, 1,
            top_indices.unsqueeze(-1).expand(-1, -1, 2)
        )  # [B, K, 2]

        # softplus: 确保正值，+0.5 保证最小尺度
        top_scales = F.softplus(top_scales) + 0.5

        return top_indices, top_scales

    def get_roi_coordinates(self, top_indices, feature_shape):
        """
        Token 索引 → 归一化 ROI 坐标。

        Args:
            top_indices:   [B, K]
            feature_shape: (H, W)

        Returns:
            rois: [B, K, 4]  (x1, y1, x2, y2) in [0, 1]
        """
        B, K = top_indices.shape
        H, W = feature_shape

        y_idx = top_indices // W
        x_idx = top_indices % W

        cx = (x_idx.float() + 0.5) / W
        cy = (y_idx.float() + 0.5) / H

        roi_w = 0.3
        roi_h = 0.3

        x1 = (cx - roi_w / 2).clamp(0, 1)
        y1 = (cy - roi_h / 2).clamp(0, 1)
        x2 = (cx + roi_w / 2).clamp(0, 1)
        y2 = (cy + roi_h / 2).clamp(0, 1)

        return torch.stack([x1, y1, x2, y2], dim=-1)
