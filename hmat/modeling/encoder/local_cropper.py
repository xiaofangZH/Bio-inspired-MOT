import torch
import torch.nn as nn
import torchvision.ops as ops


class LocalCropper(nn.Module):
    """
    GPU-parallel ROI cropping via roi_align + batched backbone inference.

    Eliminates the original nested Python for-loop bottleneck:
      - Old: B × K separate backbone(crop) calls (e.g. 2 × 10 = 20 ViT forwards)
      - New: 1 batched backbone call on all crops simultaneously (~10× faster)

    Pipeline:
      1. Convert roi_indices → pixel boxes (vectorized per batch)
      2. roi_align extracts all crops in one GPU kernel
      3. backbone(batched_crops) — single forward pass
      4. Split flat results back to [B, max_len*Np, D] padded tensor
    """

    def __init__(self, hidden_dim=256, crop_size=224, num_patches_per_roi=49):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.crop_size = crop_size
        self.num_patches_per_roi = num_patches_per_roi

    def forward(self, images, roi_indices, backbone):
        """
        Args:
            images:      [B, C, H, W] 原始图像
            roi_indices: [B, K]       ROI 中心 token 索引 (linear)
            backbone:    DINOv3Wrapper

        Returns:
            local_feats: [B, max_len * N_patches, D]  填充对齐的局部特征
        """
        B, C, H, W = images.shape
        device = images.device

        # ── 1. 估算 token 网格尺寸 (保持与原实现兼容) ──
        num_tokens_h = max(1, int((H / 16) ** 0.5 * 2))
        num_tokens_w = max(1, int((W / 16) ** 0.5 * 2))
        stride_y = H / num_tokens_h
        stride_x = W / num_tokens_w

        # ── 2. 向量化构建每 batch 的 RoI boxes ──
        #     roi_align 接受 list[Tensor[K_i, 4]]，每元素对应一个 batch
        boxes_list = []
        counts_per_batch = []

        for b in range(B):
            idx = roi_indices[b]                              # [K_b]
            K_b = idx.numel()
            counts_per_batch.append(K_b)
            if K_b == 0:
                boxes_list.append(torch.empty((0, 4), device=device))
                continue

            # 像素级中心坐标
            y_token = idx.float() // num_tokens_w
            x_token = idx.float() % num_tokens_w
            y_center = (y_token + 0.5) * stride_y
            x_center = (x_token + 0.5) * stride_x

            # 计算 crop 边界并 clamp 到图像范围
            half = self.crop_size / 2.0
            y1 = torch.clamp(y_center - half, min=0)
            y2 = torch.clamp(y1 + self.crop_size, max=H)
            y1 = torch.clamp(y2 - self.crop_size, min=0)      # 修正越界回退

            x1 = torch.clamp(x_center - half, min=0)
            x2 = torch.clamp(x1 + self.crop_size, max=W)
            x1 = torch.clamp(x2 - self.crop_size, min=0)

            boxes_list.append(torch.stack([x1, y1, x2, y2], dim=-1))

        total_rois = sum(counts_per_batch)
        if total_rois == 0:
            return torch.zeros((B, 0, self.hidden_dim), device=device)

        # ── 3. GPU 级 roi_align: 一次性 C 层面并行截取+缩放 ──
        #     spatial_scale=1.0 因为 boxes 是原图像素坐标
        batched_crops = ops.roi_align(
            images, boxes_list,
            output_size=(self.crop_size, self.crop_size),
            spatial_scale=1.0,
            aligned=True,
        )  # [total_rois, C, crop_size, crop_size]

        # ── 4. 一次性 backbone 前向 (关键加速) ──
        feats, _ = backbone(batched_crops)  # [total_rois, N_patches, D]
        N_patches, D = feats.shape[1], feats.shape[2]

        # ── 5. 按 batch 拆分并填充对齐 ──
        max_k = max(counts_per_batch)
        padded_feats = torch.zeros((B, max_k * N_patches, D), device=device)

        ptr = 0
        for b, K_b in enumerate(counts_per_batch):
            if K_b > 0:
                b_feats = feats[ptr:ptr + K_b]               # [K_b, Np, D]
                b_feats = b_feats.reshape(K_b * N_patches, D) # [K_b*Np, D]
                padded_feats[b, :K_b * N_patches] = b_feats
                ptr += K_b

        return padded_feats  # [B, max_k * N_patches, D]
