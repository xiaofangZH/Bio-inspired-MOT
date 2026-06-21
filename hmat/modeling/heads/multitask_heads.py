import torch
import torch.nn as nn
import torch.nn.functional as F


class MultitaskHeads(nn.Module):
    """
    Multi-task prediction heads (v3).

    改进:
    - 共享语义 trunk，让 class/bbox/seg 三个分支从共同特征中学习
    - track_id_head 替换为 ReID embedding head（L2 归一化向量）
      解决固定分类器无法处理动态 ID 数量的结构性问题
    - 全部使用 GELU 激活函数（非 ReLU）
    """

    def __init__(self, d_model=256, num_classes=1, num_queries=300,
                 with_mask=True, with_track_id=True, reid_dim=128):
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes
        self.with_mask = with_mask
        self.with_track_id = with_track_id

        # ── Shared semantic trunk ──
        self.shared_trunk = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # ── Task-specific heads ──

        # Classification head (Focal Loss: num_classes dims, background implicit in sigmoid)
        self.class_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, num_classes),
        )

        # Bounding box regression head
        self.bbox_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, 4),
        )

        # Optional: Segmentation mask head
        if self.with_mask:
            self.mask_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Linear(d_model, 784),
            )

        # Optional: ReID embedding head (replaces fixed-size track_id classifier)
        if self.with_track_id:
            self.reid_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Linear(d_model, reid_dim),
            )

    def forward(self, queries):
        """
        Args:
            queries: [B, N_queries, D]  decoder output embeddings

        Returns:
            outputs: dict containing predictions for each task
        """
        # ── Shared semantic trunk ──
        shared = self.shared_trunk(queries)  # [B, N, D]

        # ── 1. Classification logits ──
        class_logits = self.class_head(shared)  # [B, N, num_classes+1]

        # ── 2. Bounding box predictions ──
        bbox_preds = self.bbox_head(shared)  # [B, N, 4]
        bbox_preds = torch.sigmoid(bbox_preds)  # normalize to [0, 1]

        outputs = {
            'pred_logits': class_logits,
            'pred_boxes': bbox_preds,
        }

        # ── 3. Optional: Segmentation masks ──
        if self.with_mask:
            mask_preds = self.mask_head(shared)  # [B, N, 784]
            mask_preds = mask_preds.reshape(-1, mask_preds.shape[1], 28, 28)
            mask_preds = torch.sigmoid(mask_preds)
            outputs['pred_masks'] = mask_preds

        # ── 4. Optional: ReID embeddings (L2-normalized) ──
        if self.with_track_id:
            reid_embeds = self.reid_head(shared)               # [B, N, reid_dim]
            reid_embeds = F.normalize(reid_embeds, p=2, dim=-1)
            outputs['reid_embeds'] = reid_embeds

        return outputs


class DecisionHeads(nn.Module):
    """
    Auxiliary heads for decision making (e.g., birth/death decisions for tracks).
    """

    def __init__(self, d_model=256):
        super().__init__()

        self.is_alive_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )

        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )

    def forward(self, queries):
        """
        Predict auxiliary information for better memory management.

        Args:
            queries: [B, N, D]

        Returns:
            Dict with is_alive and confidence scores
        """
        is_alive = self.is_alive_head(queries).squeeze(-1)  # [B, N]
        confidence = self.confidence_head(queries).squeeze(-1)  # [B, N]

        return {
            'is_alive': is_alive,
            'confidence': confidence,
        }
