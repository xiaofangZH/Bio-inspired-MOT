import torch
import torch.nn as nn
import logging
from hmat.modeling.backbone.dinov3_wrapper import DINOv3Wrapper
from hmat.modeling.encoder.hierarchical_encoder import HierarchicalVisualEncoder
from hmat.modeling.decoder.memory_decoder import MemoryAugmentedDecoder
from hmat.modeling.memory.memory_bank import MemoryBank
from hmat.modeling.memory.batch_memory_bank import BatchMemoryBank
from hmat.modeling.heads.multitask_heads import MultitaskHeads


class HMAT(nn.Module):
    """
    Hierarchical Memory-Augmented Tracker (HMAT)

    End-to-end Transformer architecture for underwater small object tracking.

    Components:
    1. Hierarchical Visual Encoder: Efficient multi-level feature extraction
    2. Memory-Augmented Decoder: Context-aware decoding with gated memory attention
    3. Multi-task Heads: Joint detection, tracking, and segmentation
    4. Recurrent Memory Bank: GRU-based feature evolution
    """

    def __init__(self, num_classes=1, hidden_dim=256, num_queries=300,
                 num_detect_queries=100, max_track_age=30, with_mask=False,
                 backbone_weights=None, use_batch_memory=False,
                 max_memory_size=5, max_batch_size=32):
        super().__init__()

        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.num_detect_queries = num_detect_queries
        self.num_track_queries = num_queries - num_detect_queries
        self.use_batch_memory = use_batch_memory
        self.max_memory_size = max_memory_size
        self.max_batch_size = max_batch_size


        # 1. Backbone: DINOv3 (Vision Transformer)
        self.backbone = DINOv3Wrapper(
            output_dim=hidden_dim,
            weights_path=backbone_weights,
            use_checkpoint=True
        )

        # Expose patch_size for downstream modules
        self.patch_size = getattr(self.backbone, 'patch_size', 16)

        # 2. Hierarchical Encoder
        self.encoder = HierarchicalVisualEncoder(
            hidden_dim=hidden_dim,
            num_levels=2,
            num_rois=10,
            patch_size=self.patch_size,
        )

        # 3. Memory-Augmented Decoder
        self.decoder = MemoryAugmentedDecoder(
            d_model=hidden_dim,
            nhead=8,
            num_layers=6,
            dim_feedforward=1024,
            dropout=0.2
        )

        # 4. Multi-task Prediction Heads
        self.heads = MultitaskHeads(
            d_model=hidden_dim,
            num_classes=num_classes,
            num_queries=num_queries,
            with_mask=with_mask,
            with_track_id=True
        )

        # 5. Memory Bank for tracking
        # Dual-mode: BatchMemoryBank for training, MemoryBank for inference
        if use_batch_memory:
            self.memory_bank = BatchMemoryBank(
                hidden_dim=hidden_dim,
                max_memory_size=max_memory_size,
                max_batch_size=max_batch_size
            )
        else:
            self.memory_bank = MemoryBank(
                hidden_dim=hidden_dim,
                max_age=max_track_age
            )

        # 7. Learnable Detection Queries
        # Content embeddings (what to detect)
        self.detect_queries_embed = nn.Embedding(
            self.num_detect_queries, hidden_dim
        )
        # Position embeddings (where to detect) - 为queries添加空间分工能力
        self.detect_queries_pos = nn.Embedding(
            self.num_detect_queries, hidden_dim
        )

        # Box → Position 编码器: 将 bbox (cx,cy,w,h) 映射到与 detect_pos 相同的空间
        self.box_pos_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, images, targets=None):
        """
        Forward pass through HAMT.

        Args:
            images: [B, C, H, W] input images (supports B>1 with BatchMemoryBank)
            targets: Optional list of target annotations for training

        Returns:
            outputs: List of prediction dicts for each frame/query
        """
        # 1. Extract hierarchical features from backbone through encoder
        src_feats, src_mask, src_pos = self.encoder(self.backbone, images)
        # src_feats: [B, N_global+N_local, D]
        B = images.shape[0]

        # 2. Prepare track queries from memory (move to same device as input)
        track_embeds, track_ids, track_boxes = self.memory_bank.get_track_queries()
        dev = images.device
        if len(track_ids) > 0:
            track_embeds = track_embeds.to(dev).unsqueeze(0).expand(B, -1, -1)
            # 为track queries生成位置编码（基于上一帧的bbox）
            track_pos = self._generate_pos_from_boxes(track_boxes.detach(), dev).unsqueeze(0).expand(B, -1, -1)
        else:
            track_embeds = torch.empty((B, 0, self.hidden_dim), device=dev)
            track_pos = torch.empty((B, 0, self.hidden_dim), device=dev)

        # 3. Get detect queries (learnable)
        detect_queries = self.detect_queries_embed.weight.unsqueeze(
            0).expand(B, -1, -1)
        # detect_queries: [B, N_detect, D]
        detect_pos = self.detect_queries_pos.weight.unsqueeze(
            0).expand(B, -1, -1)
        # detect_pos: [B, N_detect, D]

        # 4. Combine track and detect queries
        if track_embeds.shape[1] > 0:
            all_queries = torch.cat([track_embeds, detect_queries], dim=1)
            # all_queries: [B, N_track + N_detect, D]
        else:
            all_queries = detect_queries

        # 5. Get memory values for cross-attention with history
        if self.use_batch_memory:
            # 批量模式：每个样本有独立的记忆
            if self.memory_bank.memory_count[:B].sum() > 0:
                memory_values = self.memory_bank.memory.data[:B].to(dev)  # [B, K, D]
            else:
                memory_values = None
        else:
            memory_values = self.memory_bank.get_memory_values()
            if memory_values is not None:
                memory_values = memory_values.to(dev).expand(B, -1, -1)

        # 6. Decode with memory augmentation
        output_embeds = self.decoder(
            src_feats, src_mask, src_pos,
            detect_queries=detect_queries,
            track_queries=track_embeds if track_embeds.shape[1] > 0 else None,
            detect_pos=detect_pos,
            track_pos=track_pos if track_embeds.shape[1] > 0 else None,
            memory_values=memory_values
        )
        # output_embeds: [B, N_track + N_detect, D]

        # 7. Generate predictions
        outputs = self.heads(output_embeds)
        # outputs: dict with 'pred_logits', 'pred_boxes', 'pred_masks' (optional), etc.

        # 8. Update memory bank
        # Memory update logic:
        # - BatchMemoryBank mode: Always update (training needs memory for gates)
        # - MemoryBank mode: Only update during inference (original behavior)
        if self.use_batch_memory or not self.training:
            if self.training and targets is not None:
                self._update_memory(output_embeds, outputs, track_ids, targets)
            else:
                self._update_memory(output_embeds, outputs, track_ids)

        outputs = self._sanitize_outputs(outputs)
        return [outputs]  # Return list for compatibility with training loop

    @staticmethod
    def _sanitize_outputs(outputs):
        """清理输出中的 NaN/Inf，防止传播到损失函数。"""
        for key in list(outputs.keys()):
            v = outputs[key]
            if isinstance(v, torch.Tensor):
                if not torch.isfinite(v).all():
                    v = torch.nan_to_num(v, nan=0.0, posinf=1e6, neginf=-1e6)
                    outputs[key] = v
        return outputs


    def _generate_pos_from_boxes(self, boxes, device):
        """
        Generate position embeddings from bounding boxes using learnable MLP.

        Args:
            boxes: [N, 4] tensor of boxes (cx, cy, w, h) normalized to [0, 1]
            device: target device

        Returns:
            pos_embeds: [N, D] position embeddings in same space as detect_pos
        """
        if boxes.numel() == 0:
            return torch.empty((0, self.hidden_dim), device=device)

        boxes = boxes.to(device)
        return self.box_pos_encoder(boxes)  # [N, D]


    def forward_with_embeds(self, images, memory_override=None):
        """
        Like forward(), but also returns decoder output embeddings.
        Used in Stage 2 temporal association training.

        Args:
            images: [B, C, H, W]
            memory_override: [B, N, D] optional tensor to use instead of
                             memory bank (with gradient, for temporal loss).
                             If None, reads from memory bank as usual.

        Returns:
            output_embeds: [B, Nq, D]  decoder output embeddings (with grad)
            outputs_list:  [dict]      same as forward() return
        """
        B, C, H, W = images.shape
        src_feats, src_mask, src_pos = self.encoder(self.backbone, images)

        track_embeds, track_ids, track_boxes = self.memory_bank.get_track_queries()
        dev = images.device
        if len(track_ids) > 0:
            track_embeds = track_embeds.to(dev).unsqueeze(0).expand(B, -1, -1)
            track_pos = self._generate_pos_from_boxes(
                track_boxes.detach(), dev).unsqueeze(0).expand(B, -1, -1)
        else:
            track_embeds = torch.empty((B, 0, self.hidden_dim), device=dev)
            track_pos = torch.empty((B, 0, self.hidden_dim), device=dev)

        detect_queries = self.detect_queries_embed.weight.unsqueeze(0).expand(B, -1, -1)
        detect_pos = self.detect_queries_pos.weight.unsqueeze(0).expand(B, -1, -1)

        if track_embeds.shape[1] > 0:
            all_queries = torch.cat([track_embeds, detect_queries], dim=1)
        else:
            all_queries = detect_queries

        # Use memory_override if provided (gradient path for temporal training)
        if memory_override is not None:
            memory_values = memory_override   # [B, N, D] — keeps gradient
        elif self.use_batch_memory:
            if self.memory_bank.memory_count[:B].sum() > 0:
                memory_values = self.memory_bank.memory.data[:B].to(dev)
            else:
                memory_values = None
        else:
            memory_values = self.memory_bank.get_memory_values()
            if memory_values is not None:
                memory_values = memory_values.to(dev).expand(B, -1, -1)

        output_embeds = self.decoder(
            src_feats, src_mask, src_pos,
            detect_queries=detect_queries,
            track_queries=track_embeds if track_embeds.shape[1] > 0 else None,
            detect_pos=detect_pos,
            track_pos=track_pos if track_embeds.shape[1] > 0 else None,
            memory_values=memory_values
        )
        outputs = self.heads(output_embeds)
        # Do NOT update memory bank here (temporal training manages memory externally)
        return output_embeds, [outputs]

    def _update_memory(self, output_embeds, outputs, track_ids, targets=None):
        """
        Update memory bank with new observations.

        Args:
            output_embeds: [B, N_track+N_detect, D] decoder outputs
            outputs: dict with predictions
            track_ids: list of current track IDs
            targets: Optional ground truth for supervised update
        """
        num_tracks = len(track_ids)

        if num_tracks > 0:
            track_embeds = output_embeds[:, :num_tracks, :]
            track_preds = {
                k: v[:, :num_tracks, ...] if v.shape[1] > 0 else v
                for k, v in outputs.items()
            }
        else:
            track_embeds = output_embeds[:, :, :].clone() * 0  # Empty tensor
            track_preds = {}

        detect_embeds = output_embeds[:, num_tracks:, :]
        detect_preds = {
            k: v[:, num_tracks:, ...] if v.shape[1] > 0 else v
            for k, v in outputs.items()
        }

        # Update memory bank
        self.memory_bank.update(
            track_embeds,
            track_preds,
            detect_embeds,
            detect_preds,
            targets=targets
        )

    def set_train_mode(self, freeze_backbone=True):
        """兼容接口：当前固定为全参数训练，忽略 freeze_backbone 参数。"""
        self.train()
        for param in self.parameters():
            param.requires_grad = True
        logging.info("Full-parameter training enabled (freeze_backbone ignored)")

    def configure_for_stage(self, stage):
        """兼容接口：当前固定为全参数训练，不再执行三阶段冻结策略。"""
        self.train()
        for param in self.parameters():
            param.requires_grad = True

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        logging.info(
            "Stage argument=%s ignored. Full-parameter training: %.2fM / %.2fM (%.1f%%)",
            stage,
            trainable / 1e6,
            total / 1e6,
            100 * trainable / max(total, 1),
        )

    def get_parameter_groups(self, base_lr=1e-4, backbone_lr_scale=1.0):
        """
        获取分层学习率的参数组。

        Args:
            base_lr: 新增模块的基础学习率
            backbone_lr_scale: backbone学习率缩放因子 (Stage 3 用 0.1)
        """
        backbone_vit_params = []
        other_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if 'backbone.vit.' in name:
                backbone_vit_params.append(param)
            else:
                other_params.append(param)

        param_groups = []
        if other_params:
            param_groups.append({
                "params": other_params,
                "lr": base_lr,
                "name": "new_modules",
            })
        if backbone_vit_params:
            param_groups.append({
                "params": backbone_vit_params,
                "lr": base_lr * backbone_lr_scale,
                "name": "backbone_vit",
            })
        return param_groups


class HMATModel(HMAT):
    pass

# Compatibility alias for import consistency
HAMT=HMAT
