import torch
import torch.nn as nn
import logging
import math
from hmat.modeling.backbone.dinov3_wrapper import DINOv3Wrapper
from hmat.modeling.encoder.hierarchical_encoder import HierarchicalVisualEncoder
from hmat.modeling.decoder.memory_decoder import MemoryAugmentedDecoder
from hmat.modeling.memory.memory_bank import MemoryBank
from hmat.modeling.memory.batch_memory_bank import BatchMemoryBank
from hmat.modeling.heads.multitask_heads import MultitaskHeads
from hmat.modeling.loss.matcher import HungarianMatcher


class BoxPosEncoder(nn.Module):
    """
    Sinusoidal Box Positional Encoding.

    将 4 维欧几里得坐标 (cx, cy, w, h) 通过正弦/余弦展开到 hidden_dim 维度，
    解决单纯 nn.Linear 难以学习高频空间细节的"谱偏差"问题。

    参考 DETR 对归一化坐标的标量位置编码方案。
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.d_per_coord = hidden_dim // 4  # 每个坐标分量分配的维度
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def _sine_cosine_expand(self, boxes):
        """
        对每个 4 维归一化坐标分量，使用不同频率的正弦/余弦函数展开到高维。

        PE(x, 2i)   = sin(2π * x / 10000^{2i / D})
        PE(x, 2i+1) = cos(2π * x / 10000^{2i / D})

        x 先乘以 2π 跨越完整正弦周期，使神经网络能区分微小空间差异。
        """
        N = boxes.shape[0]
        device = boxes.device
        dtype = boxes.dtype
        D = self.d_per_coord  # e.g. 64 for hidden_dim=256

        # 频率缩放因子: [1, 10000^{-2/D}, 10000^{-4/D}, ..., 10000^{-2(D//2-1)/D}]
        div_term = torch.exp(
            torch.arange(0, D // 2, device=device, dtype=dtype) *
            (-math.log(10000.0) / (D // 2))
        )  # [D//2]

        # 先放大到 2π 范围，使高频频段可区分微小空间差异
        boxes_scaled = boxes * (2 * math.pi)

        expanded = []
        for i in range(4):
            coord = boxes_scaled[:, i:i + 1]  # [N, 1]
            scaled = coord * div_term.unsqueeze(0)  # [N, D//2]
            expanded.append(torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1))

        return torch.cat(expanded, dim=-1)  # [N, 4*D] = [N, hidden_dim]

    def forward(self, boxes):
        """
        Args:
            boxes: [N, 4] tensor (cx, cy, w, h) normalized to [0, 1]
        Returns:
            pos_embeds: [N, D]
        """
        expanded_pos = self._sine_cosine_expand(boxes)
        return self.mlp(expanded_pos)


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
                max_batch_size=max_batch_size,
                max_track_age=max_track_age,
            )
        else:
            self.memory_bank = MemoryBank(
                hidden_dim=hidden_dim,
                max_age=max_track_age
            )

        # 6. Projection: memory hidden_dim (256) → reid_dim (128)
        #    Memory bank stores decoder features (hidden_dim) for GRU update,
        #    but cross-frame ReID loss expects reid_dim-matched embeddings.
        self.memory_reid_proj = nn.Linear(hidden_dim, hidden_dim // 2, bias=False)

        # 7. Learnable Detection Queries
        # Content embeddings (what to detect)
        self.detect_queries_embed = nn.Embedding(
            self.num_detect_queries, hidden_dim
        )
        # Position embeddings (where to detect) - 为queries添加空间分工能力
        self.detect_queries_pos = nn.Embedding(
            self.num_detect_queries, hidden_dim
        )

        # Box → Position 编码器: 将 bbox (cx,cy,w,h) 通过正弦/余弦展开到高维
        self.box_pos_encoder = BoxPosEncoder(hidden_dim)

    def forward(self, images, targets=None, teacher_force_prob=1.0):
        """
        Forward pass through HAMT.

        Args:
            images: [B, C, H, W] input images (supports B>1 with BatchMemoryBank)
            targets: Optional list of target annotations for training
            teacher_force_prob: Teacher Forcing 概率 (1.0=全量TF, <1.0=Scheduled Sampling)

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
        memory_ages = None
        memory_mask_dec = None
        if self.use_batch_memory:
            # 批量模式：每个样本有独立的记忆
            if self.memory_bank.memory_count[:B].sum() > 0:
                memory_values = self.memory_bank.memory.data[:B].to(dev)  # [B, K, D]
                # 传递 slot ages 用于时序偏置，memory mask 用于屏蔽空槽
                memory_ages = self.memory_bank.slot_ages[:B].to(dev)
                memory_mask_dec = (self.memory_bank.memory_mask[:B] == 0).to(dev)  # True=pad
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
            memory_values=memory_values,
            memory_ages=memory_ages,
            memory_mask=memory_mask_dec,
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
                self._update_memory(output_embeds, outputs, track_ids, targets,
                                    teacher_force_prob=teacher_force_prob)
            else:
                self._update_memory(output_embeds, outputs, track_ids)

        # 9. Attach memory bank track data for cross-frame ReID loss
        if self.training and hasattr(self.memory_bank, 'get_active_track_data'):
            mem_embs, mem_tids, mem_ages = self.memory_bank.get_active_track_data()
            if mem_embs is not None:
                # Project memory from hidden_dim (256) → reid_dim (128) to match ReID head
                outputs['memory_embeds'] = self.memory_reid_proj(mem_embs)
                outputs['memory_track_ids'] = mem_tids
                outputs['memory_ages'] = mem_ages

        # 10. Attach tracking metadata for two-stage matching in criterion
        outputs['num_tracks'] = len(track_ids)
        outputs['track_query_ids'] = torch.as_tensor(
            track_ids, dtype=torch.long, device=dev
        ) if len(track_ids) > 0 else torch.empty((0,), dtype=torch.long, device=dev)

        return [outputs]  # Return list for compatibility with training loop

    @staticmethod
    def _sanitize_outputs(outputs):
        """DEPRECATED: 此函数会切断 NaN 位置的梯度流,导致模型假死。现已禁用。"""
        return outputs  # 直通，不做任何修改



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
            memory_ages = None
            memory_mask_dec = None
        elif self.use_batch_memory:
            if self.memory_bank.memory_count[:B].sum() > 0:
                memory_values = self.memory_bank.memory.data[:B].to(dev)
                memory_ages = self.memory_bank.slot_ages[:B].to(dev)
                memory_mask_dec = (self.memory_bank.memory_mask[:B] == 0).to(dev)
            else:
                memory_values = None
                memory_ages = None
                memory_mask_dec = None
        else:
            memory_values = self.memory_bank.get_memory_values()
            memory_ages = None
            memory_mask_dec = None
            if memory_values is not None:
                memory_values = memory_values.to(dev).expand(B, -1, -1)

        output_embeds = self.decoder(
            src_feats, src_mask, src_pos,
            detect_queries=detect_queries,
            track_queries=track_embeds if track_embeds.shape[1] > 0 else None,
            detect_pos=detect_pos,
            track_pos=track_pos if track_embeds.shape[1] > 0 else None,
            memory_values=memory_values,
            memory_ages=memory_ages,
            memory_mask=memory_mask_dec,
        )
        outputs = self.heads(output_embeds)
        # Do NOT update memory bank here (temporal training manages memory externally)

        # Attach tracking metadata for two-stage matching in criterion
        outputs['num_tracks'] = len(track_ids)
        outputs['track_query_ids'] = torch.as_tensor(
            track_ids, dtype=torch.long, device=dev
        ) if len(track_ids) > 0 else torch.empty((0,), dtype=torch.long, device=dev)

        return output_embeds, [outputs]

    def _update_memory(self, output_embeds, outputs, track_ids, targets=None,
                       teacher_force_prob=1.0):
        """
        Update memory bank with new observations.

        训练时 (self.training + targets provided):
          使用 HungarianMatcher 匹配 detect_preds → GT，
          以 GT 的 track_id 做 Teacher Forcing 入库。
          Scheduled Sampling: 以 teacher_force_prob 概率使用 TF，
          其余使用 IoU 贪心匹配。

        推理时 (self.training == False):
          使用 IoU 贪心匹配推断 track_id。

        Args:
            output_embeds: [B, N_track+N_detect, D] decoder outputs
            outputs: dict with predictions
            track_ids: list of current track IDs
            targets: Optional ground truth for supervised update
            teacher_force_prob: Teacher Forcing 概率 (1.0=全量, 0.0=关闭)
        """
        num_tracks = len(track_ids)

        if num_tracks > 0:
            track_embeds = output_embeds[:, :num_tracks, :]
            track_preds = {
                k: v[:, :num_tracks, ...] if v.shape[1] > 0 else v
                for k, v in outputs.items()
            }
        else:
            track_embeds = output_embeds[:, :0, :]  # 空张量切片，无幽灵轨迹
            track_preds = {}

        detect_embeds = output_embeds[:, num_tracks:, :]
        detect_preds = {
            k: v[:, num_tracks:, ...] if v.shape[1] > 0 else v
            for k, v in outputs.items()
        }

        # ── Teacher Forcing / Scheduled Sampling ──
        teacher_force_ids = None
        if self.training and detect_embeds.shape[1] > 0 and targets is not None:
            use_tf = torch.rand(1).item() < teacher_force_prob
            if use_tf:
                teacher_force_ids = self._build_teacher_force_ids(
                    detect_preds, targets, detect_embeds.device)

        # Update memory bank
        self.memory_bank.update(
            track_embeds,
            track_preds,
            detect_embeds,
            detect_preds,
            targets=targets,
            teacher_force_ids=teacher_force_ids,
        )

    def _build_teacher_force_ids(self, detect_preds, targets, device):
        """
        运行 HungarianMatcher 匹配 detect queries → GT，提取 GT track_id。

        Returns:
            list[Tensor]: 每个 batch 一个 [N_detect] 张量，
                          matched → GT track_id, unmatched → -1
        """
        B = detect_preds['pred_logits'].shape[0]
        N_det = detect_preds['pred_logits'].shape[1]

        # 懒初始化 HungarianMatcher（避免每帧重建）
        if not hasattr(self, '_tf_matcher'):
            self._tf_matcher = HungarianMatcher(
                cost_class=2, cost_bbox=5, cost_ciou=2)

        match_indices = self._tf_matcher(detect_preds, targets)

        teacher_force_ids = []
        for b in range(B):
            tids = torch.full((N_det,), -1, dtype=torch.long, device=device)
            src_idx, tgt_idx = match_indices[b]
            if len(src_idx) > 0 and 'track_ids' in targets[b]:
                gt_tids = targets[b]['track_ids']
                if len(gt_tids) > 0:
                    tids[src_idx.long()] = gt_tids[tgt_idx.long()]
            teacher_force_ids.append(tids)

        return teacher_force_ids

    def set_train_mode(self, freeze_backbone=True):
        """冻结/解冻模型参数 — 保护 DINOv3 预训练特征不被破坏。"""
        self.train()
        frozen_count = 0
        trainable_count = 0
        for name, param in self.named_parameters():
            if freeze_backbone and name.startswith('backbone.'):
                param.requires_grad = False
                frozen_count += param.numel()
            else:
                param.requires_grad = True
                trainable_count += param.numel()
        logging.info(
            "Train mode: backbone %s. Frozen: %.2fM, Trainable: %.2fM",
            "frozen" if freeze_backbone else "unfrozen",
            frozen_count / 1e6,
            trainable_count / 1e6,
        )

    def configure_for_stage(self, stage, freeze_backbone=True):
        """阶段配置：控制 backbone 冻结/解冻与可训参数统计。"""
        self.set_train_mode(freeze_backbone=freeze_backbone)
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        logging.info(
            "Stage=%s configured. Trainable: %.2fM / %.2fM (%.1f%%)",
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
            if name.startswith('backbone.'):
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
