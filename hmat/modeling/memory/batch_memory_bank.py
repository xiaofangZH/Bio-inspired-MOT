"""
Batch Memory Bank for HAMT Model Training

支持 batch>1 训练，每个样本独立维护 per-object 记忆槽。
v2: 从 mean-pooling 改为逐目标独立存储，保留目标区分度。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchMemoryBank(nn.Module):
    """
    批量记忆库 —— 每个样本的每个目标独立维护一个记忆槽。

    存储形状: [B, M, D]
      B = batch_size
      M = max_memory_size (最大目标数)
      D = hidden_dim

    每个槽独立通过 GRU 更新，不再做全局 mean pooling。
    """

    def __init__(self, hidden_dim, max_memory_size=5, max_batch_size=32,
                 max_track_age=30, min_conf_score=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_memory_size = max_memory_size
        self.max_batch_size = max_batch_size
        self.max_track_age = max_track_age      # 连续未匹配帧数阈值
        self.min_conf_score = min_conf_score    # 最低置信度阈值

        # Memory storage: [max_batch_size, max_memory_size, hidden_dim]
        self.register_parameter(
            'memory',
            nn.Parameter(
                torch.zeros(max_batch_size, max_memory_size, hidden_dim),
                requires_grad=False
            )
        )

        # Slot active mask: [max_batch_size, max_memory_size]  (1=active, 0=empty)
        self.register_parameter(
            'memory_mask',
            nn.Parameter(
                torch.zeros(max_batch_size, max_memory_size),
                requires_grad=False
            )
        )

        # Slot occupation counters: [max_batch_size]  — how many active slots per batch
        self.register_buffer(
            'slot_count',
            torch.zeros(max_batch_size, dtype=torch.long)
        )

        # Age per slot: [max_batch_size, max_memory_size]
        self.register_buffer(
            'slot_ages',
            torch.zeros(max_batch_size, max_memory_size, dtype=torch.float)
        )

        # Track box coordinates per slot [cx, cy, w, h] normalized
        self.register_buffer(
            'track_boxes',
            torch.zeros(max_batch_size, max_memory_size, 4)
        )

        # Track ID per slot (-1 = unknown/empty)
        self.register_buffer(
            'track_ids',
            torch.full((max_batch_size, max_memory_size), -1, dtype=torch.long)
        )

        # Track confidence per slot [0, 1] (用于低置信度清理)
        self.register_buffer(
            'slot_scores',
            torch.zeros(max_batch_size, max_memory_size)
        )

        # Decay coefficient
        self.memory_decay_coeff = 1.0

        # EMA momentum for per-slot smooth update (无参数，无需训练)
        self.momentum = 0.8  # 越大越依赖历史记忆，越小越偏向新特征

        # 兼容 MemoryBank 接口
        self.memory_ptr = torch.zeros(max_batch_size, dtype=torch.long)
        self.memory_count = self.slot_count  # alias

    @torch.no_grad()
    def reset_batch(self, batch_size):
        """重置指定 batch 的记忆."""
        if batch_size > self.max_batch_size:
            raise ValueError(
                f"batch_size {batch_size} exceeds max_batch_size {self.max_batch_size}"
            )
        self.memory[:batch_size].zero_()
        self.memory_mask[:batch_size].zero_()
        self.slot_count[:batch_size] = 0
        self.slot_ages[:batch_size].zero_()
        self.track_boxes[:batch_size].zero_()
        self.track_ids[:batch_size].fill_(-1)
        self.slot_scores[:batch_size].zero_()

    def update(self, track_embeds, track_preds, detect_embeds, detect_preds, targets=None, teacher_force_ids=None):
        """
        更新记忆 —— 逐目标独立存储。

        Args:
            track_embeds:  [B, N_track, D]   已有 track 的 decoder 输出
            track_preds:  dict with 'pred_boxes' etc.
            detect_embeds: [B, N_detect, D]  新检测 query 的输出
            detect_preds:  dict with 'pred_boxes' etc.
            targets:       Optional ground truth
            teacher_force_ids: Optional list[Tensor] per batch, each [N_detect]
                               GT track_id for each detect query (-1 = unmatched).
                               When provided, bypasses IoU greedy matching entirely.
        """
        B = track_embeds.size(0)
        assert B <= self.max_batch_size, f"batch_size {B} > max_batch_size {self.max_batch_size}"

        # ── 更新已有 track 的记忆 ──
        if track_embeds.size(1) > 0:
            self._update_existing_slots(track_embeds, track_preds)

        # ── 为新检测创建新记忆槽 (Teacher Forcing 或 IoU 贪心) ──
        if detect_embeds.size(1) > 0:
            self._create_new_slots(detect_embeds, detect_preds, targets, teacher_force_ids)

        # ── 所有激活槽 age+1 ──
        self.slot_ages[:B] = self.slot_ages[:B] + 1.0

        # ── 超过 max_memory_size 的旧槽移除 ──
        self._prune_oldest(B)

        # ── 生命消融: 清理过期/低置信度死轨迹 ──
        self._prune_by_age(B)

    def _update_existing_slots(self, track_embeds, track_preds):
        """
        将当前 track_embeds 逐槽写入已有记忆。

        track_embeds: [B, N_track, D]
        策略：按槽索引一一对应（前 N_track 个活跃槽）
        """
        B, N_track, D = track_embeds.shape

        for b in range(B):
            active_count = self.slot_count[b].item()
            n_update = min(N_track, active_count)

            if n_update == 0:
                continue

            # 获取活跃槽的旧记忆
            old_mem = self.memory[b, :n_update]            # [n_update, D]
            new_feat = track_embeds[b, :n_update]         # [n_update, D]

            # EMA 动量更新 (无参数，天然可训练——梯度通过 new_feat 回传)
            # updated = momentum * old_mem + (1-momentum) * new_feat
            # old_mem 来自 buffer（无梯度），new_feat.detach() 后参与融合
            updated = (
                self.momentum * old_mem +
                (1.0 - self.momentum) * new_feat.detach()
            )

            # NaN guard: 跳过异常更新，保留旧记忆
            if not torch.isfinite(updated).all():
                continue

            # 写入并 detach
            self.memory.data[b, :n_update] = updated.detach()
            self.slot_ages[b, :n_update] = 0.0  # 重置年龄

            # 记录匹配置信度
            if 'pred_logits' in track_preds:
                logits = track_preds['pred_logits'][b, :n_update]    # [n_update, 1]
                scores = torch.sigmoid(logits)[:, 0].detach()        # 前景概率
                scores = torch.nan_to_num(scores, nan=0.0).clamp(0.0, 1.0)
                self.slot_scores[b, :n_update] = scores

            # 存储真实 bbox 坐标
            if 'pred_boxes' in track_preds and track_preds['pred_boxes'].shape[1] >= n_update:
                boxes = track_preds['pred_boxes'][b, :n_update].detach()
                # NaN guard
                boxes = torch.nan_to_num(boxes, nan=0.5, posinf=1.0, neginf=0.0)
                boxes = boxes.clamp(0.0, 1.0)
                self.track_boxes[b, :n_update] = boxes

    def _create_new_slots(self, detect_embeds, detect_preds, targets=None, teacher_force_ids=None):
        """
        将高置信度检测创建为新记忆槽，并分配 track_id。

        Teacher Forcing 模式 (teacher_force_ids provided):
          直接使用外部传入的 GT track_id，跳过 IoU 贪心匹配，
          彻底消除交叉遮挡场景下的身份错乱风险。

        IoU 匹配模式 (teacher_force_ids is None):
          用 IoU 贪心匹配推断 track_id（仅用于推理）。

        detect_embeds: [B, N_detect, D]
        targets: optional list of dicts with 'boxes' and 'track_ids' for GT
        teacher_force_ids: optional list[Tensor] per batch, each [N_detect]
        """
        B, N_det, D = detect_embeds.shape

        # 获取置信度
        if 'pred_logits' in detect_preds:
            logits = detect_preds['pred_logits']  # [B, N_det, 1]
            scores = torch.sigmoid(logits)[:, :, 0]  # 前景概率
        else:
            scores = torch.ones(B, N_det, device=detect_embeds.device) * 0.5

        conf_threshold = 0.4

        for b in range(B):
            n_active = self.slot_count[b].item()
            available = self.max_memory_size - n_active
            if available <= 0:
                continue

            high_conf = (scores[b] > conf_threshold).nonzero(as_tuple=True)[0]
            if high_conf.numel() == 0:
                continue

            # 只取能装下的高置信度检测
            n_new = min(len(high_conf), available)
            new_indices = high_conf[:n_new]

            # ─── Teacher Forcing: 使用预分配的 track_id ───
            forced_tids = None
            if teacher_force_ids is not None and b < len(teacher_force_ids):
                forced_tids = teacher_force_ids[b]  # [N_detect], -1 = unmatched

            # ─── IoU 模式: 准备 GT 数据 ───
            gt_boxes = None
            gt_tids = None
            if forced_tids is None and targets is not None and b < len(targets) and 'boxes' in targets[b]:
                tgt = targets[b]
                if 'track_ids' in tgt and len(tgt['boxes']) > 0:
                    gt_boxes = tgt['boxes']       # [Ng, 4] cxcywh
                    gt_tids = tgt['track_ids']    # [Ng]

            start_slot = n_active
            for i, det_idx in enumerate(new_indices):
                slot_idx = start_slot + i
                emb = detect_embeds[b, det_idx].detach()
                # NaN guard: 跳过异常检测嵌入
                if not torch.isfinite(emb).all():
                    continue
                self.memory.data[b, slot_idx] = emb
                self.memory_mask.data[b, slot_idx] = 1.0
                self.slot_ages[b, slot_idx] = 0.0
                self.slot_scores[b, slot_idx] = scores[b, det_idx].detach().clamp(0.0, 1.0)

                # 存储新检测的 bbox
                det_cxcywh = None
                if 'pred_boxes' in detect_preds and detect_preds['pred_boxes'].shape[1] > det_idx:
                    box = detect_preds['pred_boxes'][b, det_idx].detach()
                    box = torch.nan_to_num(box, nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
                    self.track_boxes[b, slot_idx] = box
                    det_cxcywh = box  # [4] cx,cy,w,h

                # ─── Teacher Forcing 模式 ───
                tid = -1
                if forced_tids is not None:
                    tid = int(forced_tids[det_idx].item())
                # ─── IoU 贪心匹配模式 (fallback for inference) ───
                elif gt_boxes is not None and det_cxcywh is not None:
                    det_xyxy = self._cxcywh_to_xyxy(det_cxcywh.unsqueeze(0))  # [1, 4]
                    gt_xyxy = self._cxcywh_to_xyxy(gt_boxes)                  # [Ng, 4]
                    ious = self._box_iou(det_xyxy, gt_xyxy)[0]                # [Ng]
                    if ious.numel() > 0:
                        best_iou, best_idx = ious.max(0)
                        if best_iou > 0.3:
                            tid = gt_tids[best_idx].item()
                self.track_ids[b, slot_idx] = tid

            self.slot_count[b] += n_new

    @staticmethod
    def _cxcywh_to_xyxy(bbox):
        cx, cy, w, h = bbox.unbind(-1)
        x1 = cx - 0.5 * w
        y1 = cy - 0.5 * h
        x2 = cx + 0.5 * w
        y2 = cy + 0.5 * h
        return torch.stack([x1, y1, x2, y2], dim=-1)

    @staticmethod
    def _box_iou(boxes1, boxes2):
        if boxes1.numel() == 0 or boxes2.numel() == 0:
            return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))
        area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
        area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, :, 0] * wh[:, :, 1]
        union = area1[:, None] + area2 - inter + 1e-6
        return inter / union

    @torch.no_grad()
    def _prune_oldest(self, B):
        """移除超出 max_memory_size 的最旧槽，并紧缩存储."""
        for b in range(B):
            if self.slot_count[b] <= self.max_memory_size:
                continue

            # 按 age 降序排列，保留最年轻的前 max_memory_size 个活跃槽
            ages = self.slot_ages[b, :self.slot_count[b]]
            _, keep_order = torch.sort(ages)  # 最年轻的在前
            keep_order = keep_order[:self.max_memory_size]
            keep_order = torch.sort(keep_order)[0]  # 保持原顺序

            old_mem = self.memory[b, :self.slot_count[b]].clone()
            old_mask = self.memory_mask[b, :self.slot_count[b]].clone()
            old_ages = self.slot_ages[b, :self.slot_count[b]].clone()
            old_boxes = self.track_boxes[b, :self.slot_count[b]].clone()
            old_tids = self.track_ids[b, :self.slot_count[b]].clone()

            self.memory.data[b, :self.max_memory_size] = old_mem[keep_order]
            self.memory_mask.data[b, :self.max_memory_size] = old_mask[keep_order]
            self.slot_ages[b, :self.max_memory_size] = old_ages[keep_order]
            self.track_boxes[b, :self.max_memory_size] = old_boxes[keep_order]
            self.track_ids[b, :self.max_memory_size] = old_tids[keep_order]
            self.memory.data[b, self.max_memory_size:] = 0.0
            self.memory_mask.data[b, self.max_memory_size:] = 0.0
            self.slot_ages[b, self.max_memory_size:] = 0.0
            self.track_boxes[b, self.max_memory_size:] = 0.0
            self.track_ids[b, self.max_memory_size:].fill_(-1)
            self.slot_count[b] = self.max_memory_size

    @torch.no_grad()
    def _prune_by_age(self, B):
        """
        动态生命周期消融：清理过期或低置信度的死轨迹。

        清理条件（满足任一即清除）：
        1. 连续 max_track_age 帧未被匹配 (slot_ages > max_track_age)
        2. 最近置信度低于 min_conf_score 且 age > max_track_age // 2
           （半衰期后仍低置信度 → 疑似误检）

        清理后紧缩存储，将被清除槽位从计算图彻底断开 (.detach)。
        """
        for b in range(B):
            n_active = self.slot_count[b].item()
            if n_active == 0:
                continue

            ages_b = self.slot_ages[b, :n_active]
            scores_b = self.slot_scores[b, :n_active]
            masks_b = self.memory_mask[b, :n_active]

            # 条件1: 超龄轨迹
            stale_mask = ages_b > self.max_track_age
            # 条件2: 半衰期后仍低置信度
            low_conf_mask = (ages_b > self.max_track_age // 2) & (scores_b < self.min_conf_score)
            prune_mask = stale_mask | low_conf_mask

            # 保留已非活跃的掩码（空槽不参与判断）
            active_mask = masks_b > 0
            prune_mask = prune_mask & active_mask

            if not prune_mask.any():
                continue

            keep_mask = ~prune_mask
            keep_indices = keep_mask.nonzero(as_tuple=True)[0]

            if len(keep_indices) == 0:
                # 全部清除
                self.memory.data[b, :n_active] = 0.0
                self.memory_mask.data[b, :n_active] = 0.0
                self.slot_ages[b, :n_active] = 0.0
                self.track_boxes[b, :n_active] = 0.0
                self.track_ids[b, :n_active].fill_(-1)
                self.slot_scores[b, :n_active] = 0.0
                self.slot_count[b] = 0
                continue

            n_new = len(keep_indices)

            # 紧缩存储：复制保留的槽位覆盖已清除的
            old_mem = self.memory[b, :n_active].clone()
            old_mask = self.memory_mask[b, :n_active].clone()
            old_ages = self.slot_ages[b, :n_active].clone()
            old_boxes = self.track_boxes[b, :n_active].clone()
            old_tids = self.track_ids[b, :n_active].clone()
            old_scores = self.slot_scores[b, :n_active].clone()

            self.memory.data[b, :n_new] = old_mem[keep_indices]
            self.memory_mask.data[b, :n_new] = old_mask[keep_indices]
            self.slot_ages[b, :n_new] = old_ages[keep_indices]
            self.track_boxes[b, :n_new] = old_boxes[keep_indices]
            self.track_ids[b, :n_new] = old_tids[keep_indices]
            self.slot_scores[b, :n_new] = old_scores[keep_indices]

            # 清零被清除的尾部
            self.memory.data[b, n_new:n_active] = 0.0
            self.memory_mask.data[b, n_new:n_active] = 0.0
            self.slot_ages[b, n_new:n_active] = 0.0
            self.track_boxes[b, n_new:n_active] = 0.0
            self.track_ids[b, n_new:n_active].fill_(-1)
            self.slot_scores[b, n_new:n_active] = 0.0
            self.slot_count[b] = n_new

    def get_memory(self, batch_size):
        """
        获取当前记忆和掩码（带年龄衰减）。

        Returns:
            memory: [B, M, D]
            mask:   [B, M]
        """
        mem = self.memory[:batch_size]
        mask = self.memory_mask[:batch_size]
        if self.memory_decay_coeff < 1.0 - 1e-6:
            decay = (self.memory_decay_coeff ** self.slot_ages[:batch_size]).unsqueeze(-1)
            mem = mem * decay.to(mem.dtype)
        return mem, mask

    def get_memory_count(self, batch_size):
        """每个样本的活跃槽数."""
        return self.slot_count[:batch_size]

    def get_memory_values(self):
        """
        获取记忆值用于解码器注意力（兼容 MemoryBank 接口）。

        Returns:
            memory: [1, M, D]  第一个样本的有效记忆
            None: 没有有效记忆时
        """
        if self.slot_count[0] > 0:
            n = self.slot_count[0].item()
            return self.memory.data[:1, :n]  # [1, n_valid, D]
        return None

    def get_track_queries(self):
        """
        获取 track queries (兼容 MemoryBank API).
        BatchMemoryBank 维护 per-object 槽位，直接返回活跃槽作为 track queries.
        同时返回真实存储的 bbox 坐标（非全零）。

        ⚠️ 设计约束：此方法仅返回 batch 0 的记忆。
        这要求训练时 batch_size=1（逐帧串行处理），确保每个 batch 索引
        都对应同一个序列的时间维度更新。
        若调用 forward 时 B>1，所有样本将共享 batch 0 的记忆，时序会错乱。
        """
        if self.slot_count[0] > 0:
            n = self.slot_count[0].item()
            embeds = self.memory[0, :n].clone()  # [n, D]
            ids = list(range(n))
            boxes = self.track_boxes[0, :n].clone()  # [n, 4] real coords
            return embeds, ids, boxes
        empty_embeds = torch.zeros(0, self.hidden_dim, device=self.memory.device)
        return empty_embeds, [], torch.zeros(0, 4, device=self.memory.device)

    def reset(self):
        """重置所有记忆（使用 in-place 操作，避免 nn.Parameter device 丢失）."""
        self.memory.data.zero_()
        self.memory_mask.data.zero_()
        self.slot_count.zero_()
        self.slot_ages.zero_()
        self.track_boxes.zero_()
        self.track_ids.fill_(-1)
        self.slot_scores.zero_()

    def get_active_track_data(self):
        """
        获取所有活跃记忆槽的 embedding、track_id 和 age，用于跨帧 ReID 损失。

        Returns:
            mem_embeds: [B_active, D]  — 所有活跃槽的 embedding
            mem_tids:   [B_active]     — 对应的 track_id (-1 = 未知)
            mem_ages:   [B_active]     — 对应的 slot age
            None, None, None 如果没有活跃槽
        """
        active_mask = (self.memory_mask > 0) & (self.slot_ages >= 0)
        if not active_mask.any():
            return None, None, None
        # Flatten across batch and slot dimensions
        B, M, D = self.memory.shape
        mem_flat = self.memory.view(B * M, D)
        tid_flat = self.track_ids.view(B * M)
        age_flat = self.slot_ages.view(B * M)
        mask_flat = active_mask.view(B * M)
        idx = mask_flat.nonzero(as_tuple=True)[0]
        return mem_flat[idx], tid_flat[idx], age_flat[idx]

    def get_statistics(self):
        """记忆统计信息."""
        active = self.slot_count[self.slot_count > 0]
        if len(active) > 0:
            return {
                'active_batches': len(active),
                'avg_slots': active.float().mean().item(),
                'max_slots': active.max().item(),
            }
        return {'active_batches': 0, 'avg_slots': 0, 'max_slots': 0}
