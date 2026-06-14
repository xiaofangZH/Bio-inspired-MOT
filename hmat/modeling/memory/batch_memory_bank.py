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

    def __init__(self, hidden_dim, max_memory_size=5, max_batch_size=32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_memory_size = max_memory_size
        self.max_batch_size = max_batch_size

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

        # Decay coefficient
        self.memory_decay_coeff = 1.0

        # GRU for per-slot smooth update
        self.gru_update = nn.GRUCell(hidden_dim, hidden_dim)

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

    def update(self, track_embeds, track_preds, detect_embeds, detect_preds, targets=None):
        """
        更新记忆 —— 逐目标独立存储。

        Args:
            track_embeds:  [B, N_track, D]   已有 track 的 decoder 输出
            track_preds:  dict with 'pred_boxes' etc.
            detect_embeds: [B, N_detect, D]  新检测 query 的输出
            detect_preds:  dict with 'pred_boxes' etc.
            targets:       Optional ground truth
        """
        B = track_embeds.size(0)
        assert B <= self.max_batch_size, f"batch_size {B} > max_batch_size {self.max_batch_size}"

        # ── 更新已有 track 的记忆 ──
        if track_embeds.size(1) > 0:
            self._update_existing_slots(track_embeds, track_preds)

        # ── 为新检测创建新记忆槽 ──
        if detect_embeds.size(1) > 0:
            self._create_new_slots(detect_embeds, detect_preds)

        # ── 所有激活槽 age+1 ──
        self.slot_ages[:B] = self.slot_ages[:B] + 1.0

        # ── 超过 max_memory_size 的旧槽移除 ──
        self._prune_oldest(B)

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
            old_mem = self.memory[b, :n_update]          # [n_update, D]
            new_feat = track_embeds[b, :n_update]         # [n_update, D]

            # GRU 平滑更新
            updated = self.gru_update(
                new_feat.reshape(-1, D),
                old_mem.reshape(-1, D)
            ).reshape(n_update, D)

            # NaN guard: 跳过异常更新，保留旧记忆
            if not torch.isfinite(updated).all():
                continue

            # 写入并 detach
            self.memory.data[b, :n_update] = updated.detach()
            self.slot_ages[b, :n_update] = 0.0  # 重置年龄

            # 存储真实 bbox 坐标
            if 'pred_boxes' in track_preds and track_preds['pred_boxes'].shape[1] >= n_update:
                boxes = track_preds['pred_boxes'][b, :n_update].detach()
                # NaN guard
                boxes = torch.nan_to_num(boxes, nan=0.5, posinf=1.0, neginf=0.0)
                boxes = boxes.clamp(0.0, 1.0)
                self.track_boxes[b, :n_update] = boxes

    def _create_new_slots(self, detect_embeds, detect_preds):
        """
        将高置信度检测创建为新记忆槽。

        detect_embeds: [B, N_detect, D]
        """
        B, N_det, D = detect_embeds.shape

        # 获取置信度
        if 'pred_logits' in detect_preds:
            logits = detect_preds['pred_logits']  # [B, N_det, num_classes+1]
            scores = torch.softmax(logits, dim=-1)[:, :, 0]  # 前景概率
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

                # 存储新检测的 bbox
                if 'pred_boxes' in detect_preds and detect_preds['pred_boxes'].shape[1] > det_idx:
                    box = detect_preds['pred_boxes'][b, det_idx].detach()
                    box = torch.nan_to_num(box, nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
                    self.track_boxes[b, slot_idx] = box

            self.slot_count[b] += n_new

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

            self.memory.data[b, :self.max_memory_size] = old_mem[keep_order]
            self.memory_mask.data[b, :self.max_memory_size] = old_mask[keep_order]
            self.slot_ages[b, :self.max_memory_size] = old_ages[keep_order]
            self.memory.data[b, self.max_memory_size:] = 0.0
            self.memory_mask.data[b, self.max_memory_size:] = 0.0
            self.slot_ages[b, self.max_memory_size:] = 0.0
            self.slot_count[b] = self.max_memory_size

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
        """重置所有记忆."""
        device = self.memory.device
        dtype = self.memory.dtype
        self.memory.data = torch.zeros_like(self.memory.data).to(device=device, dtype=dtype)
        self.memory_mask.data = torch.zeros_like(self.memory_mask.data).to(device=device)
        self.slot_count.zero_()
        self.slot_ages.zero_()
        self.track_boxes.zero_()

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
