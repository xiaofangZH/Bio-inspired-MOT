import torch
import torch.nn.functional as F
from torch import nn

from hmat.utils.box_utils import box_cxcywh_to_xyxy, complete_box_iou


class SetCriterion(nn.Module):
    """
    The process happens in two steps:
        1. We compute hungarian assignment between ground truth boxes and the outputs of the model
        2. We supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses, label_smoothing=0.1):
        """
        weight_dict: dict containing as key the names of the losses and as values their relative weight.
        losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.label_smoothing = label_smoothing
        empty_weight = torch.ones(self.num_classes)
        empty_weight[-1] = self.eos_coef  # applied to bg via target_classes >= num_classes
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (Focal Loss with empty_weight for background)."""
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        # indices may be dict (with ignore mask) or list of (src, tgt) tuples
        if isinstance(indices, dict):
            matched_to_zero = indices.get('matched_to_zero', [])
            match_indices = indices.get('match_indices', [])
        else:
            matched_to_zero = []
            match_indices = indices

        idx = self._get_src_permutation_idx(match_indices)
        target_classes_o = torch.cat([t["labels"][J]
                                     for t, (_, J) in zip(targets, match_indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        # 标记被 ignore_regions 命中的匹配预测
        ignore_weight = torch.ones(src_logits.shape[:2], device=src_logits.device)
        if matched_to_zero:
            for (b, q_idx) in matched_to_zero:
                ignore_weight[b, q_idx] = 0.0

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]
        if self.label_smoothing > 0:
            target_classes_onehot = target_classes_onehot * (1 - self.label_smoothing) + self.label_smoothing / src_logits.shape[2]
        loss_ce = sigmoid_focal_loss(
            src_logits, target_classes_onehot, num_boxes,
            alpha=0.25, gamma=2,
            class_weights=self.empty_weight,
            target_classes=target_classes,
            ignore_weight=ignore_weight,
        )
        losses = {'loss_ce': loss_ce}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute L1 + CIoU box losses."""
        assert 'pred_boxes' in outputs

        # indices may be dict (with ignore mask) or list of (src, tgt) tuples
        if isinstance(indices, dict):
            match_indices = indices.get('match_indices', [])
        else:
            match_indices = indices

        idx = self._get_src_permutation_idx(match_indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i]
                                 for t, (_, i) in zip(targets, match_indices)], dim=0)

        # NaN/Inf guard: return zero loss to allow graceful skip upstream
        if not torch.isfinite(src_boxes).all() or not torch.isfinite(target_boxes).all():
            z = torch.tensor(0.0, device=src_boxes.device, requires_grad=True)
            return {'loss_bbox': z, 'loss_ciou': z}

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        # CIoU = 1 - complete_box_iou (lower is better)
        ciou = complete_box_iou(
            box_cxcywh_to_xyxy(src_boxes),
            box_cxcywh_to_xyxy(target_boxes),
        )
        losses['loss_ciou'] = (1.0 - torch.diag(ciou)).sum() / num_boxes
        return losses

    def loss_reid(self, outputs, targets, indices, num_boxes):
        """
        Cross-frame contrastive ReID loss using InfoNCE with Memory Bank.

        Uses memory bank's stored embeddings from previous frames as positive/negative anchors
        for the current frame's matched query embeddings. This solves the "no positive pairs
        within a single frame" problem by leveraging the temporal memory.
        """
        assert 'reid_embeds' in outputs
        reid_embeds = outputs['reid_embeds']  # [B, N, reid_dim]

        # Skip if any target is missing track_ids
        if not all('track_ids' in t for t in targets):
            return {'loss_reid': torch.tensor(0.0, device=reid_embeds.device,
                                              requires_grad=True)}

        # indices may be dict (with ignore mask) or list of (src, tgt) tuples
        if isinstance(indices, dict):
            match_indices = indices.get('match_indices', [])
        else:
            match_indices = indices

        # Get matched query embeddings and their track_ids
        idx = self._get_src_permutation_idx(match_indices)
        matched_embeds = reid_embeds[idx]     # [M, reid_dim]
        target_ids = torch.cat([t['track_ids'][i]
                                for t, (_, i) in zip(targets, match_indices)], dim=0)

        M = matched_embeds.shape[0]
        if M < 1:
            return {'loss_reid': torch.tensor(0.0, device=reid_embeds.device,
                                              requires_grad=True)}

        # Check for memory bank data (cross-frame positives)
        if 'memory_embeds' not in outputs or 'memory_track_ids' not in outputs:
            # Fallback: per-frame InfoNCE (may return zero if no intra-frame positives)
            if M < 2:
                return {'loss_reid': torch.tensor(0.0, device=reid_embeds.device,
                                                  requires_grad=True)}
            sim = torch.matmul(matched_embeds, matched_embeds.T)
            temperature = 0.07
            pos_mask = (target_ids.unsqueeze(0) == target_ids.unsqueeze(1)) & \
                       (~torch.eye(M, dtype=torch.bool, device=reid_embeds.device))
            sim_scaled = sim / temperature
            sim_max = sim_scaled.max(dim=1, keepdim=True)[0].detach()
            sim_stable = sim_scaled - sim_max
            exp_sim = torch.exp(sim_stable)
            log_sum_exp = torch.log(exp_sim.sum(dim=1) + 1e-8)
            pos_exp = exp_sim * pos_mask.float()
            pos_log_sum_exp = torch.log(pos_exp.sum(dim=1) + 1e-8)
            has_pos = pos_mask.any(dim=1)
            if has_pos.sum() == 0:
                return {'loss_reid': torch.tensor(0.0, device=reid_embeds.device,
                                                  requires_grad=True)}
            nce = -(pos_log_sum_exp[has_pos] - log_sum_exp[has_pos]).mean()
            return {'loss_reid': nce}

        mem_embeds = outputs['memory_embeds']      # [K, reid_dim]
        mem_tids = outputs['memory_track_ids']     # [K]
        K = mem_embeds.shape[0]

        if K < 1:
            return {'loss_reid': torch.tensor(0.0, device=reid_embeds.device,
                                              requires_grad=True)}

        # Only use memory slots with valid track_ids (!= -1)
        valid_mask = mem_tids >= 0
        if not valid_mask.any():
            return {'loss_reid': torch.tensor(0.0, device=reid_embeds.device,
                                              requires_grad=True)}

        mem_embeds = mem_embeds[valid_mask]        # [K_v, reid_dim]
        mem_tids = mem_tids[valid_mask]            # [K_v]

        # Normalize for cosine similarity
        matched_norm = F.normalize(matched_embeds, p=2, dim=-1)
        mem_norm = F.normalize(mem_embeds, p=2, dim=-1)

        temperature = 0.07

        # Cosine similarity: [M, K_v]
        sim = torch.matmul(matched_norm, mem_norm.T) / temperature

        # Positive mask: memory slot has same track_id as query
        pos_mask = (target_ids.unsqueeze(1) == mem_tids.unsqueeze(0))  # [M, K_v]

        # Numerically stable log-sum-exp
        sim_max = sim.max(dim=1, keepdim=True)[0].detach()
        sim_stable = sim - sim_max

        exp_sim = torch.exp(sim_stable)
        log_sum_exp = torch.log(exp_sim.sum(dim=1) + 1e-8)

        pos_exp = exp_sim * pos_mask.float()
        pos_log_sum_exp = torch.log(pos_exp.sum(dim=1) + 1e-8)

        has_pos = pos_mask.any(dim=1)
        if has_pos.sum() == 0:
            return {'loss_reid': torch.tensor(0.0, device=reid_embeds.device,
                                              requires_grad=True)}

        nce = -(pos_log_sum_exp[has_pos] - log_sum_exp[has_pos]).mean()
        return {'loss_reid': nce}

    def _compute_ignore_mask(self, outputs, targets, indices, ignore_regions, iou_thresh=0.5):
        """
        对与 ignore_regions 高度重叠的预测框，将其分类 Loss 权重归零。
        
        Args:
            outputs: model outputs dict
            targets: list of target dicts
            indices: matched indices from Hungarian matcher
            ignore_regions: list of [M_i, 4] tensors per batch element (cxcywh format)
            iou_thresh: IoU threshold above which prediction is considered 'ignored'
        
        Returns:
            dict with 'matched_idx_to_zero': list of (batch_idx, query_idx) to zero out
        """
        device = next(iter(outputs.values())).device
        pred_boxes = outputs['pred_boxes']  # [B, N, 4] in cxcywh
        
        # Track which matched predictions hit ignore regions
        matched_to_zero = []  # list of (b, src_idx) tuples
        
        for b in range(len(targets)):
            if ignore_regions[b] is None or len(ignore_regions[b]) == 0:
                continue
                
            src_idx, tgt_idx = indices[b]
            if len(src_idx) == 0:
                continue
                
            # 当前 batch 中被匹配的预测框
            matched_preds = pred_boxes[b][src_idx]  # [K, 4], cxcywh
            ignore_boxes = ignore_regions[b]  # [M, 4], cxcywh
            
            if len(ignore_boxes) == 0:
                continue
            
            # 计算 IoU 矩阵
            iou_mat = self._box_iou_cxcywh(matched_preds, ignore_boxes)  # [K, M]
            
            # 标记与任一 ignore box 的 IoU > threshold 的预测
            max_iou_per_pred = iou_mat.max(dim=1)[0]  # [K]
            hit_mask = max_iou_per_pred > iou_thresh
            
            for k in range(len(src_idx)):
                if hit_mask[k]:
                    matched_to_zero.append((b, src_idx[k].item()))
        
        return {
            'matched_to_zero': matched_to_zero,
            'ignore_loss': torch.tensor(0.0, device=device, requires_grad=True),
        }
    
    @staticmethod
    def _box_iou_cxcywh(boxes_a, boxes_b):
        """Compute pairwise IoU between two sets of boxes in cxcywh format."""
        # Convert to xyxy
        def _to_xyxy(boxes):
            cx, cy, w, h = boxes.unbind(-1)
            x1 = cx - 0.5 * w
            y1 = cy - 0.5 * h
            x2 = cx + 0.5 * w
            y2 = cy + 0.5 * h
            return torch.stack([x1, y1, x2, y2], dim=-1)
        
        xyxy_a = _to_xyxy(boxes_a)
        xyxy_b = _to_xyxy(boxes_b)
        
        # IoU via intersection / union
        lt = torch.max(xyxy_a[:, None, :2], xyxy_b[None, :, :2])  # [K, M, 2]
        rb = torch.min(xyxy_a[:, None, 2:], xyxy_b[None, :, 2:])  # [K, M, 2]
        wh = (rb - lt).clamp(min=0)  # [K, M, 2]
        inter = wh[:, :, 0] * wh[:, :, 1]  # [K, M]
        
        area_a = (xyxy_a[:, 2] - xyxy_a[:, 0]) * (xyxy_a[:, 3] - xyxy_a[:, 1])  # [K]
        area_b = (xyxy_b[:, 2] - xyxy_b[:, 0]) * (xyxy_b[:, 3] - xyxy_b[:, 1])  # [M]
        union = area_a[:, None] + area_b[None, :] - inter
        
        return inter / (union + 1e-6)

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i)
                              for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
            'reid': self.loss_reid,
        }
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, ignore_regions=None):
        """
        两阶段匹配（MOTR 架构适配）:
          Stage 1: track_queries → 通过 track_id 确定性绑定到同 ID 的 GT
          Stage 2: detect_queries → 匈牙利匹配到剩余未分配的 GT
        解决 DETR 全局匹配导致的身份错乱问题。
        """
        num_boxes = sum(len(t["labels"]) for t in targets)
        device = next(iter(outputs.values())).device
        num_boxes_t = torch.as_tensor([num_boxes], dtype=torch.float, device=device)

        # ── Level 3-1: 空帧背景惩罚 ──
        if num_boxes == 0:
            src_logits = outputs['pred_logits']
            target_classes_onehot = torch.zeros_like(src_logits)
            loss_ce = sigmoid_focal_loss(
                src_logits, target_classes_onehot, num_boxes=1,
                alpha=0.25, gamma=2,
            )
            z = torch.tensor(0.0, device=device, requires_grad=True)
            empty_indices = [(torch.empty((0,), dtype=torch.int64, device=device),
                              torch.empty((0,), dtype=torch.int64, device=device))
                             for _ in range(len(targets))]
            return {"loss_ce": loss_ce * self.eos_coef, "loss_bbox": z,
                    "loss_ciou": z, "loss_reid": z}, empty_indices

        # ── Level 2: 两阶段匹配 ──
        num_tracks = outputs.get('num_tracks', 0)
        track_query_ids = outputs.get('track_query_ids', None)

        if num_tracks > 0 and track_query_ids is not None and len(track_query_ids) > 0:
            indices = self._two_stage_matching(
                outputs, targets, num_tracks, track_query_ids, device)
        else:
            indices = self.matcher(outputs, targets)

        # ── Ignore regions: 零化与 ignore box 高度重合的预测的 class loss ──
        losses = {}
        if ignore_regions is not None and len(ignore_regions) > 0:
            ignore_result = self._compute_ignore_mask(
                outputs, targets, indices, ignore_regions)
            losses['ignore_penalty'] = ignore_result.get('ignore_loss',
                torch.tensor(0.0, device=device, requires_grad=True))
            if ignore_result['matched_to_zero']:
                indices = {
                    'match_indices': indices,
                    'matched_to_zero': ignore_result['matched_to_zero'],
                }

        for loss in self.losses:
            losses.update(self.get_loss(
                loss, outputs, targets, indices, num_boxes_t))

        return losses, indices

    def _two_stage_matching(self, outputs, targets, num_tracks, track_query_ids, device):
        """
        Stage 1: 确定性 track_id 匹配 (track_queries → GT)
        Stage 2: 匈牙利匹配 (detect_queries → 剩余 GT)

        Returns indices 列表，格式与 HungarianMatcher.forward 一致。
        """
        bs = len(targets)
        track_qids = track_query_ids  # [num_tracks]

        # ── Stage 1: 确定性 track_id 匹配 ──
        stage1_indices = []
        matched_gt_sets = []  # per-batch set of matched GT positions

        for b in range(bs):
            gt_tids = targets[b].get('track_ids',
                torch.empty(0, dtype=torch.long, device=device))
            batch_src = []
            batch_tgt = []
            matched_gt_set = set()

            for tq_idx in range(num_tracks):
                tid = track_qids[tq_idx].item()
                if tid < 0:
                    continue  # skip invalid/background track_id
                # 在当前帧 GT 中寻找同 track_id 的目标
                matches = (gt_tids == tid).nonzero(as_tuple=True)[0]
                if len(matches) > 0:
                    batch_src.append(tq_idx)
                    batch_tgt.append(matches[0].item())
                    matched_gt_set.add(matches[0].item())

            stage1_indices.append((
                torch.tensor(batch_src, dtype=torch.int64, device=device),
                torch.tensor(batch_tgt, dtype=torch.int64, device=device),
            ))
            matched_gt_sets.append(matched_gt_set)

        # ── Stage 2: 匈牙利匹配 detect_queries vs 未匹配 GT ──
        # 构建 reduced targets（剔除已匹配的 GT），并记录索引映射
        reduced_targets = []
        reduced_to_original = []  # tensor: reduced tgt index → original tgt index

        for b in range(bs):
            num_gt = len(targets[b]['boxes'])
            keep_mask = torch.ones(num_gt, dtype=torch.bool, device=device)
            for gt_idx in matched_gt_sets[b]:
                if gt_idx < num_gt:
                    keep_mask[gt_idx] = False

            unmatched_indices = torch.where(keep_mask)[0]
            reduced_to_original.append(unmatched_indices)

            reduced_targets.append({
                'labels': targets[b]['labels'][keep_mask],
                'boxes': targets[b]['boxes'][keep_mask],
                'track_ids': targets[b].get('track_ids',
                    torch.empty((0,), dtype=torch.long, device=device))[keep_mask],
            })

        # 仅在有未匹配 GT 时运行匈牙利匹配
        total_unmatched = sum(len(t['labels']) for t in reduced_targets)
        if total_unmatched > 0:
            detect_outputs = {
                'pred_logits': outputs['pred_logits'][:, num_tracks:, :],
                'pred_boxes': outputs['pred_boxes'][:, num_tracks:, :],
            }
            stage2_indices = self.matcher(detect_outputs, reduced_targets)
            # 映射回原始 target 索引 + 偏移 query 索引
            stage2_mapped = []
            for b in range(bs):
                src, tgt = stage2_indices[b]
                # matcher 返回 CPU 索引，需要移到对应 device 再参与 cat
                src = src.to(device)
                if len(tgt) > 0:
                    tgt_original = reduced_to_original[b][tgt.long()]
                else:
                    tgt_original = tgt.new_empty((0,))
                stage2_mapped.append((src + num_tracks, tgt_original))
            stage2_indices = stage2_mapped
        else:
            stage2_indices = [
                (torch.empty((0,), dtype=torch.int64, device=device),
                 torch.empty((0,), dtype=torch.int64, device=device))
                for _ in range(bs)
            ]

        # ── 合并 Stage 1 和 Stage 2 的匹配结果 ──
        indices = []
        for b in range(bs):
            s1_src, s1_tgt = stage1_indices[b]
            s2_src, s2_tgt = stage2_indices[b]
            if len(s1_src) > 0 and len(s2_src) > 0:
                merged_src = torch.cat([s1_src, s2_src])
                merged_tgt = torch.cat([s1_tgt, s2_tgt])
            elif len(s1_src) > 0:
                merged_src, merged_tgt = s1_src, s1_tgt
            else:
                merged_src, merged_tgt = s2_src, s2_tgt
            indices.append((merged_src, merged_tgt))

        return indices


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2, class_weights=None, target_classes=None, ignore_weight=None):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        class_weights: per-class weight tensor [num_classes+1], where last element = eos_coef for bg.
        target_classes: [B, N] with class indices (0=fg, num_classes=bg).
        ignore_weight: [B, N] per-sample weight multiplier (0.0 = ignore this sample entirely).
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(
        inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    # Apply per-sample class weights (e.g., eos_coef for background)
    if class_weights is not None and target_classes is not None:
        bg_weight = class_weights[-1]  # eos_coef for background class
        sample_weight = torch.where(
            target_classes.unsqueeze(-1) >= targets.shape[-1],
            torch.tensor(bg_weight, device=loss.device, dtype=loss.dtype),
            torch.tensor(1.0, device=loss.device, dtype=loss.dtype),
        )
        loss = loss * sample_weight

    # Apply per-sample ignore weight (zero out loss for ignored predictions)
    if ignore_weight is not None:
        loss = loss * ignore_weight.unsqueeze(-1)

    return loss.sum() / num_boxes
