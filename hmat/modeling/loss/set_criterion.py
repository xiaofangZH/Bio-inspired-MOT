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
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (Focal Loss)"""
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J]
                                     for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]
        if self.label_smoothing > 0:
            target_classes_onehot = target_classes_onehot * (1 - self.label_smoothing) + self.label_smoothing / src_logits.shape[2]
        loss_ce = sigmoid_focal_loss(
            src_logits, target_classes_onehot, num_boxes, alpha=0.25, gamma=2)
        losses = {'loss_ce': loss_ce}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute L1 + CIoU box losses."""
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i]
                                 for t, (_, i) in zip(targets, indices)], dim=0)

        # NaN/Inf guard
        if not torch.isfinite(src_boxes).all() or not torch.isfinite(target_boxes).all():
            nan_val = torch.tensor(float('nan'), device=src_boxes.device)
            return {'loss_bbox': nan_val, 'loss_ciou': nan_val}

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
        Contrastive ReID loss using InfoNCE.

        Gracefully returns zero when targets lack 'track_ids' (e.g. MOT17).
        """
        assert 'reid_embeds' in outputs
        reid_embeds = outputs['reid_embeds']  # [B, N, reid_dim]

        # Skip if any target is missing track_ids
        if not all('track_ids' in t for t in targets):
            return {'loss_reid': torch.tensor(0.0, device=reid_embeds.device,
                                              requires_grad=True)}

        idx = self._get_src_permutation_idx(indices)
        matched_embeds = reid_embeds[idx]     # [M, reid_dim]
        target_ids = torch.cat([t['track_ids'][i]
                                for t, (_, i) in zip(targets, indices)], dim=0)

        M = matched_embeds.shape[0]
        if M < 2:
            return {'loss_reid': torch.tensor(0.0, device=reid_embeds.device,
                                              requires_grad=True)}

        sim = torch.matmul(matched_embeds, matched_embeds.T)  # cosine sim
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

    def forward(self, outputs, targets):
        """This performs the loss computation."""
        num_boxes = sum(len(t["labels"]) for t in targets)
        device = next(iter(outputs.values())).device
        num_boxes_t = torch.as_tensor([num_boxes], dtype=torch.float, device=device)

        if num_boxes == 0:
            z = torch.tensor(0.0, device=device, requires_grad=True)
            return {"loss_ce": z, "loss_bbox": z, "loss_ciou": z}

        indices = self.matcher(outputs, targets)

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(
                loss, outputs, targets, indices, num_boxes_t))

        return losses


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(
        inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes
