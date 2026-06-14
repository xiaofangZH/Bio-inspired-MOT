import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from hmat.utils.box_utils import box_cxcywh_to_xyxy, complete_box_iou


class HungarianMatcher(nn.Module):
    """
    Hungarian Matcher optimized for DanceTrack (MOT).

    Uses CIoU (Complete IoU) in cost matrix for better bbox overlap
    estimation, accounting for center distance and aspect ratio.
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 5, cost_ciou: float = 2):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_ciou = cost_ciou
        assert cost_class != 0 or cost_bbox != 0 or cost_ciou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets, group='all'):
        """
        Args:
            outputs: dict with "pred_logits" [B, Nq, C] and "pred_boxes" [B, Nq, 4]
            targets: list of dicts with "labels" and "boxes"
            group: 'all', 'tracks', or 'detects' - to filter queries
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [B*Nq, 4]

        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # ── Classification Cost (Focal style) ──
        alpha = 0.25
        gamma = 2.0
        neg_cost_class = (1 - alpha) * (out_prob ** gamma) * \
            (-(1 - out_prob + 1e-8).log())
        pos_cost_class = alpha * \
            ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

        # ── L1 Cost ──
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # ── CIoU Cost ──
        cost_ciou = 1.0 - complete_box_iou(
            box_cxcywh_to_xyxy(out_bbox),
            box_cxcywh_to_xyxy(tgt_bbox),
        )

        # NaN guard
        if not torch.isfinite(cost_ciou).all():
            cost_ciou = torch.where(
                torch.isfinite(cost_ciou), cost_ciou,
                torch.tensor(1.0, device=cost_ciou.device),
            )

        # ── Final Cost Matrix ──
        C = (self.cost_bbox * cost_bbox +
             self.cost_class * cost_class +
             self.cost_ciou * cost_ciou)

        if not torch.isfinite(C).all():
            C = torch.where(torch.isfinite(C), C,
                            torch.tensor(1e8, device=C.device))
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i])
                   for i, c in enumerate(C.split(sizes, -1))]

        return [(torch.as_tensor(i, dtype=torch.int64),
                 torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
