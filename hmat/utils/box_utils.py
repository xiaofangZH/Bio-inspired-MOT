"""统一的 BBox 工具函数，消除各模块间的重复代码."""

import torch
from torchvision.ops.boxes import box_area


def box_cxcywh_to_xyxy(x):
    """Convert (cx, cy, w, h) to (x1, y1, x2, y2)."""
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    """Convert (x1, y1, x2, y2) to (cx, cy, w, h)."""
    x1, y1, x2, y2 = x.unbind(-1)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = x2 - x1
    h = y2 - y1
    return torch.stack([cx, cy, w, h], dim=-1)


def box_iou(boxes1, boxes2):
    """
    Compute IoU between two sets of boxes in (x1, y1, x2, y2).

    Args:
        boxes1: (N, 4)
        boxes2: (M, 4)

    Returns:
        iou:   (N, M) pairwise IoU
        union: (N, M) pairwise union areas
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]

    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    union = area1[:, None] + area2 - inter
    eps_iou = 1e-6
    iou = inter / union.clamp(min=eps_iou)
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU.
    Input boxes in (x1, y1, x2, y2).
    """
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area.clamp(min=1e-6)


def complete_box_iou(boxes1, boxes2):
    """
    CIoU (Complete IoU) — 同时考虑重叠面积、中心点距离和长宽比。

    CIoU = IoU - ρ²/ c² - α· v

    其中:
        ρ² = 中心点欧式距离平方
        c²  = 最小包围框对角线长度平方
        v   = (4/π²)· (arctan(wgt/hgt) - arctan(w/h))²
        α   = v / (1 - IoU + v)
    """
    eps = 1e-7
    w1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=eps)
    h1 = (boxes1[:, 3] - boxes1[:, 1]).clamp(min=eps)
    w2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=eps)
    h2 = (boxes2[:, 3] - boxes2[:, 1]).clamp(min=eps)

    area1 = w1 * h1
    area2 = w2 * h2

    inter_l = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
    inter_t = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    inter_r = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
    inter_b = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])
    inter_w = (inter_r - inter_l).clamp(min=0)
    inter_h = (inter_b - inter_t).clamp(min=0)
    inter = inter_w * inter_h
    union = area1[:, None] + area2[None, :] - inter
    iou = inter / union.clamp(min=eps)

    # 中心距离
    c1x = (boxes1[:, 0] + boxes1[:, 2]) / 2
    c1y = (boxes1[:, 1] + boxes1[:, 3]) / 2
    c2x = (boxes2[:, 0] + boxes2[:, 2]) / 2
    c2y = (boxes2[:, 1] + boxes2[:, 3]) / 2
    rho2 = (c1x[:, None] - c2x[None, :]) ** 2 + (c1y[:, None] - c2y[None, :]) ** 2

    enclose_l = torch.min(boxes1[:, None, 0], boxes2[None, :, 0])
    enclose_t = torch.min(boxes1[:, None, 1], boxes2[None, :, 1])
    enclose_r = torch.max(boxes1[:, None, 2], boxes2[None, :, 2])
    enclose_b = torch.max(boxes1[:, None, 3], boxes2[None, :, 3])
    c2 = (enclose_r - enclose_l).clamp(min=eps) ** 2 + (enclose_b - enclose_t).clamp(min=eps) ** 2

    atan1 = torch.atan(w1 / h1)
    atan2 = torch.atan(w2 / h2)
    import math
    v = (4.0 / (math.pi ** 2)) * (atan2[None, :] - atan1[:, None]) ** 2

    with torch.no_grad():
        alpha = v / ((1.0 - iou) + v + eps)

    return iou - rho2 / c2.clamp(min=eps) - alpha * v
