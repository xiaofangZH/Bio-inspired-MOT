#!/usr/bin/env python3
"""统一目标解析器 — 根据数据集类型过滤/分类目标."""

import torch


class TargetParser:
    """统一目标解析：无论如何输入，统一输出 valid_targets + ignore_regions."""

    # MOT 数据集中的 "类人非人" 类别（长得像人但不应作为追踪目标）
    MOT_IGNORE_CLASSES = {2, 7, 8, 12}  # Person on vehicle, Static person, Distractor, Reflection
    MOT_VISIBILITY_THRESHOLD = 0.1

    @staticmethod
    def parse_crowdhuman(annotations):
        """
        CrowdHuman 标注：提取 fbox 作为 GT，hbox 丢弃。
        extra.ignore == 1 的目标存入 ignore_regions。

        Args:
            annotations: list of gtbox dicts (from ODGT), each has
                {'fbox': [x,y,w,h], 'hbox': [...], 'extra': {'ignore': 0/1, 'box_id': N}}

        Returns:
            valid_targets: list of {'box': [x,y,w,h], 'track_id': int, 'class_id': 0}
            ignore_regions: list of {'box': [x,y,w,h]} — fbox for ignored persons
        """
        valid_targets = []
        ignore_regions = []

        for i, ann in enumerate(annotations):
            fbox = ann.get('fbox', None)
            if fbox is None:
                continue
            extra = ann.get('extra', {})
            is_ignore = extra.get('ignore', 0)

            if is_ignore == 1:
                ignore_regions.append({'box': fbox, 'type': 'crowd_ignore'})
            else:
                # 使用 box_id 或 遍历索引作为 track_id
                track_id = extra.get('box_id', i)
                valid_targets.append({
                    'box': fbox,           # [x, y, w, h] 绝对坐标
                    'track_id': int(track_id),
                    'class_id': 0,          # 单一类别：人
                })

        return valid_targets, ignore_regions

    @staticmethod
    def parse_mot(annotations, orig_img_size=(1920, 1080)):
        """
        MOT17/MOT20 标注处理：
        - class_id == 1 行人 → valid
        - class_id in {2,7,8,12} → ignore
        - visibility < 0.1 → ignore

        Args:
            annotations: list of annotation dicts from MOTDataset._load_gt()
            orig_img_size: (W, H) original image size for bbox normalization

        Returns:
            valid_targets: list of annotation dicts (with bbox_norm preserved)
            ignore_regions: list of {'box': [x,y,w,h], 'type': str}
        """
        valid_targets = []
        ignore_regions = []

        for ann in annotations:
            class_id = ann.get('class', 1)
            visibility = ann.get('visibility', 1.0)

            is_ignore_class = class_id in TargetParser.MOT_IGNORE_CLASSES
            is_low_vis = visibility < TargetParser.MOT_VISIBILITY_THRESHOLD

            if class_id == 1 and visibility >= TargetParser.MOT_VISIBILITY_THRESHOLD:
                # 有效行人
                valid_targets.append(ann)
            elif is_ignore_class or is_low_vis:
                # 忽略区域 — 使用 bbox_norm 或 bbox
                box = ann.get('bbox', [0, 0, 0, 0])  # [x,y,w,h]
                ignore_type = 'low_visibility' if is_low_vis else f'mot_ignore_cls_{class_id}'
                ignore_regions.append({'box': box, 'type': ignore_type})

        return valid_targets, ignore_regions

    @staticmethod
    def parse_dancetrack(annotations):
        """DanceTrack 纯净标注，无需过滤."""
        return annotations, []

    @staticmethod
    def to_tensors(valid_targets, ignore_regions, orig_img_size, device='cpu'):
        """
        将解析后的 targets/regions 转为训练用张量格式。

        Args:
            valid_targets: TargetParser 解析结果
            ignore_regions: ignore boxes (绝对坐标)
            orig_img_size: (W, H)
            device: torch device

        Returns:
            targets dict: {'boxes': (N,4), 'labels': (N,), 'track_ids': (N,)}
            ignore_tensor: (M,4) in cxcywh normalized, or None
        """
        W, H = orig_img_size

        boxes = []
        labels = []
        track_ids = []

        for t in valid_targets:
            x, y, w, h = t['box']
            # 转为 (cx, cy, w, h) 归一化
            cx = (x + 0.5 * w) / max(W, 1)
            cy = (y + 0.5 * h) / max(H, 1)
            nw = w / max(W, 1)
            nh = h / max(H, 1)
            boxes.append([cx, cy, nw, nh])
            labels.append(t.get('class_id', 0))
            track_ids.append(t.get('track_id', -1))

        if boxes:
            boxes_t = torch.tensor(boxes, dtype=torch.float32, device=device)
            labels_t = torch.tensor(labels, dtype=torch.long, device=device)
            track_ids_t = torch.tensor(track_ids, dtype=torch.long, device=device)
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32, device=device)
            labels_t = torch.zeros((0,), dtype=torch.long, device=device)
            track_ids_t = torch.zeros((0,), dtype=torch.long, device=device)

        ignore_tensor = None
        if ignore_regions:
            ignore_boxes = []
            for r in ignore_regions:
                x, y, w, h = r['box']
                cx = (x + 0.5 * w) / max(W, 1)
                cy = (y + 0.5 * h) / max(H, 1)
                nw = w / max(W, 1)
                nh = h / max(H, 1)
                ignore_boxes.append([cx, cy, nw, nh])
            ignore_tensor = torch.tensor(ignore_boxes, dtype=torch.float32, device=device)

        target_dict = {
            'boxes': boxes_t,
            'labels': labels_t,
            'track_ids': track_ids_t,
        }
        return target_dict, ignore_tensor
