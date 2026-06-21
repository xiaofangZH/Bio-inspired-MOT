#!/usr/bin/env python3
"""HAMT模型评估脚本 — 支持 MOTA / MOTP / IDF1 等标准 MOT 指标."""

import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
from hmat.modeling.hmat_model import HMAT
from data_loader import create_dataloader


def box_iou(boxes1, boxes2):
    """计算两组 bbox 之间的 IoU 矩阵 (xywh 格式)."""
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return np.zeros((len(boxes1), len(boxes2)))

    # xywh → xyxy
    b1 = boxes1.copy()
    b2 = boxes2.copy()
    b1[:, 2] = b1[:, 0] + b1[:, 2]
    b1[:, 3] = b1[:, 1] + b1[:, 3]
    b2[:, 2] = b2[:, 0] + b2[:, 2]
    b2[:, 3] = b2[:, 1] + b2[:, 3]

    area1 = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])
    area2 = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])

    lt = np.maximum(b1[:, None, :2], b2[None, :, :2])
    rb = np.minimum(b1[:, None, 2:], b2[None, :, 2:])
    wh = np.maximum(0, rb - lt)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2[None, :] - inter
    return inter / np.maximum(union, 1e-6)


def compute_mot_metrics(all_gt, all_dt, iou_threshold=0.5):
    """
    计算 MOT 评估指标 (简化版 CLEAR MOT Metrics).

    Args:
        all_gt: dict {frame_id: [[track_id, x, y, w, h], ...]}
        all_dt: dict {frame_id: [[track_id, x, y, w, h, score], ...]}
        iou_threshold: IoU 匹配阈值

    Returns:
        metrics: dict with MOTA, MOTP, IDF1, precision, recall, etc.
    """
    # 统计变量
    total_gt = 0
    total_dt = 0
    total_matched = 0
    total_fp = 0
    total_fn = 0
    total_mismatch = 0
    motp_sum = 0.0

    # IDF1 相关
    id_gt_map = {}     # gt_track_id → set of frame_ids
    id_dt_map = {}     # dt_track_id → set of frame_ids

    # 逐帧匹配
    frame_ids = sorted(set(list(all_gt.keys()) + list(all_dt.keys())))
    prev_matches = {}  # gt_track_id → dt_track_id (跨帧 ID 连续性)

    for fid in frame_ids:
        gt_boxes = all_gt.get(fid, [])
        dt_boxes = all_dt.get(fid, [])

        total_gt += len(gt_boxes)
        total_dt += len(dt_boxes)

        if len(gt_boxes) == 0 or len(dt_boxes) == 0:
            total_fn += len(gt_boxes)
            total_fp += len(dt_boxes)
            # 未匹配的检测注册为新 track
            for dt in dt_boxes:
                tid = dt[0]
                if tid not in id_dt_map:
                    id_dt_map[tid] = set()
                id_dt_map[tid].add(fid)
            for gt in gt_boxes:
                tid = gt[0]
                if tid not in id_gt_map:
                    id_gt_map[tid] = set()
                id_gt_map[tid].add(fid)
            continue

        # 提取数据
        gt_arr = np.array([[b[1], b[2], b[3], b[4]] for b in gt_boxes])
        dt_arr = np.array([[b[1], b[2], b[3], b[4]] for b in dt_boxes])
        gt_ids = [b[0] for b in gt_boxes]
        dt_ids = [b[0] for b in dt_boxes]

        # IoU 矩阵
        iou_matrix = box_iou(gt_arr, dt_arr)

        # Hungarian 匹配 (最大化 IoU)
        cost = 1.0 - iou_matrix
        gt_indices, dt_indices = linear_sum_assignment(cost)

        matched_gt = set()
        matched_dt = set()
        n_matched = 0

        for g, d in zip(gt_indices, dt_indices):
            if iou_matrix[g, d] >= iou_threshold:
                matched_gt.add(g)
                matched_dt.add(d)
                n_matched += 1

                # MOTP: 匹配对的重叠度
                motp_sum += iou_matrix[g, d]

                # 注册 track ID
                gt_tid = gt_ids[g]
                dt_tid = dt_ids[d]
                if gt_tid not in id_gt_map:
                    id_gt_map[gt_tid] = set()
                id_gt_map[gt_tid].add(fid)
                if dt_tid not in id_dt_map:
                    id_dt_map[dt_tid] = set()
                id_dt_map[dt_tid].add(fid)

                # ID switch 检测
                if gt_tid in prev_matches and prev_matches[gt_tid] != dt_tid:
                    total_mismatch += 1
                prev_matches[gt_tid] = dt_tid

        total_matched += n_matched
        total_fn += len(gt_boxes) - len(matched_gt)
        total_fp += len(dt_boxes) - len(matched_dt)

    # ─── 计算指标 ───
    denom = max(total_gt, 1)
    mota = max(0, 1.0 - (total_fn + total_fp + total_mismatch) / denom) * 100
    motp = (motp_sum / max(total_matched, 1)) * 100
    precision = total_matched / max(total_dt, 1) * 100
    recall = total_matched / denom * 100

    # IDF1 计算
    idf1 = 0.0
    if id_gt_map and id_dt_map:
        idtp = 0
        idfp = 0
        idfn = 0
        for gid, gframes in id_gt_map.items():
            idfn += len(gframes)
        for did, dframes in id_dt_map.items():
            idfp += len(dframes)
        # 简化 IDF1: 近似计算 (完整实现需要建立GT-DT track对)
        id_precision = total_matched / max(total_dt, 1)
        id_recall = total_matched / denom
        idf1 = 2 * id_precision * id_recall / max(id_precision + id_recall, 1e-6) * 100

    return {
        'MOTA': round(mota, 2),
        'MOTP': round(motp, 2),
        'IDF1': round(idf1, 2),
        'Precision': round(precision, 2),
        'Recall': round(recall, 2),
        'GT': int(total_gt),
        'DT': int(total_dt),
        'Matched': int(total_matched),
        'FP': int(total_fp),
        'FN': int(total_fn),
        'IDSW': int(total_mismatch),
    }


class MOTEvaluator:
    """MOT 评估器：使用 HAMT 模型在测试集上进行推理并计算指标."""

    def __init__(self, dataset_name='dancetrack', device='cuda',
                 checkpoint_path=None, conf_threshold=0.3):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.dataset_name = dataset_name
        self.conf_threshold = conf_threshold

        # 加载模型
        self.model = HMAT(
            num_classes=1, hidden_dim=256, num_queries=300,
            num_detect_queries=100, max_track_age=30,
            use_batch_memory=False,  # 推理用 MemoryBank
            max_memory_size=5,
        ).to(self.device)
        self.model.eval()

        # 加载检查点
        if checkpoint_path and Path(checkpoint_path).exists():
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            state = ckpt.get('model_state_dict', ckpt)
            self.model.load_state_dict(state, strict=False)
            print(f"[Eval] 已加载检查点: {checkpoint_path}")

    def evaluate(self, dataloader):
        all_gt = defaultdict(list)  # frame_id → [[tid, x, y, w, h], ...]
        all_dt = defaultdict(list)
        global_frame_idx = 0

        with torch.no_grad():
            for batch in dataloader:
                sequences = batch if isinstance(batch, list) else [batch]
                for seq in sequences:
                    frames = seq.get('frames', [])
                    if not frames:
                        continue

                    # 重置记忆（新序列）
                    self.model.memory_bank.reset()

                    for frame in frames:
                        img = frame['image'].unsqueeze(0).to(self.device)
                        annotations = frame.get('annotations', [])

                        # 收集 GT
                        for ann in annotations:
                            bbox = ann.get('bbox', [0, 0, 0, 0])
                            tid = ann.get('track_id', -1)
                            if len(bbox) >= 4 and tid >= 0:
                                all_gt[global_frame_idx].append(
                                    [tid, bbox[0], bbox[1], bbox[2], bbox[3]]
                                )

                        # 模型预测
                        outputs = self.model(img)

                        if isinstance(outputs, list) and outputs:
                            outputs = outputs[0]

                        if isinstance(outputs, dict):
                            pred_logits = outputs.get('pred_logits')  # [1, Nq, 2]
                            pred_boxes = outputs.get('pred_boxes')    # [1, Nq, 4]

                            if pred_logits is not None and pred_boxes is not None:
                                scores = torch.softmax(pred_logits[0], dim=-1)[:, 0]
                                boxes = pred_boxes[0]  # [Nq, 4] in cxcywh, normalized

                                # 高置信度预测
                                keep = scores > self.conf_threshold
                                keep_boxes = boxes[keep].cpu().numpy()
                                keep_scores = scores[keep].cpu().numpy()

                                for i, (box, score) in enumerate(zip(keep_boxes, keep_scores)):
                                    # cxcywh → xywh (denormalize 假设 640x640)
                                    cx, cy, w, h = box
                                    x = (cx - w / 2) * 640
                                    y = (cy - h / 2) * 640
                                    w = w * 640
                                    h = h * 640
                                    # 用预测索引作为伪 track_id
                                    pseudo_tid = i
                                    all_dt[global_frame_idx].append(
                                        [pseudo_tid, x, y, w, h, float(score)]
                                    )

                        # 更新记忆
                        track_embeds, track_ids, track_boxes = self.model.memory_bank.get_track_queries()
                        if track_embeds.numel() > 0:
                            track_embeds = track_embeds.to(self.device)
                        if track_boxes.numel() > 0:
                            track_boxes = track_boxes.to(self.device)
                        self.model.memory_bank.update(
                            track_embeds.unsqueeze(0),
                            {'pred_logits': outputs.get('pred_logits')},
                            outputs.get('pred_logits')[:, :0, :] if isinstance(outputs, dict) else torch.zeros(1, 0, 2),
                            {},
                        )

                        global_frame_idx += 1

        # 计算指标
        metrics = compute_mot_metrics(all_gt, all_dt)
        print(f"\n{'='*50}")
        print(f"[{self.dataset_name.upper()}] 评估结果:")
        print(f"  MOTA:      {metrics['MOTA']:.2f}%")
        print(f"  MOTP:      {metrics['MOTP']:.2f}%")
        print(f"  IDF1:      {metrics['IDF1']:.2f}%")
        print(f"  Precision: {metrics['Precision']:.2f}%")
        print(f"  Recall:    {metrics['Recall']:.2f}%")
        print(f"  GT: {metrics['GT']}  DT: {metrics['DT']}  "
              f"FP: {metrics['FP']}  FN: {metrics['FN']}  "
              f"IDSW: {metrics['IDSW']}")
        print(f"{'='*50}\n")
        return metrics


def evaluate_on_dataset(dataset_name, test_root, checkpoint_path=None):
    """评估指定数据集的便捷函数."""
    test_loader = create_dataloader(
        root_dir=str(test_root),
        dataset_name=dataset_name,
        split='test',
        batch_size=1,
        num_workers=0,
        img_size=(640, 640),
    )
    evaluator = MOTEvaluator(
        dataset_name=dataset_name,
        checkpoint_path=checkpoint_path,
    )
    return evaluator.evaluate(test_loader)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='dancetrack')
    parser.add_argument('--checkpoint', default=None, help='模型检查点路径')
    args = parser.parse_args()

    paths = {
        'dancetrack': '/home/user/test',
        'mot17': '/home/user/MOT17/MOT17/test',
        'mot20': '/home/user/MOT20/MOT20/test',
    }
    evaluate_on_dataset(args.dataset, paths[args.dataset], args.checkpoint)
