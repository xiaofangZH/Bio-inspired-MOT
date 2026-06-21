#!/usr/bin/env python3
"""
HAMT 模型评估脚本 — 修复版 v2
策略: 使用全部检测 query 预测 (忽略记忆库追踪)，
      在线 SORT 式 IoU 关联做跨帧身份绑定。

用法:
  python evaluate_checkpoint.py --checkpoint results/hamt_deep_20260620_101740/phase3/phase3_best.pth --dataset dancetrack
"""
import argparse
import sys
import os
import json
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
from hmat.modeling.hmat_model import HMAT

# ─── 数据集路径 ───
DATA_ROOTS = {
    'dancetrack': {
        'val': PROJECT_ROOT / 'hmat/data/OpenDataLab___DanceTrack/val',
        'test': PROJECT_ROOT / 'hmat/data/OpenDataLab___DanceTrack/test',
    },
    'mot17': {
        'val': PROJECT_ROOT / 'hmat/data/OpenDataLab___MOT17/train',
        'test': PROJECT_ROOT / 'hmat/data/OpenDataLab___MOT17/test',
    },
    'mot20': {
        'val': PROJECT_ROOT / 'hmat/data/OpenDataLab___MOT20/train',
        'test': PROJECT_ROOT / 'hmat/data/OpenDataLab___MOT20/test',
    },
}
MOT17_TRAIN_SEQ = {'MOT17-02', 'MOT17-04', 'MOT17-05', 'MOT17-09',
                   'MOT17-10', 'MOT17-11', 'MOT17-13'}


# ═══════════════════════════════════════════════════════════════
# 图像加载
# ═══════════════════════════════════════════════════════════════
def load_image(img_path, img_size=640):
    from PIL import Image
    img = Image.open(img_path).convert('RGB')
    img = img.resize((img_size, img_size), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=tensor.dtype).view(3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor.unsqueeze(0)


# ═══════════════════════════════════════════════════════════════
# GT 读取
# ═══════════════════════════════════════════════════════════════
def read_gt_boxes(gt_path):
    gt_frames = defaultdict(list)
    if not os.path.exists(gt_path):
        return gt_frames
    with open(gt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) < 8:
                continue
            frame_id = int(parts[0])
            tid = int(parts[1])
            x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            conf = float(parts[6]) if len(parts) > 6 else 1.0
            cls = int(parts[7]) if len(parts) > 7 else 1
            vis = float(parts[8]) if len(parts) > 8 else 1.0
            if cls == 1 and conf > 0 and vis > 0:
                gt_frames[frame_id].append([tid, x, y, w, h])
    return gt_frames


# ═══════════════════════════════════════════════════════════════
# IoU 工具
# ═══════════════════════════════════════════════════════════════
def box_iou_xywh(boxes1, boxes2):
    """xywh → IoU [N1, N2]."""
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)))
    b1 = boxes1.copy()
    b2 = boxes2.copy()
    # 保存面积 (xywh 格式: w*h)
    area1 = b1[:, 2] * b1[:, 3]
    area2 = b2[:, 2] * b2[:, 3]
    # xywh → xyxy
    b1[:, 2:] = b1[:, :2] + b1[:, 2:]
    b2[:, 2:] = b2[:, :2] + b2[:, 2:]
    lt = np.maximum(b1[:, None, :2], b2[None, :, :2])
    rb = np.minimum(b1[:, None, 2:], b2[None, :, 2:])
    wh = np.maximum(0, rb - lt)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2[None, :] - inter + 1e-6
    return inter / union


# ═══════════════════════════════════════════════════════════════
# 在线追踪器 (SORT 式 IoU 关联)
# ═══════════════════════════════════════════════════════════════
class OnlineTracker:
    """
    带生命周期管理的在线追踪器，解决推理时轨迹雪崩效应。

    核心机制:
      - Birth Threshold (birth_conf): 新轨迹只在检测置信度 >= birth_conf 时创建
      - Keep Threshold (keep_conf):   已建立轨迹可匹配低至 keep_conf 的检测 (keep_conf < birth_conf)
      - Track Patience (patience):    轨迹失配后不立即死亡，保留 patience 帧等待恢复
      - 失配期间轨迹处于 "dormant" 状态: 不输出预测，但保留身份和状态
    """

    def __init__(self, iou_threshold=0.3, birth_conf=0.4, keep_conf=0.15,
                 patience=8, max_age=30):
        self.iou_threshold = iou_threshold
        self.birth_conf = birth_conf      # 新生阈值
        self.keep_conf = keep_conf        # 保持阈值 (< birth_conf)
        self.patience = patience          # 轨迹耐心帧数
        self.max_age = max_age            # 最多失配帧数后强制删除
        self.next_id = 0
        self.tracks = []                  # [{id, box, score, dormant_age, last_seen_frame}]
        self.frame_count = 0

    def update(self, det_boxes, det_scores):
        """
        Args:
            det_boxes: [N, 4] xywh pixel
            det_scores: [N] confidence
        Returns:
            tracked_ids: [N] assigned track ids (-1 = 未匹配或低于新生阈值)
        """
        n_det = len(det_boxes)
        assigned = np.full(n_det, -1, dtype=int)
        self.frame_count += 1

        # ─── 第一帧：所有高置信度检测创建轨迹 ───
        if len(self.tracks) == 0:
            for i in range(n_det):
                if det_scores[i] >= self.birth_conf:
                    tid = self.next_id
                    self.next_id += 1
                    self.tracks.append({
                        'id': tid, 'box': det_boxes[i], 'score': det_scores[i],
                        'dormant_age': 0, 'active': True,
                    })
                    assigned[i] = tid
            return assigned

        # ─── IoU 匹配：已有轨迹 ↔ 所有检测（使用 keep_conf 而非 birth_conf）───
        trk_boxes = np.array([t['box'] for t in self.tracks])
        matched_det = set()
        matched_trk = set()

        if n_det > 0:
            ious = box_iou_xywh(det_boxes, trk_boxes)
            cost = 1.0 - ious
            row_ind, col_ind = linear_sum_assignment(cost)

            for d, t in zip(row_ind, col_ind):
                if ious[d, t] >= self.iou_threshold and det_scores[d] >= self.keep_conf:
                    matched_det.add(d)
                    matched_trk.add(t)
                    assigned[d] = self.tracks[t]['id']
                    self.tracks[t]['box'] = det_boxes[d]
                    self.tracks[t]['score'] = det_scores[d]
                    self.tracks[t]['dormant_age'] = 0
                    self.tracks[t]['active'] = True

        # ─── 未匹配的检测：高于 birth_conf 才创建新轨迹 ───
        for i in range(n_det):
            if i not in matched_det and det_scores[i] >= self.birth_conf:
                tid = self.next_id
                self.next_id += 1
                self.tracks.append({
                    'id': tid, 'box': det_boxes[i], 'score': det_scores[i],
                    'dormant_age': 0, 'active': True,
                })
                assigned[i] = tid

        # ─── 未匹配的轨迹：增加 dormant_age，进入隐身状态 ───
        for t in range(len(self.tracks)):
            if t not in matched_trk:
                self.tracks[t]['dormant_age'] += 1
                self.tracks[t]['active'] = False

        # ─── 移除死亡轨迹 ───
        alive = []
        for t in range(len(self.tracks)):
            age = self.tracks[t]['dormant_age']
            if age <= self.max_age:
                alive.append(t)
        self.tracks = [self.tracks[t] for t in alive]

        return assigned

    def reset(self):
        self.next_id = 0
        self.tracks = []
        self.frame_count = 0


# ═══════════════════════════════════════════════════════════════
# MOT 指标计算
# ═══════════════════════════════════════════════════════════════
def compute_mot_metrics(all_gt, all_dt, iou_threshold=0.5):
    total_gt = 0
    total_dt = 0
    total_matched = 0
    total_fp = 0
    total_fn = 0
    total_mismatch = 0
    motp_sum = 0.0
    id_gt_map = {}
    id_dt_map = {}
    frame_ids = sorted(set(list(all_gt.keys()) + list(all_dt.keys())))
    prev_matches = {}

    for fid in frame_ids:
        gt_boxes = all_gt.get(fid, [])
        dt_boxes = all_dt.get(fid, [])
        total_gt += len(gt_boxes)
        total_dt += len(dt_boxes)

        if len(gt_boxes) == 0 or len(dt_boxes) == 0:
            total_fn += len(gt_boxes)
            total_fp += len(dt_boxes)
            for dt in dt_boxes:
                tid = dt[0]
                id_dt_map.setdefault(tid, set()).add(fid)
            for gt in gt_boxes:
                tid = gt[0]
                id_gt_map.setdefault(tid, set()).add(fid)
            continue

        gt_arr = np.array([[b[1], b[2], b[3], b[4]] for b in gt_boxes])
        dt_arr = np.array([[b[1], b[2], b[3], b[4]] for b in dt_boxes])
        gt_tids = [b[0] for b in gt_boxes]
        dt_tids = [b[0] for b in dt_boxes]

        iou_matrix = box_iou_xywh(gt_arr, dt_arr)
        cost = 1.0 - iou_matrix
        gt_idx, dt_idx = linear_sum_assignment(cost)

        matched_gt = set()
        matched_dt = set()
        n_matched = 0

        for g, d in zip(gt_idx, dt_idx):
            if iou_matrix[g, d] >= iou_threshold:
                matched_gt.add(g)
                matched_dt.add(d)
                n_matched += 1
                motp_sum += iou_matrix[g, d]

                gt_tid = gt_tids[g]
                dt_tid = dt_tids[d]
                id_gt_map.setdefault(gt_tid, set()).add(fid)
                id_dt_map.setdefault(dt_tid, set()).add(fid)

                if gt_tid in prev_matches and prev_matches[gt_tid] != dt_tid:
                    total_mismatch += 1
                prev_matches[gt_tid] = dt_tid

        total_matched += n_matched
        total_fn += len(gt_boxes) - len(matched_gt)
        total_fp += len(dt_boxes) - len(matched_dt)

        for i, dt in enumerate(dt_boxes):
            if i not in matched_dt:
                id_dt_map.setdefault(dt[0], set()).add(fid)
        for i, gt in enumerate(gt_boxes):
            if i not in matched_gt:
                id_gt_map.setdefault(gt[0], set()).add(fid)

    denom = max(total_gt, 1)
    mota = max(0, 1.0 - (total_fn + total_fp + total_mismatch) / denom) * 100
    motp = (motp_sum / max(total_matched, 1)) * 100
    precision = total_matched / max(total_dt, 1) * 100
    recall = total_matched / denom * 100

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


# ═══════════════════════════════════════════════════════════════
# 核心评估函数
# ═══════════════════════════════════════════════════════════════
def evaluate_model(model, dataset_name, split='val', img_size=640,
                   conf_threshold=0.3, max_frames=None, device='cuda'):
    """
    使用 HAMT 模型评估 MOT 指标。

    策略 v2:
      不使用记忆库的追踪（推理时有缺陷），而是直接取所有 detect query
      的预测框，配合独立的在线 IoU 追踪器做跨帧身份绑定。
    """
    data_root = DATA_ROOTS[dataset_name][split]
    model.eval()

    sequences = sorted([d for d in os.listdir(data_root)
                       if os.path.isdir(os.path.join(data_root, d))
                       and not d.startswith('.')])

    if dataset_name == 'mot17':
        if split == 'val':
            sequences = [s for s in sequences if s in MOT17_TRAIN_SEQ]
        else:
            sequences = [s for s in sequences if s not in MOT17_TRAIN_SEQ]

    print(f"\n{'='*60}")
    print(f"评估 {dataset_name.upper()} [{split}] — {len(sequences)} 个序列")
    print(f"{'='*60}")

    all_gt = {}
    all_dt = {}
    global_frame = 0
    tracker = OnlineTracker(iou_threshold=0.3, birth_conf=0.5, keep_conf=0.15, patience=10, max_age=30)

    for si, seq_name in enumerate(sequences):
        seq_path = os.path.join(data_root, seq_name)
        img_dir = os.path.join(seq_path, 'img1')
        gt_file = os.path.join(seq_path, 'gt', 'gt.txt')

        if not os.path.exists(img_dir):
            continue

        gt_data = read_gt_boxes(gt_file) if os.path.exists(gt_file) else {}

        # 获取原始图像尺寸
        from PIL import Image
        ini_path = os.path.join(seq_path, 'seqinfo.ini')
        orig_w, orig_h = 1920, 1080
        if os.path.exists(ini_path):
            import configparser
            cfg = configparser.ConfigParser()
            cfg.read(ini_path)
            if 'Sequence' in cfg:
                orig_w = int(cfg['Sequence'].get('imWidth', orig_w))
                orig_h = int(cfg['Sequence'].get('imHeight', orig_h))
        else:
            frame_files_tmp = sorted([f for f in os.listdir(img_dir)
                                     if f.endswith(('.jpg', '.png', '.jpeg'))])
            if frame_files_tmp:
                with Image.open(os.path.join(img_dir, frame_files_tmp[0])) as im:
                    orig_w, orig_h = im.size

        frame_files = sorted([f for f in os.listdir(img_dir)
                             if f.endswith(('.jpg', '.png', '.jpeg'))])
        if max_frames:
            frame_files = frame_files[:max_frames]

        # 重置记忆与追踪器
        model.memory_bank.reset()
        tracker.reset()

        n_detect_queries = model.num_detect_queries

        for frame_idx, fname in enumerate(frame_files):
            img_path = os.path.join(img_dir, fname)
            img_tensor = load_image(img_path, img_size).to(device)

            global_frame += 1

            with torch.no_grad():
                outputs = model(img_tensor)
                out = outputs[0] if isinstance(outputs, list) else outputs

            # ─── 提取所有检测预测 ───
            # pred_boxes: [1, N_track+N_detect, 4] cxcywh [0,1]
            # pred_logits: [1, N_track+N_detect, 1] (二分类logits)
            n_active = int(model.memory_bank.slot_count[0].item())
            n_total = out['pred_boxes'].shape[1]

            # 只取 detect queries 部分 (跳过 track queries)
            detect_start = n_active if n_active > 0 else 0
            remaining = n_total - detect_start

            if remaining > 0:
                pred_boxes = out['pred_boxes'][0, detect_start:].cpu()  # [remaining, 4]
                pred_logits = out['pred_logits'][0, detect_start:].cpu()  # [remaining, 1]
                scores = torch.sigmoid(pred_logits)[:, 0]                 # 前景概率

                # 高置信度过滤
                high_conf = scores > conf_threshold
                if high_conf.sum() > 0:
                    boxes_cxcywh = pred_boxes[high_conf].numpy()  # [K, 4]
                    det_scores = scores[high_conf].numpy()

                    # cxcywh [0,1] → xywh pixel
                    det_xywh = np.zeros_like(boxes_cxcywh)
                    det_xywh[:, 0] = (boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2) * orig_w
                    det_xywh[:, 1] = (boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2) * orig_h
                    det_xywh[:, 2] = boxes_cxcywh[:, 2] * orig_w
                    det_xywh[:, 3] = boxes_cxcywh[:, 3] * orig_h

                    # Clamp
                    det_xywh[:, 0] = np.clip(det_xywh[:, 0], 0, orig_w - 1)
                    det_xywh[:, 1] = np.clip(det_xywh[:, 1], 0, orig_h - 1)
                    det_xywh[:, 2] = np.clip(det_xywh[:, 2], 1, orig_w - det_xywh[:, 0])
                    det_xywh[:, 3] = np.clip(det_xywh[:, 3], 1, orig_h - det_xywh[:, 1])

                    # 在线追踪获取 track ids
                    track_ids = tracker.update(det_xywh, det_scores)

                    for i in range(len(det_xywh)):
                        tid = int(track_ids[i])
                        all_dt.setdefault(global_frame, []).append(
                            [tid, det_xywh[i, 0], det_xywh[i, 1],
                             det_xywh[i, 2], det_xywh[i, 3], float(det_scores[i])]
                        )

            # ─── 收集 GT ───
            mot_frame = frame_idx + 1
            for gt in gt_data.get(mot_frame, []):
                tid, gx, gy, gw, gh = gt
                all_gt.setdefault(global_frame, []).append([tid, gx, gy, gw, gh])

        # 进度
        if (si + 1) % 5 == 0 or si == 0:
            print(f"  已处理: {si+1}/{len(sequences)} 序列, "
                  f"当前 DT={sum(len(v) for v in all_dt.values())}  GT={sum(len(v) for v in all_gt.values())}")

    metrics = compute_mot_metrics(all_gt, all_dt)
    return metrics


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description='HAMT Model Evaluation')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--dataset', type=str, default='dancetrack',
                       choices=['dancetrack', 'mot17', 'mot20', 'all'])
    parser.add_argument('--split', type=str, default='val', choices=['val', 'test'])
    parser.add_argument('--conf', type=float, default=0.3, help='置信度阈值')
    parser.add_argument('--max-frames', type=int, default=None, help='每序列最大帧数')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--img-size', type=int, default=640)
    parser.add_argument('--output-dir', type=str, default=None,
                       help='保存结果目录 (默认与checkpoint同目录)')
    parser.add_argument('--sweep', type=str, default=None,
                       help='置信度阈值扫描, 逗号分隔, 例: "0.1,0.2,0.3,0.5,0.7"')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}")

    # 构建模型
    print("构建 HMAT 模型...")
    model = HMAT(
        num_classes=1,
        hidden_dim=256,
        num_queries=100,
        use_batch_memory=True,
    ).to(device)

    # 加载检查点
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"错误: 检查点不存在: {ckpt_path}")
        sys.exit(1)

    print(f"加载检查点: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get('model_state_dict', ckpt)

    model_dict = model.state_dict()
    filtered = {}
    skipped = []
    for k, v in state.items():
        key = k.replace('module.', '')
        if key in model_dict and model_dict[key].shape == v.shape:
            filtered[key] = v
        else:
            skipped.append(key)

    model.load_state_dict(filtered, strict=False)
    print(f"  加载权重: {len(filtered)} 项, 跳过: {len(skipped)} 项")

    # 输出目录
    output_dir = Path(args.output_dir) if args.output_dir else ckpt_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # 确定置信度阈值列表
    if args.sweep:
        conf_thresholds = [float(x.strip()) for x in args.sweep.split(',')]
        print(f"阈值扫描模式: {conf_thresholds}")
    else:
        conf_thresholds = [args.conf]

    # 评估
    datasets = [args.dataset] if args.dataset != 'all' else ['dancetrack', 'mot17', 'mot20']
    all_sweep_results = {}

    for ds_name in datasets:
        for conf in conf_thresholds:
            try:
                start = time.time()
                metrics = evaluate_model(
                    model, ds_name, split=args.split,
                    img_size=args.img_size, conf_threshold=conf,
                    max_frames=args.max_frames, device=device,
                )
                elapsed = time.time() - start

                key = f"{ds_name}_conf{conf:.1f}"
                all_sweep_results[key] = metrics

                print(f"\n{'='*60}")
                print(f"[{ds_name.upper()} | conf={conf:.1f}] 评估结果 ({elapsed:.0f}s):")
                print(f"  MOTA:      {metrics['MOTA']:.2f}%")
                print(f"  MOTP:      {metrics['MOTP']:.2f}%")
                print(f"  IDF1:      {metrics['IDF1']:.2f}%")
                print(f"  Precision: {metrics['Precision']:.2f}%")
                print(f"  Recall:    {metrics['Recall']:.2f}%")
                print(f"  GT={metrics['GT']}  DT={metrics['DT']}  "
                      f"FP={metrics['FP']}  FN={metrics['FN']}  "
                      f"IDSW={metrics['IDSW']}")
                print(f"{'='*60}")

            except Exception as e:
                print(f"\n  x {ds_name}/conf={conf} 评估失败: {e}")
                import traceback
                traceback.print_exc()

    # 保存结果
    if all_sweep_results:
        ckpt_stem = ckpt_path.stem
        suffix = "_sweep" if args.sweep else ""
        output_file = output_dir / f"eval_{ckpt_stem}_{args.dataset}_{args.split}{suffix}.json"
        with open(output_file, 'w') as f:
            json.dump({
                'checkpoint': str(ckpt_path),
                'dataset': args.dataset,
                'split': args.split,
                'conf_thresholds': conf_thresholds,
                'max_frames': args.max_frames,
                'results': all_sweep_results,
            }, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存: {output_file}")


if __name__ == '__main__':
    main()
