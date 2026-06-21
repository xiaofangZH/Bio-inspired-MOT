#!/usr/bin/env python3
"""诊断脚本 v2: 分析所有预测框与 GT 的 IoU 分布，定位定位偏差根源。"""
import sys, os
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
from hmat.modeling.hmat_model import HMAT

CKPT = PROJECT_ROOT / "results/hamt_deep_20260620_101740/phase3/phase3_best.pth"
VAL_DIR = PROJECT_ROOT / "hmat/data/OpenDataLab___DanceTrack/val"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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


def read_gt(gt_path):
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


def box_iou_xywh(b1, b2):
    b1 = b1.copy(); b2 = b2.copy()
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


def main():
    print(f"设备: {DEVICE}")

    # 加载模型
    print("加载模型...")
    model = HMAT(num_classes=1, hidden_dim=256, num_queries=100, use_batch_memory=True).to(DEVICE)
    ckpt = torch.load(CKPT, map_location=DEVICE)
    state = ckpt.get('model_state_dict', ckpt)
    model_dict = model.state_dict()
    filtered = {k.replace('module.',''): v for k, v in state.items()
                if k.replace('module.','') in model_dict and model_dict[k.replace('module.','')].shape == v.shape}
    model.load_state_dict(filtered, strict=False)
    model.eval()

    # 遍历前 3 个序列，每序列取前 20 帧
    sequences = sorted([d for d in os.listdir(VAL_DIR)
                       if os.path.isdir(os.path.join(VAL_DIR, d)) and not d.startswith('.')])[:3]

    all_best_ious = []  # 每帧每个GT的最好IoU
    all_pred_boxes = []  # 收集所有预测框的统计
    all_gt_boxes = []    # 收集所有GT框的统计

    for seq_name in sequences:
        seq_path = os.path.join(VAL_DIR, seq_name)
        img_dir = os.path.join(seq_path, 'img1')
        gt_file = os.path.join(seq_path, 'gt', 'gt.txt')

        if not os.path.exists(img_dir):
            continue

        gt_data = read_gt(gt_file)

        import configparser
        ini_path = os.path.join(seq_path, 'seqinfo.ini')
        orig_w, orig_h = 1920, 1080
        if os.path.exists(ini_path):
            cfg = configparser.ConfigParser()
            cfg.read(ini_path)
            if 'Sequence' in cfg:
                orig_w = int(cfg['Sequence'].get('imWidth', orig_w))
                orig_h = int(cfg['Sequence'].get('imHeight', orig_h))

        frame_files = sorted([f for f in os.listdir(img_dir)
                             if f.endswith(('.jpg', '.png', '.jpeg'))])[:20]

        model.memory_bank.reset()

        for frame_idx, fname in enumerate(frame_files):
            img_tensor = load_image(os.path.join(img_dir, fname)).to(DEVICE)
            mot_frame = frame_idx + 1

            with torch.no_grad():
                outputs = model(img_tensor)
                out = outputs[0] if isinstance(outputs, list) else outputs

            n_active = int(model.memory_bank.slot_count[0].item())
            detect_start = n_active if n_active > 0 else 0
            n_total = out['pred_boxes'].shape[1]

            # 使用全部 query 预测 (track + detect)
            pred_boxes = out['pred_boxes'][0].cpu().numpy()  # [N_total, 4] cxcywh [0,1]
            pred_logits = out['pred_logits'][0].cpu().numpy()  # [N_total, 1]
            scores = 1.0 / (1.0 + np.exp(-pred_logits[:, 0]))  # sigmoid

            # 转换为 xywh pixel
            dt_xywh = np.zeros((pred_boxes.shape[0], 4))
            dt_xywh[:, 0] = (pred_boxes[:, 0] - pred_boxes[:, 2] / 2) * orig_w
            dt_xywh[:, 1] = (pred_boxes[:, 1] - pred_boxes[:, 3] / 2) * orig_h
            dt_xywh[:, 2] = pred_boxes[:, 2] * orig_w
            dt_xywh[:, 3] = pred_boxes[:, 3] * orig_h

            # GT boxes
            gt_entries = gt_data.get(mot_frame, [])
            if not gt_entries:
                continue

            gt_arr = np.array([[e[1], e[2], e[3], e[4]] for e in gt_entries])

            # 收集盒子统计
            all_pred_boxes.append(dt_xywh)
            all_gt_boxes.append(gt_arr)

            # 计算所有预测 vs GT 的 IoU
            if len(gt_arr) > 0 and len(dt_xywh) > 0:
                ious = box_iou_xywh(gt_arr, dt_xywh)  # [Ng, Nd]
                best_per_gt = ious.max(axis=1)
                all_best_ious.extend(best_per_gt.tolist())

        print(f"  {seq_name}: {len(frame_files)} 帧完成")

    # ─── 统计分布 ───
    all_best_ious = np.array(all_best_ious)
    print(f"\n{'='*60}")
    print(f"IoU 分布统计 (所有帧, 每个GT框的最佳匹配IoU):")
    print(f"  样本数: {len(all_best_ious)}")
    if len(all_best_ious) > 0:
        print(f"  Mean:   {all_best_ious.mean():.4f}")
        print(f"  Median: {np.median(all_best_ious):.4f}")
        print(f"  Max:    {all_best_ious.max():.4f}")
        print(f"  Min:    {all_best_ious.min():.4f}")
        print(f"  Std:    {all_best_ious.std():.4f}")

        # 分位数
        for p in [25, 50, 75, 90, 95, 99]:
            print(f"  P{p}:    {np.percentile(all_best_ious, p):.4f}")

        # 按范围统计
        thresholds = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        print(f"\n  IoU 范围分布:")
        for i in range(len(thresholds) - 1):
            lo, hi = thresholds[i], thresholds[i+1]
            count = ((all_best_ious >= lo) & (all_best_ious < hi)).sum()
            pct = count / len(all_best_ious) * 100
            print(f"    [{lo:.1f}, {hi:.1f}): {count:5d} ({pct:5.1f}%)")

    # 预测框 vs GT 框的尺寸分布
    if all_pred_boxes:
        pred_all = np.concatenate(all_pred_boxes, axis=0)
        gt_all = np.concatenate(all_gt_boxes, axis=0)

        # 过滤无效框
        pred_valid = pred_all[(pred_all[:, 2] > 0) & (pred_all[:, 3] > 0)]
        gt_valid = gt_all[(gt_all[:, 2] > 0) & (gt_all[:, 3] > 0)]

        print(f"\n  预测框数量: {len(pred_valid)}, GT框数量: {len(gt_valid)}")
        print(f"  GT    宽: mean={gt_all[:, 2].mean():.1f} median={np.median(gt_all[:, 2]):.1f} "
              f"高: mean={gt_all[:, 3].mean():.1f} median={np.median(gt_all[:, 3]):.1f}")
        print(f"  预测  宽: mean={pred_all[:, 2].mean():.1f} median={np.median(pred_all[:, 2]):.1f} "
              f"高: mean={pred_all[:, 3].mean():.1f} median={np.median(pred_all[:, 3]):.1f}")
        print(f"  GT    中心: x_mean={gt_all[:, 0]+gt_all[:, 2]/2:.mean():.0f} "
              f"y_mean={(gt_all[:, 1]+gt_all[:, 3]/2):.mean():.0f}")
        print(f"  预测  中心: x_mean={pred_all[:, 0]+pred_all[:, 2]/2:.mean():.0f} "
              f"y_mean={(pred_all[:, 1]+pred_all[:, 3]/2):.mean():.0f}")


if __name__ == '__main__':
    main()
