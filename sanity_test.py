#!/usr/bin/env python3
"""验证 IoU 计算是否正确，以及 Phase2 在 MOT17 上的表现。"""
import sys, os
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
from hmat.modeling.hmat_model import HMAT

CKPT = PROJECT_ROOT / "results/hamt_deep_20260620_101740/phase2/phase2_best.pth"
VAL_DIR = PROJECT_ROOT / "hmat/data/OpenDataLab___MOT17/train"
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
    if not os.path.exists(gt_path): return gt_frames
    with open(gt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split(',')
            if len(parts) < 8: continue
            frame_id, tid = int(parts[0]), int(parts[1])
            x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            conf = float(parts[6]) if len(parts) > 6 else 1.0
            cls = int(parts[7]) if len(parts) > 7 else 1
            vis = float(parts[8]) if len(parts) > 8 else 1.0
            if cls == 1 and conf > 0 and vis > 0:
                gt_frames[frame_id].append([tid, x, y, w, h])
    return gt_frames


def box_iou_xywh(b1, b2):
    if len(b1) == 0 or len(b2) == 0: return np.zeros((len(b1), len(b2)))
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
    # 1. Sanity: self-IoU should be 1.0
    print("=== IoU 自检 ===")
    boxes = np.array([[100, 200, 50, 80]])  # xywh
    self_iou = box_iou_xywh(boxes, boxes)
    print(f"  Self IoU: {self_iou[0,0]:.4f} (期望 1.0)")

    # 2. Overlapping boxes
    b1 = np.array([[100, 200, 50, 80]])
    b2 = np.array([[110, 210, 50, 80]])  # 10px overlap
    iou = box_iou_xywh(b1, b2)
    print(f"  Partial overlap IoU: {iou[0,0]:.4f}")

    # 3. Non-overlapping boxes
    b3 = np.array([[1000, 200, 50, 80]])
    iou2 = box_iou_xywh(b1, b3)
    print(f"  No overlap IoU: {iou2[0,0]:.4f} (期望 0.0)")

    # 4. Test Phase2 on MOT17
    print(f"\n=== Phase2 on MOT17 ===")
    sequences = sorted([d for d in os.listdir(VAL_DIR)
                       if os.path.isdir(os.path.join(VAL_DIR, d)) and not d.startswith('.')])[:2]
    print(f"  序列: {sequences}")

    model = HMAT(num_classes=1, hidden_dim=256, num_queries=100, use_batch_memory=True).to(DEVICE)
    ckpt = torch.load(CKPT, map_location=DEVICE)
    state = ckpt.get('model_state_dict', ckpt)
    model_dict = model.state_dict()
    filtered = {}
    skipped = []
    for k, v in state.items():
        key = k.replace('module.','')
        if key in model_dict and model_dict[key].shape == v.shape:
            filtered[key] = v
        else:
            skipped.append(key)
    model.load_state_dict(filtered, strict=False)
    print(f"  加载权重: {len(filtered)}, 跳过: {len(skipped)}")
    if skipped:
        for s in skipped[:5]:
            print(f"    跳过: {s}")
    model.eval()

    all_best_ious = []

    for seq_name in sequences:
        seq_path = os.path.join(VAL_DIR, seq_name)
        img_dir = os.path.join(seq_path, 'img1')
        gt_file = os.path.join(seq_path, 'gt', 'gt.txt')
        if not os.path.exists(img_dir): continue

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
        print(f"  {seq_name}: orig={orig_w}x{orig_h}")

        frame_files = sorted([f for f in os.listdir(img_dir)
                             if f.endswith(('.jpg', '.png', '.jpeg'))])[:10]
        model.memory_bank.reset()

        for frame_idx, fname in enumerate(frame_files):
            img_tensor = load_image(os.path.join(img_dir, fname)).to(DEVICE)
            mot_frame = frame_idx + 1

            with torch.no_grad():
                outputs = model(img_tensor)
                out = outputs[0] if isinstance(outputs, list) else outputs

            pred_boxes = out['pred_boxes'][0].cpu().numpy()
            pred_logits = out['pred_logits'][0].cpu().numpy()
            scores = 1.0 / (1.0 + np.exp(-pred_logits[:, 0]))

            dt_xywh = np.zeros((pred_boxes.shape[0], 4))
            dt_xywh[:, 0] = (pred_boxes[:, 0] - pred_boxes[:, 2] / 2) * orig_w
            dt_xywh[:, 1] = (pred_boxes[:, 1] - pred_boxes[:, 3] / 2) * orig_h
            dt_xywh[:, 2] = pred_boxes[:, 2] * orig_w
            dt_xywh[:, 3] = pred_boxes[:, 3] * orig_h

            gt_entries = gt_data.get(mot_frame, [])
            if not gt_entries: continue
            gt_arr = np.array([[e[1], e[2], e[3], e[4]] for e in gt_entries])

            if len(gt_arr) > 0 and len(dt_xywh) > 0:
                ious = box_iou_xywh(gt_arr, dt_xywh)
                all_best_ious.extend(ious.max(axis=1).tolist())

                if frame_idx < 2:
                    # Print first frame details
                    print(f"\n  Frame {mot_frame}: {len(gt_arr)} GT, {len(dt_xywh)} predictions")
                    top_scores = np.argsort(scores)[-3:]
                    for i, idx in enumerate(top_scores[::-1]):
                        cx, cy, w, h_pred = pred_boxes[idx]
                        x_px = (cx - w/2) * orig_w
                        y_px = (cy - h_pred/2) * orig_h
                        w_px = w * orig_w
                        h_px = h_pred * orig_h
                        best_iou = ious[:, idx].max()
                        best_gt = ious[:, idx].argmax()
                        print(f"    Pred #{i}: score={scores[idx]:.4f} "
                              f"box=[{x_px:.0f},{y_px:.0f},{w_px:.0f},{h_px:.0f}] "
                              f"best_gt={best_gt} IoU={best_iou:.4f}")
                    for j, gt in enumerate(gt_arr):
                        print(f"    GT #{j}: box=[{gt[0]:.0f},{gt[1]:.0f},{gt[2]:.0f},{gt[3]:.0f}] "
                              f"tid={gt_entries[j][0]}")

    all_best_ious = np.array(all_best_ious)
    print(f"\n{'='*50}")
    print(f"IoU统计: n={len(all_best_ious)}")
    if len(all_best_ious) > 0:
        print(f"  Mean: {all_best_ious.mean():.4f}")
        print(f"  Max:  {all_best_ious.max():.4f}")
        for th in [0.1, 0.3, 0.5]:
            cnt = (all_best_ious >= th).sum()
            print(f"  IoU>={th}: {cnt} ({cnt/len(all_best_ious)*100:.1f}%)")


if __name__ == '__main__':
    main()
