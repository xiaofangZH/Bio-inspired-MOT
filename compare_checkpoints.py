#!/usr/bin/env python3
"""快速测试: Phase 1/2/3 检查点的检测定位能力对比 (IoU分布)。"""
import sys, os
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
from hmat.modeling.hmat_model import HMAT

CHECKPOINTS = {
    'Phase1': PROJECT_ROOT / "results/hamt_deep_20260620_101740/phase1/phase1_best.pth",
    'Phase2': PROJECT_ROOT / "results/hamt_deep_20260620_101740/phase2/phase2_best.pth",
    'Phase3': PROJECT_ROOT / "results/hamt_deep_20260620_101740/phase3/phase3_best.pth",
}
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


def test_checkpoint(name, ckpt_path, sequences):
    print(f"\n{'='*60}")
    print(f"测试: {name}")
    print(f"{'='*60}")

    model = HMAT(num_classes=1, hidden_dim=256, num_queries=100, use_batch_memory=True).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    state = ckpt.get('model_state_dict', ckpt)
    model_dict = model.state_dict()
    filtered = {k.replace('module.',''): v for k, v in state.items()
                if k.replace('module.','') in model_dict and model_dict[k.replace('module.','')].shape == v.shape}
    model.load_state_dict(filtered, strict=False)
    model.eval()

    all_best_ious = []

    for seq_name in sequences[:2]:  # 前2个序列,每序列20帧
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

        frame_files = sorted([f for f in os.listdir(img_dir)
                             if f.endswith(('.jpg', '.png', '.jpeg'))])[:20]
        model.memory_bank.reset()

        for frame_idx, fname in enumerate(frame_files):
            img_tensor = load_image(os.path.join(img_dir, fname)).to(DEVICE)
            mot_frame = frame_idx + 1

            with torch.no_grad():
                outputs = model(img_tensor)
                out = outputs[0] if isinstance(outputs, list) else outputs

            pred_boxes = out['pred_boxes'][0].cpu().numpy()
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

    all_best_ious = np.array(all_best_ious)
    if len(all_best_ious) == 0:
        print("  无数据")
        return {'n': 0, 'mean': 0, 'max': 0, 'iou_ge_0.5': 0}

    stats = {
        'n': int(len(all_best_ious)),
        'mean': float(all_best_ious.mean()),
        'median': float(np.median(all_best_ious)),
        'max': float(all_best_ious.max()),
        'p90': float(np.percentile(all_best_ious, 90)),
        'p95': float(np.percentile(all_best_ious, 95)),
        'iou_ge_0.5': int((all_best_ious >= 0.5).sum()),
        'iou_ge_0.3': int((all_best_ious >= 0.3).sum()),
        'iou_ge_0.1': int((all_best_ious >= 0.1).sum()),
    }

    print(f"  样本数: {stats['n']}")
    print(f"  Mean:   {stats['mean']:.4f}")
    print(f"  Median: {stats['median']:.4f}")
    print(f"  Max:    {stats['max']:.4f}")
    print(f"  P90:    {stats['p90']:.4f}")
    print(f"  P95:    {stats['p95']:.4f}")
    print(f"  IoU>=0.5: {stats['iou_ge_0.5']} ({stats['iou_ge_0.5']/stats['n']*100:.1f}%)")
    print(f"  IoU>=0.3: {stats['iou_ge_0.3']} ({stats['iou_ge_0.3']/stats['n']*100:.1f}%)")
    print(f"  IoU>=0.1: {stats['iou_ge_0.1']} ({stats['iou_ge_0.1']/stats['n']*100:.1f}%)")

    return stats


def main():
    import json
    sequences = sorted([d for d in os.listdir(VAL_DIR)
                       if os.path.isdir(os.path.join(VAL_DIR, d)) and not d.startswith('.')])

    all_results = {}
    for name, ckpt_path in CHECKPOINTS.items():
        if ckpt_path.exists():
            stats = test_checkpoint(name, ckpt_path, sequences)
            all_results[name] = stats
        else:
            print(f"\n{name}: 检查点不存在: {ckpt_path}")

    # 保存结果
    output_file = PROJECT_ROOT / "results" / "hamt_deep_20260620_101740" / "compare_iou.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n对比结果已保存: {output_file}")


if __name__ == '__main__':
    main()
