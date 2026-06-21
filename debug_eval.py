#!/usr/bin/env python3
"""诊断脚本: 逐帧打印模型预测与GT的详细对比，定位 IoU=0 的根因。"""
import sys, os
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
from hmat.modeling.hmat_model import HMAT

CKPT = PROJECT_ROOT / "results/hamt_deep_20260620_101740/phase3/phase3_best.pth"
SEQ_NAME = "dancetrack0004"
VAL_DIR = PROJECT_ROOT / "hmat/data/OpenDataLab___DanceTrack/val"
SEQ_PATH = VAL_DIR / SEQ_NAME
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


def read_gt_raw(gt_path, verbose=True):
    """读取 GT 文件并返回所有帧的数据（不过滤）。"""
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
            passed = cls == 1 and conf > 0 and vis > 0.1
            gt_frames[frame_id].append({
                'tid': tid, 'x': x, 'y': y, 'w': w, 'h': h,
                'conf': conf, 'cls': cls, 'vis': vis, 'passed': passed
            })
    return gt_frames


def box_iou_xywh(a, b):
    """xywh → IoU."""
    a_xyxy = np.column_stack([a[:, 0], a[:, 1], a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]])
    b_xyxy = np.column_stack([b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]])
    lt = np.maximum(a_xyxy[:, None, :2], b_xyxy[None, :, :2])
    rb = np.minimum(a_xyxy[:, None, 2:], b_xyxy[None, :, 2:])
    wh = np.maximum(0, rb - lt)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area_a = a[:, 2] * a[:, 3]
    area_b = b[:, 2] * b[:, 3]
    union = area_a[:, None] + area_b[None, :] - inter + 1e-6
    return inter / union


def main():
    print(f"设备: {DEVICE}")
    print(f"序列: {SEQ_NAME}")

    # 1. 加载模型
    print("\n--- 加载模型 ---")
    model = HMAT(num_classes=1, hidden_dim=256, num_queries=100, use_batch_memory=True).to(DEVICE)
    ckpt = torch.load(CKPT, map_location=DEVICE)
    state = ckpt.get('model_state_dict', ckpt)
    model_dict = model.state_dict()
    filtered = {}
    for k, v in state.items():
        key = k.replace('module.', '')
        if key in model_dict and model_dict[key].shape == v.shape:
            filtered[key] = v
    model.load_state_dict(filtered, strict=False)
    model.eval()
    print(f"加载权重: {len(filtered)}")

    # 2. 读取 GT
    gt_file = SEQ_PATH / "gt" / "gt.txt"
    print(f"\n--- 读取 GT ---")
    gt_raw = read_gt_raw(str(gt_file))
    print(f"GT 帧数: {len(gt_raw)}")

    # 3. 获取图像尺寸
    from PIL import Image
    import configparser
    ini_path = SEQ_PATH / "seqinfo.ini"
    orig_w, orig_h = 1920, 1080
    if ini_path.exists():
        cfg = configparser.ConfigParser()
        cfg.read(str(ini_path))
        if 'Sequence' in cfg:
            orig_w = int(cfg['Sequence'].get('imWidth', orig_w))
            orig_h = int(cfg['Sequence'].get('imHeight', orig_h))
    print(f"原始尺寸: {orig_w}x{orig_h}")

    # 4. 获取帧列表
    img_dir = SEQ_PATH / "img1"
    frame_files = sorted([f for f in os.listdir(str(img_dir)) if f.endswith(('.jpg', '.png', '.jpeg'))])
    frame_files = frame_files[:10]
    print(f"分析前 {len(frame_files)} 帧")

    # 5. 逐帧分析
    model.memory_bank.reset()
    for fi, fname in enumerate(frame_files):
        mot_frame = fi + 1
        img_path = str(img_dir / fname)
        img_tensor = load_image(img_path).to(DEVICE)

        with torch.no_grad():
            outputs = model(img_tensor)
            out = outputs[0] if isinstance(outputs, list) else outputs

        # 模型原始输出
        pred_boxes_raw = out['pred_boxes'][0].cpu()  # [Nq, 4] cxcywh [0,1]
        pred_logits = out['pred_logits'][0].cpu()    # [Nq, 1]
        scores = torch.sigmoid(pred_logits)[:, 0]     # 二分类: sigmoid

        n_active = int(model.memory_bank.slot_count[0].item())

        print(f"\n{'='*80}")
        print(f"Frame {mot_frame} ({fname}) | 活跃槽: {n_active}")

        # GT boxes (all, with and without filter)
        gt_entries = gt_raw.get(mot_frame, [])
        gt_passed = [e for e in gt_entries if e['passed']]
        gt_filtered = [e for e in gt_entries if not e['passed']]

        print(f"  GT 原始条目: {len(gt_entries)} (通过过滤: {len(gt_passed)}, 被过滤: {len(gt_filtered)})")
        for e in gt_entries:
            tag = "✓" if e['passed'] else "✗"
            print(f"    {tag} tid={e['tid']} box=[{e['x']:.0f},{e['y']:.0f},{e['w']:.0f},{e['h']:.0f}] "
                  f"conf={e['conf']} cls={e['cls']} vis={e['vis']}")

        # 模型 top-5 原始预测（按分数排序）
        top_k = torch.topk(scores, min(10, len(scores)))
        print(f"\n  模型 Top-10 预测 (原始 cxcywh [0,1]):")
        for rank, idx in enumerate(top_k.indices.tolist()):
            cx, cy, w_box, h_box = pred_boxes_raw[idx].tolist()
            sc = scores[idx].item()
            # 转换到像素
            x_px = (cx - w_box / 2) * orig_w
            y_px = (cy - h_box / 2) * orig_h
            w_px = w_box * orig_w
            h_px = h_box * orig_h
            print(f"    #{rank} idx={idx} score={sc:.4f} "
                  f"norm=[{cx:.4f},{cy:.4f},{w_box:.4f},{h_box:.4f}] "
                  f"pixel=[{x_px:.0f},{y_px:.0f},{w_px:.0f},{h_px:.0f}]")

        # 活跃槽的预测（track queries）
        if n_active > 0:
            print(f"\n  活跃槽预测 (前 {min(n_active, 5)} 个, pixel xywh):")
            pred_boxes_px = []
            for i in range(min(n_active, len(scores))):
                cx, cy, w_box, h_box = pred_boxes_raw[i].tolist()
                x_px = (cx - w_box / 2) * orig_w
                y_px = (cy - h_box / 2) * orig_h
                w_px = w_box * orig_w
                h_px = h_box * orig_h
                pred_boxes_px.append([x_px, y_px, w_px, h_px])
                track_id = int(model.memory_bank.track_ids[0, i].item())
                print(f"    slot={i} track_id={track_id} score={scores[i].item():.4f} "
                      f"box=[{x_px:.0f},{y_px:.0f},{w_px:.0f},{h_px:.0f}]")

            # 计算与 GT 的 IoU
            if gt_passed:
                gt_arr = np.array([[e['x'], e['y'], e['w'], e['h']] for e in gt_passed])
                dt_arr = np.array(pred_boxes_px)
                ious = box_iou_xywh(dt_arr, gt_arr)
                best_iou = ious.max(axis=1)
                best_gt = ious.argmax(axis=1)
                for i in range(len(pred_boxes_px)):
                    print(f"    slot={i}: 最佳 GT={best_gt[i]} (tid={gt_passed[best_gt[i]]['tid']}) IoU={best_iou[i]:.4f}")

        # 同时检测 queries 的 top-5
        detect_start = n_active if n_active > 0 else 0
        detect_scores = scores[detect_start:]
        if len(detect_scores) > 0:
            detect_top = torch.topk(detect_scores, min(5, len(detect_scores)))
            print(f"\n  检测Query Top-5 (索引 {detect_start}~):")
            for rank, idx in enumerate(detect_top.indices.tolist()):
                real_idx = detect_start + idx
                cx, cy, w_box, h_box = pred_boxes_raw[real_idx].tolist()
                x_px = (cx - w_box / 2) * orig_w
                y_px = (cy - h_box / 2) * orig_h
                w_px = w_box * orig_w
                h_px = h_box * orig_h
                print(f"    #{rank} idx={real_idx} score={detect_scores[idx].item():.4f} "
                      f"pixel=[{x_px:.0f},{y_px:.0f},{w_px:.0f},{h_px:.0f}]")


if __name__ == '__main__':
    main()
