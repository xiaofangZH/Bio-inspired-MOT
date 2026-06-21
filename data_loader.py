#!/usr/bin/env python3
"""多数据集数据加载器（延迟加载版 — 避免全量加载导致 OOM）。

关键设计：
- MOTDataset.__getitem__ 只返回图像路径和标注元数据，不加载任何图像张量。
- 图像加载由 Trainer 在训练循环中按需逐帧完成，内存占用仅 batch_size=1 帧。
"""

from pathlib import Path
from collections import defaultdict
import configparser

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parent.parent

# 硬性上限：单序列最多处理的帧数，防止极长序列（如 DanceTrack 3000+ 帧）
# 即使 YAML 未设置 max_frames_per_seq，也会被这个安全上限截断。
_SAFETY_MAX_FRAMES = 300


def load_frame_tensor(img_path: str, img_size: tuple) -> torch.Tensor:
    """加载单帧图像并转为归一化 Tensor [3, H, W]（独立函数，供 Trainer 按需调用）。

    替代原来 MOTDataset._to_tensor 的内联加载逻辑，避免在 Dataset 阶段
    一次性将所有帧加载到内存中。
    """
    with Image.open(img_path) as img:
        img = img.convert('RGB')
        if img_size is not None:
            h, w = img_size
            img = img.resize((w, h), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=tensor.dtype).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=tensor.dtype).view(3, 1, 1)
        return (tensor - mean) / std


class MOTDataset(Dataset):
    """MOT格式数据集加载器 (DanceTrack, MOT17, MOT20) — 延迟加载版。

    每个样本返回一个序列的元数据（图像路径 + 标注），不加载图像张量。
    图像由调用方按需逐帧加载。
    """

    def __init__(self, root_dir, dataset_name='dancetrack', split='train', img_size=(384, 640), max_frames_per_seq=None):
        self.root = Path(root_dir)
        self.dataset_name = dataset_name
        self.split = split
        self.img_size = img_size  # (H, W)
        # max_frames_per_seq 受安全上限约束
        if max_frames_per_seq is None or max_frames_per_seq <= 0:
            self.max_frames_per_seq = _SAFETY_MAX_FRAMES
        else:
            self.max_frames_per_seq = min(max_frames_per_seq, _SAFETY_MAX_FRAMES)
        self.sequence_frame_counts = []
        self.sequences = []
        self._scan_sequences()
        safe_note = " (安全上限)" if self.max_frames_per_seq == _SAFETY_MAX_FRAMES else ""
        print(f"[{dataset_name.upper()}] {split}: {len(self.sequences)} sequences, "
              f"{self.total_frames} frames, max_frames_per_seq={self.max_frames_per_seq}{safe_note}")

    def _parse_seqinfo(self, seq_dir: Path):
        """解析 seqinfo.ini 获取原始图像尺寸。

        Returns:
            (img_width, img_height) 原始图像尺寸，用于 bbox 归一化。
            若 seqinfo.ini 不存在，则尝试从首帧图像推断。
        """
        ini_path = seq_dir / 'seqinfo.ini'
        if ini_path.exists():
            config = configparser.ConfigParser()
            config.read(str(ini_path))
            if 'Sequence' in config:
                w = int(config['Sequence'].get('imWidth', 0))
                h = int(config['Sequence'].get('imHeight', 0))
                if w > 0 and h > 0:
                    return w, h
        # 回退：从首帧图像推断尺寸
        img_dir = seq_dir / 'img1'
        if img_dir.exists():
            imgs = sorted(img_dir.glob('*.jpg'))
            if not imgs:
                imgs = sorted(img_dir.glob('*.png'))
            if imgs:
                with Image.open(imgs[0]) as im:
                    return im.size  # (W, H)
        return 1920, 1080  # 最终回退默认值

    def _scan_sequences(self):
        if not self.root.exists():
            return
        for seq_dir in sorted(self.root.iterdir()):
            if not seq_dir.is_dir():
                continue
            if (seq_dir / 'img1').exists() and (seq_dir / 'gt' / 'gt.txt').exists():
                self.sequences.append(seq_dir)
                self.sequence_frame_counts.append(len(list((seq_dir / 'img1').glob('*.jpg'))))

    def __len__(self):
        return len(self.sequences)

    @property
    def total_frames(self):
        return sum(self.sequence_frame_counts)

    def _load_gt(self, gt_file, img_width=1920, img_height=1080):
        """加载 GT 标注，并使用原始图像尺寸预计算归一化 bbox。

        关键修复：bbox 必须用 *原始* 图像尺寸归一化（来自 seqinfo.ini），
        而非 resize 后的尺寸。train.py 的 _build_targets 会优先使用 bbox_norm，
        避免用 640x640 的 resize 尺寸错误归一化原始坐标空间下的 bbox。

        Args:
            gt_file: gt.txt 路径
            img_width: 原始图像宽度（来自 seqinfo.ini）
            img_height: 原始图像高度（来自 seqinfo.ini）
        """
        annotations = defaultdict(list)
        if not gt_file.exists():
            return annotations

        with open(gt_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 6:
                    continue
                frame_id = int(parts[0])
                track_id = int(parts[1])
                x, y, w, h = map(float, parts[2:6])
                conf = float(parts[6]) if len(parts) > 6 else 1.0
                class_id = int(parts[7]) if len(parts) > 7 else 1
                vis = float(parts[8]) if len(parts) > 8 else 1.0
                if conf < 0 or vis < 0:
                    continue
                if w <= 0 or h <= 0:
                    continue
                # 用原始图像尺寸归一化 (cx, cy, w, h) -> [0,1]
                cx = (x + 0.5 * w) / max(img_width, 1)
                cy = (y + 0.5 * h) / max(img_height, 1)
                nw = w / max(img_width, 1)
                nh = h / max(img_height, 1)
                # 钳制到合理范围
                cx = min(max(cx, 0.0), 1.0)
                cy = min(max(cy, 0.0), 1.0)
                nw = min(max(nw, 1e-6), 1.0)
                nh = min(max(nh, 1e-6), 1.0)
                annotations[frame_id].append({
                    'track_id': track_id,
                    'bbox': [x, y, w, h],
                    'bbox_norm': [cx, cy, nw, nh],
                    'conf': conf,
                    'class': class_id,
                    'visibility': vis,
                })
        return annotations

    def __getitem__(self, idx):
        """返回序列元数据（仅路径+标注），不加载任何图像张量。

        Returns:
            dict with keys:
                seq_name, seq_dir, img_paths (list[str]),
                img_size (tuple), annotations (dict frame_id->list),
                num_frames (int), orig_img_size (tuple (W, H))
        """
        seq_dir = self.sequences[idx]
        img_dir = seq_dir / 'img1'
        gt_file = seq_dir / 'gt' / 'gt.txt'

        img_files = sorted([p for p in img_dir.iterdir()
                           if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}])
        if self.max_frames_per_seq is not None and self.max_frames_per_seq > 0:
            img_files = img_files[:self.max_frames_per_seq]

        # 解析原始图像尺寸（来自 seqinfo.ini），用于 bbox 归一化
        orig_w, orig_h = self._parse_seqinfo(seq_dir)
        annotations = self._load_gt(gt_file, img_width=orig_w, img_height=orig_h)

        return {
            'seq_name': seq_dir.name,
            'seq_dir': str(seq_dir),
            'img_paths': [str(f) for f in img_files],
            'img_size': self.img_size,
            'annotations': annotations,  # dict: frame_id -> list of annotation dicts
            'num_frames': len(img_files),
            'orig_img_size': (orig_w, orig_h),  # 原始图像尺寸 (W, H)
        }


class DanceTrackFrameGapDataset(MOTDataset):
    """DanceTrack 数据集 — 支持帧间隔采样。
    
    使用 frame_gap 跳过间隔帧，强迫模型学习更长距离的运动变化。
    例如 frame_gap=3 时读取 (T, T+3, T+6, ...) 而不是连续帧。
    """
    
    def __init__(self, root_dir, img_size=(384, 640), max_frames_per_seq=None, frame_gap=3):
        self.frame_gap = frame_gap
        super().__init__(
            root_dir=root_dir,
            dataset_name='dancetrack',
            split='train',
            img_size=img_size,
            max_frames_per_seq=max_frames_per_seq,
        )
    
    def __getitem__(self, idx):
        """返回帧间隔采样的序列元数据。
        
        跳帧采样: 读取 frame 0, frame_gap, 2*frame_gap, ...
        而不是连续的 frame 0, 1, 2, ...
        """
        seq_dir = self.sequences[idx]
        img_dir = seq_dir / 'img1'
        gt_file = seq_dir / 'gt' / 'gt.txt'
        
        img_files = sorted([p for p in img_dir.iterdir()
                           if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}])
        
        # 帧间隔采样
        sampled = img_files[::self.frame_gap]
        
        if self.max_frames_per_seq is not None and self.max_frames_per_seq > 0:
            sampled = sampled[:self.max_frames_per_seq]
        
        orig_w, orig_h = self._parse_seqinfo(seq_dir)
        annotations = self._load_gt(gt_file, img_width=orig_w, img_height=orig_h)
        
        return {
            'seq_name': seq_dir.name,
            'seq_dir': str(seq_dir),
            'img_paths': [str(f) for f in sampled],
            'img_size': self.img_size,
            'annotations': annotations,
            'num_frames': len(sampled),
            'orig_img_size': (orig_w, orig_h),
            'frame_gap': self.frame_gap,
        }


def collate_sequences(batch):
    return batch


def create_phase_dataloader(
    phase: str,
    img_size=(640, 640),
    batch_size=1,
    num_workers=8,
    config=None,
):
    """
    为各训练阶段创建对应的 DataLoader。

    Phase 1 (crowdhuman): CrowdHumanPseudoVideoDataset — 伪视频检测筑基
    Phase 2 (mot): MOT17 + MOT20 — 线性运动关联
    Phase 3 (dancetrack): DanceTrack — 极难域微调（随机帧间隔采样）
    """
    if phase == 'crowdhuman' or phase == 'phase1':
        from hmat.data.crowdhuman_dataset import create_crowdhuman_dataloader
        data_root = str(PROJECT_ROOT / 'hmat' / 'data' / 'OpenDataLab___CrowdHuman')
        ann_path = str(Path(data_root) / 'annotation_train.odgt')
        return create_crowdhuman_dataloader(
            data_root=data_root,
            annotation_path=ann_path,
            img_size=img_size,
            batch_size=batch_size,
            num_workers=num_workers,
        )
    elif phase == 'mot' or phase == 'phase2':
        # MOT17
        mot17_root = str(PROJECT_ROOT / 'hmat' / 'data' / 'OpenDataLab___MOT17' / 'train')
        mot17_dataset = MOTDataset(
            root_dir=mot17_root,
            dataset_name='mot17',
            split='train',
            img_size=img_size,
            max_frames_per_seq=None,
        )
        
        # MOT20
        mot20_root = str(PROJECT_ROOT / 'hmat' / 'data' / 'OpenDataLab___MOT20' / 'train')
        mot20_dataset = MOTDataset(
            root_dir=mot20_root,
            dataset_name='mot20',
            split='train',
            img_size=img_size,
            max_frames_per_seq=None,
        )
        
        combined = ConcatDataset([mot17_dataset, mot20_dataset])
        sample_weights = \
            [len(mot17_dataset)] * len(mot17_dataset) + \
            [len(mot20_dataset)] * len(mot20_dataset)
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(combined))
        
        return DataLoader(
            combined,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=collate_sequences,
            pin_memory=True,
            prefetch_factor=2 if num_workers > 0 else None,
        )
    elif phase == 'dancetrack' or phase == 'phase3':
        dancetrack_root = str(PROJECT_ROOT / 'hmat' / 'data' / 'OpenDataLab___DanceTrack' / 'train')
        dataset = DanceTrackFrameGapDataset(
            root_dir=dancetrack_root,
            img_size=img_size,
            max_frames_per_seq=None,
            frame_gap=3,  # T and T+3, not T and T+1
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_sequences,
            pin_memory=True,
            prefetch_factor=2 if num_workers > 0 else None,
        )
    else:
        raise ValueError(f"Unknown phase: {phase}. Expected 'crowdhuman'/'phase1', 'mot'/'phase2', or 'dancetrack'/'phase3'.")


def create_dataloader(root_dir, dataset_name, split, batch_size=1, num_workers=0, img_size=(384, 640), max_frames_per_seq=None):
    root_path = Path(root_dir)
    # 相对路径基于 PROJECT_ROOT 解析，避免依赖 CWD
    if not root_path.is_absolute():
        root_path = (PROJECT_ROOT / root_path).resolve()
    if not root_path.exists():
        if dataset_name == 'dancetrack':
            fallback = PROJECT_ROOT / 'hmat' / 'data' / 'OpenDataLab___DanceTrack' / 'raw' / split
        elif dataset_name == 'mot17':
            fallback = WORKSPACE_ROOT / 'MOT17' / 'MOT17' / split
        elif dataset_name == 'mot20':
            fallback = WORKSPACE_ROOT / 'MOT20' / 'MOT20' / split
        else:
            fallback = root_path

        if fallback.exists():
            print(f"[DATA_LOADER] root '{root_dir}' not found, fallback -> {fallback}")
            root_path = fallback

    dataset = MOTDataset(
        root_dir=str(root_path),
        dataset_name=dataset_name,
        split=split,
        img_size=img_size,
        max_frames_per_seq=max_frames_per_seq,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        collate_fn=collate_sequences,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
    )


if __name__ == '__main__':
    dt = create_dataloader('hmat/data/OpenDataLab___DanceTrack/raw/train', 'dancetrack', 'train')
    print(f"DanceTrack: {len(dt)} batches")
