#!/usr/bin/env python3
"""多数据集数据加载器。"""

from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parent.parent


class MOTDataset(Dataset):
    """MOT格式数据集加载器 (DanceTrack, MOT17, MOT20)."""

    def __init__(self, root_dir, dataset_name='dancetrack', split='train', img_size=(384, 640), max_frames_per_seq=None):
        self.root = Path(root_dir)
        self.dataset_name = dataset_name
        self.split = split
        self.img_size = img_size  # (H, W)
        self.max_frames_per_seq = max_frames_per_seq
        self.sequence_frame_counts = []
        self.sequences = []
        self._scan_sequences()
        print(f"[{dataset_name.upper()}] {split}: {len(self.sequences)} sequences, {self.total_frames} frames")

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

    def _load_gt(self, gt_file):
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
                annotations[frame_id].append({
                    'track_id': track_id,
                    'bbox': [x, y, w, h],
                    'conf': conf,
                    'class': class_id,
                    'visibility': vis,
                })
        return annotations

    @staticmethod
    def _to_tensor(img):
        arr = np.array(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=tensor.dtype).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=tensor.dtype).view(3, 1, 1)
        return (tensor - mean) / std

    def __getitem__(self, idx):
        seq_dir = self.sequences[idx]
        img_dir = seq_dir / 'img1'
        gt_file = seq_dir / 'gt' / 'gt.txt'

        img_files = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}])
        if self.max_frames_per_seq is not None and self.max_frames_per_seq > 0:
            img_files = img_files[:self.max_frames_per_seq]

        annotations = self._load_gt(gt_file)
        frames = []

        for frame_idx, img_file in enumerate(img_files, start=1):
            with Image.open(img_file) as img:
                img = img.convert('RGB')
                if self.img_size is not None:
                    h, w = self.img_size
                    img = img.resize((w, h), Image.BILINEAR)
                img_tensor = self._to_tensor(img)

            frames.append({
                'image': img_tensor,
                'frame_id': frame_idx,
                'annotations': annotations.get(frame_idx, []),
                'img_path': str(img_file),
            })

        return {
            'seq_name': seq_dir.name,
            'seq_dir': str(seq_dir),
            'frames': frames,
            'num_frames': len(frames),
        }


def collate_sequences(batch):
    return batch


def create_dataloader(root_dir, dataset_name, split, batch_size=1, num_workers=0, img_size=(384, 640), max_frames_per_seq=None):
    root_path = Path(root_dir)
    if not root_path.exists():
        if dataset_name == 'dancetrack':
            fallback = PROJECT_ROOT / 'data' / 'DanceTrack' / split
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
        pin_memory=torch.cuda.is_available(),
    )


if __name__ == '__main__':
    dt = create_dataloader('/home/user/MOT项目/HAMT/data/DanceTrack/train', 'dancetrack', 'train')
    print(f"DanceTrack: {len(dt)} batches")
