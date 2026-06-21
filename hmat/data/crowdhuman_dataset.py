#!/usr/bin/env python3
"""CrowdHuman 数据集 — 静态图转伪视频序列，用于检测筑基训练(Phase 1).

关键设计:
- 从单张 CrowdHuman 图片通过仿射变换生成 2 帧伪视频序列。
- 两帧间同一个人共享相同 track_id。
- 使用 fbox (全身边界框) 作为 Ground Truth，丢弃 hbox (头部框)。
- extra.ignore == 1 的标注标记为忽略区域。
"""

import json
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from hmat.data.target_parser import TargetParser


# ImageNet 归一化参数
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def load_crowdhuman_odgt(odgt_path: str):
    """加载 CrowdHuman ODGT 格式标注文件。

    每行一个 JSON 对象，格式:
    {"ID": "image_path", "gtboxes": [{"fbox": [...], "hbox": [...], "extra": {...}}, ...]}
    """
    samples = []
    with open(odgt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
                samples.append(sample)
            except json.JSONDecodeError:
                continue
    return samples


class CrowdHumanPseudoVideoDataset(Dataset):
    """CrowdHuman 伪视频数据集。

    每张静态图通过小幅度仿射变换生成 2 帧序列：
    - Frame 0: 原始图像 + 变换 A
    - Frame 1: 同一图像 + 变换 B
    两帧共享同一 track_id，模拟相邻帧的追踪任务。

    不依赖集成的 MemoryBank － Phase 1 只训练检测能力。
    """

    def __init__(
        self,
        data_root: str,
        annotation_path: str,
        img_size: tuple = (640, 640),
        phase: str = 'train',
        max_samples: Optional[int] = None,
    ):
        self.data_root = Path(data_root)
        self.img_size = img_size  # (H, W)
        self.phase = phase

        # 加载 ODGT 标注
        print(f"[CrowdHuman] 加载标注: {annotation_path}")
        self.annotations = load_crowdhuman_odgt(annotation_path)

        # 预过滤：只保留有有效 fbox 标注的样本
        self.valid_samples = []
        for ann in self.annotations:
            if 'gtboxes' in ann and len(ann['gtboxes']) > 0:
                self.valid_samples.append(ann)

        if max_samples is not None and max_samples > 0:
            self.valid_samples = self.valid_samples[:max_samples]

        print(f"[CrowdHuman] {phase}: {len(self.valid_samples)} valid images "
              f"(from {len(self.annotations)} total)")

        # 仿射变换参数范围
        self.translate_range = 0.03      # 最大平移 (相对于图像尺寸)
        self.scale_range = 0.05          # 最大缩放变化
        self.rotation_degrees = 2.0      # 最大旋转角度

    def __len__(self):
        return len(self.valid_samples)

    def _generate_affine_params(self, h, w):
        """生成随机仿射变换参数 (平移 + 缩放 + 旋转)。

        Returns:
            affine_matrix: 2x3 numpy array
        """
        # 平移 (相对于图像尺寸)
        dx = random.uniform(-self.translate_range, self.translate_range) * w
        dy = random.uniform(-self.translate_range, self.translate_range) * h

        # 缩放 (以图像中心为原点)
        scale = 1.0 + random.uniform(-self.scale_range, self.scale_range)

        # 旋转
        angle = random.uniform(-self.rotation_degrees, self.rotation_degrees)

        # 构建仿射矩阵: 平移 + 缩放 + 旋转
        cx, cy = w / 2.0, h / 2.0
        cos_a = np.cos(np.deg2rad(angle))
        sin_a = np.sin(np.deg2rad(angle))

        # 组合矩阵: translate_to_origin * scale * rotate * translate_back * translate_final
        M = np.array([
            [scale * cos_a, -scale * sin_a, dx + cx - scale * (cx * cos_a - cy * sin_a)],
            [scale * sin_a,  scale * cos_a, dy + cy - scale * (cx * sin_a + cy * cos_a)],
        ], dtype=np.float32)
        return M

    def _apply_affine_to_boxes(self, boxes, M, h, w):
        """对 boxes [x,y,w,h] 绝对坐标应用仿射变换矩阵 M。

        使用角点变换再重建 bbox 的方法以保证变换后的 bbox 紧贴对象。
        """
        if len(boxes) == 0:
            return boxes

        new_boxes = []
        for box in boxes:
            x, y, bw, bh = box
            # 四个角点
            corners = np.array([
                [x, y],
                [x + bw, y],
                [x + bw, y + bh],
                [x, y + bh],
            ], dtype=np.float32)
            # 应用仿射变换
            ones = np.ones((4, 1), dtype=np.float32)
            corners_h = np.hstack([corners, ones])
            transformed = (M @ corners_h.T).T
            # 重建 bbox (min_x, min_y, max_x, max_y)
            min_x = max(0.0, transformed[:, 0].min())
            min_y = max(0.0, transformed[:, 1].min())
            max_x = min(float(w), transformed[:, 0].max())
            max_y = min(float(h), transformed[:, 1].max())
            if max_x > min_x and max_y > min_y:
                new_boxes.append([min_x, min_y, max_x - min_x, max_y - min_y])
        return new_boxes

    def _image_to_tensor(self, img):
        """PIL Image → normalized tensor [3, H, W]."""
        if img.mode != 'RGB':
            img = img.convert('RGB')
        arr = np.array(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        return (tensor - _IMAGENET_MEAN) / _IMAGENET_STD

    def __getitem__(self, idx):
        """返回一个伪视频序列 (2 帧)。

        Returns:
            dict:
                'dataset_type': 'crowdhuman'
                'seq_name': 图像 ID
                'frames': [
                    {
                        'image': Tensor [3, H, W] (normalized),
                        'targets': {'boxes': (N,4), 'labels': (N,), 'track_ids': (N,)},
                        'ignore_regions': (M,4) or None,
                        'orig_img_size': (W, H),
                    },
                    ... (2 frames)
                ]
                'num_frames': 2
        """
        sample = self.valid_samples[idx]
        img_id = sample['ID']
        gtboxes = sample['gtboxes']

        # 加载图像 — ODGT 中的 ID 不含扩展名，需自动补全
        _img_id = img_id if '.' in img_id else f"{img_id}.jpg"
        img_path = self.data_root / 'train' / 'Images' / _img_id
        if not img_path.exists():
            img_path = self.data_root / 'val' / 'Images' / _img_id
        if not img_path.exists():
            img_path = self.data_root / _img_id

        try:
            with Image.open(img_path) as img:
                img = img.convert('RGB')
                orig_w, orig_h = img.size
                img_resized = img.resize((self.img_size[1], self.img_size[0]), Image.BILINEAR)
        except Exception:
            # 返回空序列避免崩溃
            return {
                'dataset_type': 'crowdhuman',
                'seq_name': img_id,
                'frames': [],
                'num_frames': 0,
            }

        img_resized_np = np.array(img_resized)

        # 使用 TargetParser 解析标注
        valid_targets, ignore_regions = TargetParser.parse_crowdhuman(gtboxes)

        # 缩放 bbox 到 img_size 坐标空间
        scale_w = self.img_size[1] / max(orig_w, 1)
        scale_h = self.img_size[0] / max(orig_h, 1)

        def scale_boxes(targets_list):
            scaled = []
            for t in targets_list:
                x, y, w, h = t['box']
                scaled.append({
                    **t,
                    'box': [x * scale_w, y * scale_h, w * scale_w, h * scale_h],
                })
            return scaled

        def scale_regions(regions_list):
            scaled = []
            for r in regions_list:
                x, y, w, h = r['box']
                scaled.append({
                    **r,
                    'box': [x * scale_w, y * scale_h, w * scale_w, h * scale_h],
                })
            return scaled

        valid_targets = scale_boxes(valid_targets)
        ignore_regions = scale_regions(ignore_regions)

        # 生成两帧的仿射变换
        M0 = self._generate_affine_params(self.img_size[0], self.img_size[1])
        M1 = self._generate_affine_params(self.img_size[0], self.img_size[1])

        frames = []
        from scipy.ndimage import affine_transform as scipy_affine_transform

        for M in [M0, M1]:
            try:
                # 使用 scipy 仿射变换 (更稳定)
                transformed = np.zeros_like(img_resized_np, dtype=np.float32)
                for c in range(3):
                    transformed[..., c] = scipy_affine_transform(
                        img_resized_np[..., c].astype(np.float32),
                        M[:2, :2].T,  # scipy requires transposed matrix
                        offset=(M[0, 2], M[1, 2]),
                        order=1,  # bilinear interpolation
                        mode='constant',
                        cval=0.0,
                    )
            except Exception:
                # Fallback: 直接使用 resize 后的图
                transformed = img_resized_np.astype(np.float32)

            # 转 tensor 并归一化
            tensor = torch.from_numpy(transformed).permute(2, 0, 1).contiguous() / 255.0
            tensor = (tensor - _IMAGENET_MEAN) / _IMAGENET_STD

            # 变换 bbox 坐标
            transformed_targets = []
            for t in valid_targets:
                tx, ty, tw, th = t['box']
                # 对 bbox 中心应用仿射变换
                cx = tx + 0.5 * tw
                cy = ty + 0.5 * th
                p = np.array([cx, cy, 1.0])
                new_pt = M @ p
                ncw = new_pt[0] / max(self.img_size[1], 1)
                nch = new_pt[1] / max(self.img_size[0], 1)
                nw = tw * abs(M[0, 0]) / max(self.img_size[1], 1)  # 近似缩放
                nh = th * abs(M[1, 1]) / max(self.img_size[0], 1)
                # 钳制
                ncw = min(max(ncw, 0.0), 1.0)
                nch = min(max(nch, 0.0), 1.0)
                nw = min(max(nw, 1e-6), 1.0)
                nh = min(max(nh, 1e-6), 1.0)
                transformed_targets.append({
                    'box': [ncw, nch, nw, nh],
                    'track_id': t['track_id'],
                    'class_id': t.get('class_id', 0),
                })

            # 变换 ignore_regions
            transformed_ignore = []
            for r in ignore_regions:
                tx, ty, tw, th = r['box']
                cx = tx + 0.5 * tw
                cy = ty + 0.5 * th
                p = np.array([cx, cy, 1.0])
                new_pt = M @ p
                ncw = new_pt[0] / max(self.img_size[1], 1)
                nch = new_pt[1] / max(self.img_size[0], 1)
                nw = tw * abs(M[0, 0]) / max(self.img_size[1], 1)
                nh = th * abs(M[1, 1]) / max(self.img_size[0], 1)
                ncw = min(max(ncw, 0.0), 1.0)
                nch = min(max(nch, 0.0), 1.0)
                nw = min(max(nw, 1e-6), 1.0)
                nh = min(max(nh, 1e-6), 1.0)
                transformed_ignore.append({
                    'box': [ncw, nch, nw, nh],
                    'type': r.get('type', 'crowd_ignore'),
                })

            target_dict, ignore_tensor = TargetParser.to_tensors(
                transformed_targets, transformed_ignore,
                orig_img_size=(self.img_size[1], self.img_size[0]),
            )

            frames.append({
                'image': tensor,
                'targets': target_dict,
                'ignore_regions': ignore_tensor,
                'orig_img_size': (self.img_size[1], self.img_size[0]),
            })

        return {
            'dataset_type': 'crowdhuman',
            'seq_name': img_id,
            'frames': frames,
            'num_frames': 2,
        }


def create_crowdhuman_dataloader(
    data_root: str,
    annotation_path: str,
    img_size=(640, 640),
    batch_size=1,
    num_workers=8,
    max_samples=None,
):
    """创建 CrowdHuman DataLoader."""
    dataset = CrowdHumanPseudoVideoDataset(
        data_root=data_root,
        annotation_path=annotation_path,
        img_size=img_size,
        max_samples=max_samples,
    )

    def pseudo_collate(batch):
        return batch  # 保持原始格式

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=pseudo_collate,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
    )
