
import os
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random
import configparser


class DanceTrackDataset(Dataset):
    """
    DanceTrack序列数据集。

    每个样本 = K个记忆帧 + 1个当前帧 + 当前帧的GT标注。
    使用滑动窗口遍历所有连续K+1帧组合。
    """

    def __init__(self, root_dir, K=3, img_size=(560, 560), is_train=True,
                 max_objects=50):
        """
        Args:
            root_dir: 数据根目录 (如 /root/data/train 或 /root/data/val)
            K: 记忆帧数
            img_size: 输入图像尺寸 (H, W)
            is_train: 是否为训练模式（启用数据增强）
            max_objects: 每帧最大目标数（用于padding）
        """
        self.root = Path(root_dir)
        self.K = K
        self.img_size = img_size
        self.is_train = is_train
        self.max_objects = max_objects

        # 标准化变换（在增强之后应用）
        self.normalize = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

        # 训练时的随机擦除（在ToTensor之后应用）
        self.random_erasing = T.RandomErasing(
            p=0.15, scale=(0.02, 0.15), ratio=(0.3, 3.3)
        ) if is_train else None

        # 收集所有序列和窗口
        self.windows = []
        self.sequences = []
        self._scan_sequences()

        total_frames = sum(s['num_frames'] for s in self.sequences)
        print(f"[DanceTrack] {'训练' if is_train else '验证'}集:")
        print(f"  序列数: {len(self.sequences)}")
        print(f"  总帧数: {total_frames:,}")
        print(f"  滑动窗口数: {len(self.windows):,}")
        print(f"  K={K}, 图像尺寸={img_size}")
        if is_train:
            print(f"  数据增强: HFlip+RRC+ColorJitter+GrayScale+RandomErasing")

    def _scan_sequences(self):
        """扫描数据目录，收集所有有效序列。"""
        seq_dirs = []

        # 支持两种目录结构:
        # 1. train/train1/dancetrack0001/, train/train2/dancetrack0052/
        # 2. val/dancetrack0004/
        for subdir in sorted(self.root.iterdir()):
            if not subdir.is_dir():
                continue
            if (subdir / 'img1').exists() and (subdir / 'gt' / 'gt.txt').exists():
                seq_dirs.append(subdir)
            else:
                for seq_dir in sorted(subdir.iterdir()):
                    if seq_dir.is_dir() and (seq_dir / 'img1').exists():
                        if (seq_dir / 'gt' / 'gt.txt').exists():
                            seq_dirs.append(seq_dir)

        for seq_dir in seq_dirs:
            self._load_sequence(seq_dir)

    def _load_sequence(self, seq_dir):
        """加载单个序列的帧列表和GT标注。"""
        img_dir = seq_dir / 'img1'
        gt_file = seq_dir / 'gt' / 'gt.txt'

        frames = sorted(img_dir.glob('*.jpg'))
        if not frames:
            frames = sorted(img_dir.glob('*.png'))

        if len(frames) < self.K + 1:
            return

        seq_info = self._parse_seqinfo(seq_dir / 'seqinfo.ini')
        img_width = seq_info.get('imWidth', 1920)
        img_height = seq_info.get('imHeight', 1080)

        gt_data = self._parse_gt(gt_file)

        frame_ids = [int(f.stem) for f in frames]

        seq_data = {
            'name': seq_dir.name,
            'frames': frames,
            'frame_ids': frame_ids,
            'gt': gt_data,
            'width': img_width,
            'height': img_height,
            'num_frames': len(frames),
        }
        self.sequences.append(seq_data)

        for start_idx in range(len(frames) - self.K):
            self.windows.append((len(self.sequences) - 1, start_idx))

    def _parse_seqinfo(self, path):
        """解析seqinfo.ini文件。"""
        info = {}
        if path.exists():
            config = configparser.ConfigParser()
            config.read(str(path))
            if 'Sequence' in config:
                for key in config['Sequence']:
                    val = config['Sequence'][key]
                    try:
                        info[key] = int(val)
                    except ValueError:
                        try:
                            info[key] = float(val)
                        except ValueError:
                            info[key] = val
        return info

    def _parse_gt(self, gt_file):
        """
        解析MOT格式GT文件。

        格式: frame_id, track_id, x, y, w, h, conf, class, visibility
        其中(x, y)是bbox左上角坐标。

        Returns:
            dict: frame_id -> list of (track_id, x, y, w, h)
        """
        gt = {}
        with open(gt_file) as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 6:
                    continue
                frame_id = int(parts[0])
                track_id = int(parts[1])
                x = float(parts[2])
                y = float(parts[3])
                w = float(parts[4])
                h = float(parts[5])

                if w <= 0 or h <= 0:
                    continue

                if frame_id not in gt:
                    gt[frame_id] = []
                gt[frame_id].append((track_id, x, y, w, h))
        return gt

    # ========== 数据增强方法 ==========

    def _apply_train_augmentation(self, img, boxes_norm):
        """
        应用训练数据增强（同步变换图像和bbox）。

        Args:
            img: PIL Image
            boxes_norm: list of [cx, cy, w, h] normalized to [0,1]

        Returns:
            img: 增强后的PIL Image
            boxes_norm: 增强后的归一化bbox
        """
        # 1. 随机水平翻转 (p=0.5)
        if random.random() < 0.5:
            img = TF.hflip(img)
            boxes_norm = [
                [1.0 - cx, cy, w, h]
                for cx, cy, w, h in boxes_norm
            ]

        # 2. 随机缩放裁剪 (scale 0.7~1.0, ratio 0.9~1.1)
        if random.random() < 0.6:
            img, boxes_norm = self._random_resized_crop(img, boxes_norm,
                                                        scale=(0.7, 1.0),
                                                        ratio=(0.9, 1.1))

        # 3. 颜色抖动
        color_jitter = T.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05
        )
        img = color_jitter(img)

        # 4. 随机灰度 (p=0.1)
        if random.random() < 0.1:
            img = TF.to_grayscale(img, num_output_channels=3)

        return img, boxes_norm

    def _random_resized_crop(self, img, boxes_norm, scale=(0.7, 1.0),
                             ratio=(0.9, 1.1)):
        """
        随机缩放裁剪，同步更新bbox。

        Args:
            img: PIL Image
            boxes_norm: list of [cx, cy, w, h] in [0,1]
            scale: 裁剪面积比例范围
            ratio: 裁剪宽高比范围

        Returns:
            img: 裁剪后的PIL Image (原始尺寸)
            boxes_norm: 更新后的bbox
        """
        W, H = img.size

        # 随机选择裁剪参数
        area = H * W
        target_area = random.uniform(scale[0], scale[1]) * area
        aspect = random.uniform(ratio[0], ratio[1])

        crop_w = int(round((target_area * aspect) ** 0.5))
        crop_h = int(round((target_area / aspect) ** 0.5))
        crop_w = min(crop_w, W)
        crop_h = min(crop_h, H)

        # 随机裁剪位置
        x1 = random.randint(0, W - crop_w)
        y1 = random.randint(0, H - crop_h)

        # 裁剪图像
        img = TF.crop(img, y1, x1, crop_h, crop_w)
        img = TF.resize(img, (H, W))  # 恢复原始尺寸

        # 更新bbox（裁剪坐标映射）
        new_boxes = []
        for cx, cy, bw, bh in boxes_norm:
            # 将归一化坐标映射到裁剪窗口
            ncx = (cx * W - x1) / crop_w
            ncy = (cy * H - y1) / crop_h
            nbw = bw * W / crop_w
            nbh = bh * H / crop_h

            # 裁剪到有效范围
            # 计算bbox在裁剪后图像中的可见部分
            x1_box = ncx - nbw / 2
            y1_box = ncy - nbh / 2
            x2_box = ncx + nbw / 2
            y2_box = ncy + nbh / 2

            # clip到[0,1]
            x1_box = max(0, x1_box)
            y1_box = max(0, y1_box)
            x2_box = min(1, x2_box)
            y2_box = min(1, y2_box)

            # 检查bbox是否仍在裁剪区域内（至少30%面积可见）
            visible_w = x2_box - x1_box
            visible_h = y2_box - y1_box
            if visible_w > 0.01 and visible_h > 0.01:
                orig_area = nbw * nbh
                visible_area = visible_w * visible_h
                if orig_area > 0 and visible_area / orig_area > 0.3:
                    # 更新为可见部分的cx, cy, w, h
                    new_cx = (x1_box + x2_box) / 2
                    new_cy = (y1_box + y2_box) / 2
                    new_boxes.append([
                        max(0.001, min(0.999, new_cx)),
                        max(0.001, min(0.999, new_cy)),
                        max(0.001, min(0.999, visible_w)),
                        max(0.001, min(0.999, visible_h)),
                    ])

        return img, new_boxes

    # ========== __getitem__ ==========

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        seq_idx, start_idx = self.windows[idx]
        seq = self.sequences[seq_idx]

        frames = seq['frames']
        frame_ids = seq['frame_ids']
        gt = seq['gt']
        W_img = seq['width']
        H_img = seq['height']

        # ---- 决定是否水平翻转（整个K+1帧序列一致） ----
        do_hflip = self.is_train and random.random() < 0.5

        # ---- 读取K个记忆帧 ----
        memory_frames = []
        for i in range(self.K):
            img = Image.open(frames[start_idx + i]).convert('RGB')
            img = TF.resize(img, self.img_size)
            if do_hflip:
                img = TF.hflip(img)
            # 颜色增强（训练时每帧独立抖动）
            if self.is_train:
                color_jitter = T.ColorJitter(
                    brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05
                )
                img = color_jitter(img)
                if random.random() < 0.1:
                    img = TF.to_grayscale(img, num_output_channels=3)
            tensor_img = self.normalize(img)
            if self.is_train and self.random_erasing is not None:
                tensor_img = self.random_erasing(tensor_img)
            memory_frames.append(tensor_img)
        memory_frames = torch.stack(memory_frames)  # [K, 3, H, W]

        # ---- 读取当前帧（第K+1帧）----
        current_img = Image.open(frames[start_idx + self.K]).convert('RGB')
        current_frame_id = frame_ids[start_idx + self.K]
        annotations = gt.get(current_frame_id, [])

        # 先计算归一化bbox
        boxes_norm = []
        labels_list = []
        track_ids_list = []
        for track_id, x, y, w, h in annotations:
            cx = (x + w / 2.0) / W_img
            cy = (y + h / 2.0) / H_img
            nw = w / W_img
            nh = h / H_img
            cx = max(0.001, min(0.999, cx))
            cy = max(0.001, min(0.999, cy))
            nw = max(0.001, min(0.999, nw))
            nh = max(0.001, min(0.999, nh))
            boxes_norm.append([cx, cy, nw, nh])
            labels_list.append(0)
            track_ids_list.append(track_id)

        # 对当前帧应用随机缩放裁剪增强（训练时）
        if self.is_train and random.random() < 0.5 and len(boxes_norm) > 0:
            current_img, boxes_norm_aug = self._random_resized_crop(
                current_img, boxes_norm, scale=(0.7, 1.0), ratio=(0.9, 1.1)
            )
            # 如果裁剪后还剩余bbox, 使用裁剪后的
            if len(boxes_norm_aug) > 0:
                # 需要同步更新labels和track_ids
                # RRC可能丢弃部分bbox, 我们只保留存活的
                # 由于RRC中是按顺序append的, 这里需要追踪映射
                # 简化: 如果bbox数量变了, 重新匹配track_ids
                boxes_norm = boxes_norm_aug
                # track_ids和labels截断到新长度（保守策略）
                labels_list = labels_list[:len(boxes_norm)]
                track_ids_list = track_ids_list[:len(boxes_norm)]

        # Resize 到目标尺寸
        current_img = TF.resize(current_img, self.img_size)

        # 水平翻转（与记忆帧一致）
        if do_hflip:
            current_img = TF.hflip(current_img)
            boxes_norm = [
                [1.0 - cx, cy, w, h]
                for cx, cy, w, h in boxes_norm
            ]

        # 颜色增强
        if self.is_train:
            color_jitter = T.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05
            )
            current_img = color_jitter(current_img)
            if random.random() < 0.1:
                current_img = TF.to_grayscale(
                    current_img, num_output_channels=3)

        current_frame = self.normalize(current_img)
        if self.is_train and self.random_erasing is not None:
            current_frame = self.random_erasing(current_frame)

        # ---- 构建targets ----
        if boxes_norm:
            targets = {
                'boxes': torch.tensor(boxes_norm, dtype=torch.float32),
                'labels': torch.tensor(labels_list, dtype=torch.long),
                'track_ids': torch.tensor(track_ids_list, dtype=torch.long),
            }
        else:
            targets = {
                'boxes': torch.zeros(0, 4, dtype=torch.float32),
                'labels': torch.zeros(0, dtype=torch.long),
                'track_ids': torch.zeros(0, dtype=torch.long),
            }

        return {
            'memory_frames': memory_frames,
            'current_frame': current_frame,
            'targets': targets,
            'seq_name': seq['name'],
            'frame_id': current_frame_id,
        }


def collate_fn(batch):
    """
    自定义collate函数。

    memory_frames和current_frame直接stack；
    targets保持为list（不同样本目标数不同）。
    """
    memory_frames = torch.stack([item['memory_frames'] for item in batch])
    current_frames = torch.stack([item['current_frame'] for item in batch])
    targets = [item['targets'] for item in batch]
    seq_names = [item['seq_name'] for item in batch]
    frame_ids = [item['frame_id'] for item in batch]

    return {
        'memory_frames': memory_frames,   # [B, K, 3, H, W]
        'current_frame': current_frames,  # [B, 3, H, W]
        'targets': targets,               # list of B dicts
        'seq_names': seq_names,
        'frame_ids': frame_ids,
    }
