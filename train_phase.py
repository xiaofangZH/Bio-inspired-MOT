#!/usr/bin/env python3
"""
HAMT 三阶段训练脚本 (新方案)

Phase 1 (CrowdHuman, 15 epochs): 纯检测筑基 — 冻结backbone前5epoch, 解冻最后2层后10epoch
Phase 2 (MOT17+MOT20, 20 epochs): 线性运动关联 — 全解冻 + MemoryBank + ReID
Phase 3 (DanceTrack, 20 epochs): 极难域微调 — 帧间隔采样 + Scheduled Sampling + EMA
"""

import argparse
import logging
import os
import time
import gc
from datetime import datetime
from pathlib import Path

import torch
import torch.optim as optim
import torch.utils.data
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter

# 环境变量：缓解 CUDA 内存碎片化
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

# cuDNN 自动搜索最优卷积算法（固定输入尺寸下加速显著）
torch.backends.cudnn.benchmark = True
# 允许 TF32 加速（RTX 5090 Blackwell 架构支持，精度损失可忽略）
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ─── 项目内导入 ───
PROJECT_ROOT = Path(__file__).resolve().parent

from data_loader import load_frame_tensor, create_phase_dataloader
from hmat.modeling.hmat_model import HMAT
from hmat.modeling.loss.matcher import HungarianMatcher
from hmat.modeling.loss.set_criterion import SetCriterion
from hmat.data.target_parser import TargetParser


def setup_logger(log_dir: str) -> logging.Logger:
    """创建训练日志器"""
    log_path = Path(log_dir) / 'logs'
    log_path.mkdir(parents=True, exist_ok=True)

    log_file = log_path / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

    logger = logging.getLogger('HAMT_Phase')
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


class PhaseConfig:
    """阶段配置容器 — 从 YAML 加载并支持命令行覆盖"""

    def __init__(self, phase_yaml: dict, phase_name: str):
        self.phase_name = phase_name
        cfg = phase_yaml

        self.epochs = cfg.get('epochs', 15)
        self.phase_type = cfg.get('phase_type', 'crowdhuman')
        self.img_size = tuple(cfg.get('img_size', [640, 640]))
        self.batch_size = cfg.get('batch_size', 1)
        self.gradient_accumulation = cfg.get('gradient_accumulation', 32)
        self.num_workers = cfg.get('num_workers', 8)
        self.use_amp = cfg.get('use_amp', True)
        self.grad_clip_norm = cfg.get('grad_clip_norm', 1.0)

        # 学习率
        lr_cfg = cfg.get('lr', {})
        self.backbone_lr = lr_cfg.get('backbone_lr', 2e-4)
        self.head_lr = lr_cfg.get('head_lr', 2e-4)
        self.min_lr = lr_cfg.get('min_lr', 1e-6)
        self.warmup_steps = lr_cfg.get('warmup_steps', 500)
        self.warmup_start_factor = lr_cfg.get('warmup_start_factor', 0.1)
        self.weight_decay = lr_cfg.get('weight_decay', 1e-4)

        # Loss 权重
        loss_cfg = cfg.get('loss', {})
        self.cls_weight = loss_cfg.get('cls_weight', 0.5)
        self.l1_weight = loss_cfg.get('l1_weight', 5.0)
        self.ciou_weight = loss_cfg.get('ciou_weight', 2.0)
        self.reid_weight = loss_cfg.get('reid_weight', 0.0)  # Phase 1 = 0, Phase 2/3 = 1.0
        self.reid_warmup_epochs = loss_cfg.get('reid_warmup_epochs', 0)  # ReID 预热轮数
        self.losses = loss_cfg.get('losses', ['labels', 'boxes'])

        # 主干解冻策略
        freeze_cfg = cfg.get('freeze', {})
        self.freeze_backbone_epochs = freeze_cfg.get('freeze_epochs', 0)
        self.unfreeze_last_n = freeze_cfg.get('unfreeze_last_n', 2)

        # Memory Bank — Phase 2/3 使用 BatchMemoryBank
        mem_cfg = cfg.get('memory', {})
        self.use_memory_bank = mem_cfg.get('enabled', False)
        self.teacher_forcing = mem_cfg.get('teacher_forcing', False)

        # Scheduled Sampling
        ss_cfg = cfg.get('scheduled_sampling', {})
        self.scheduled_sampling = ss_cfg.get('enabled', False)
        self.ss_start_prob = ss_cfg.get('start_prob', 1.0)
        self.ss_end_prob = ss_cfg.get('end_prob', 0.3)

        # EMA
        ema_cfg = cfg.get('ema', {})
        self.use_ema = ema_cfg.get('enabled', False)
        self.ema_decay = ema_cfg.get('decay', 0.999)

        # Checkpoint
        self.save_interval = cfg.get('save_interval', 5)
        self.num_queries = cfg.get('num_queries', 300)
        self.hidden_dim = cfg.get('hidden_dim', 256)
        self.num_classes = cfg.get('num_classes', 1)

        # Scheduled Sampling decay
        self.ss_prob_decay = (self.ss_start_prob - self.ss_end_prob) / max(self.epochs, 1)

        # Backbone weights (可跨阶段传递)
        self.backbone_weights = cfg.get('backbone_weights', None)


class EMAModel:
    """指数移动平均模型包装器"""

    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self._register()

    def _register(self):
        for name, param in self.model.named_parameters():
            self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if name in self.shadow:
                new_val = (self.decay * self.shadow[name].data +
                           (1.0 - self.decay) * param.data)
                self.shadow[name].data = new_val

    def apply_shadow(self):
        """将 EMA 权重应用到模型 (验证前使用)"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name].data)

    def restore(self):
        """恢复原始权重 (验证后使用)"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup.clear()


# ─── 目标张量构建辅助 ───

def _build_targets_from_parsed(valid_targets, ignore_regions,
                                orig_img_size, device):
    """将 TargetParser 输出转为训练用张量。

    valid_targets: 可能来自 parse_mot/parse_dancetrack (原始 annotation dict)
                  或 CrowdHuman 已处理过的 targets。
    ignore_regions: list of {'box': [x,y,w,h], 'type': str}
    orig_img_size: (W, H)
    device: torch device

    Returns:
        target_dict: {'boxes': (N,4) cxcywh normalized, 'labels': (N,), 'track_ids': (N,)}
        ignore_tensor: (M,4) cxcywh normalized, or None
    """
    W, H = orig_img_size

    boxes = []
    labels = []
    track_ids = []

    for t in valid_targets:
        # MOTDataset 已提供 bbox_norm (cx,cy,w,h 归一化)
        if 'bbox_norm' in t:
            cx, cy, nw, nh = t['bbox_norm']
            boxes.append([cx, cy, nw, nh])
        elif 'box' in t:
            # CrowdHuman 已处理格式: box=[x,y,w,h] 绝对坐标
            x, y, w, h = t['box']
            cx = (x + 0.5 * w) / max(W, 1)
            cy = (y + 0.5 * h) / max(H, 1)
            nw = w / max(W, 1)
            nh = h / max(H, 1)
            boxes.append([cx, cy, nw, nh])
        else:
            continue

        labels.append(t.get('class_id', t.get('class', 1)) - 1)
        labels[-1] = min(max(labels[-1], 0), 0)  # clamp to valid class
        track_ids.append(t.get('track_id', -1))

    if boxes:
        boxes_t = torch.tensor(boxes, dtype=torch.float32, device=device)
        labels_t = torch.tensor(labels, dtype=torch.long, device=device)
        track_ids_t = torch.tensor(track_ids, dtype=torch.long, device=device)
    else:
        boxes_t = torch.zeros((0, 4), dtype=torch.float32, device=device)
        labels_t = torch.zeros((0,), dtype=torch.long, device=device)
        track_ids_t = torch.zeros((0,), dtype=torch.long, device=device)

    ignore_tensor = None
    if ignore_regions:
        ignore_boxes = []
        for r in ignore_regions:
            x, y, w, h = r['box']
            cx = (x + 0.5 * w) / max(W, 1)
            cy = (y + 0.5 * h) / max(H, 1)
            nw = w / max(W, 1)
            nh = h / max(H, 1)
            ignore_boxes.append([cx, cy, nw, nh])
        ignore_tensor = torch.tensor(ignore_boxes, dtype=torch.float32, device=device)

    target_dict = {
        'boxes': boxes_t,
        'labels': labels_t,
        'track_ids': track_ids_t,
    }
    return target_dict, ignore_tensor


def _infer_dataset_name(seq_dir: str) -> str:
    """从序列目录路径推断数据集名称."""
    seq_path = Path(seq_dir)
    for candidate in ['MOT17', 'MOT20', 'dancetrack', 'DanceTrack']:
        if candidate in seq_path.parts or candidate.lower() in str(seq_path).lower():
            return candidate.lower()
    return 'mot'


class PhaseTrainer:
    """阶段训练器 — 单一阶段内执行训练循环"""

    def __init__(self, config: PhaseConfig, logger: logging.Logger,
                 output_dir: Path, gpu_id: int = 0):
        self.config = config
        self.logger = logger
        self.output_dir = output_dir
        self.device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
        self.global_step = 0

        # 模型
        self.model = None
        self.criterion = None
        self.matcher = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.ema = None

        # 指标
        self.history = []
        self.writer = None

    def build_model(self, load_weights_path=None):
        """构建 HMAT 模型及训练组件"""
        self.logger.info("构建 HMAT 模型...")

        # Phase 1: 关闭记忆库 (use_batch_memory=False → MemoryBank，不做跨帧关联)
        # Phase 2/3: 开启记忆库 (use_batch_memory=True → BatchMemoryBank)
        use_batch_memory = self.config.use_memory_bank

        self.model = HMAT(
            num_classes=self.config.num_classes,
            hidden_dim=self.config.hidden_dim,
            num_queries=self.config.num_queries,
            use_batch_memory=use_batch_memory,
        ).to(self.device)

        # 加载预训练权重
        if load_weights_path:
            self.logger.info(f"加载权重: {load_weights_path}")
            checkpoint = torch.load(load_weights_path, map_location='cpu', weights_only=False)
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint

            # 过滤不匹配的键
            model_dict = self.model.state_dict()
            filtered_dict = {}
            unmatched_from_ckpt = []
            for k, v in state_dict.items():
                key = k.replace('module.', '')
                if key in model_dict and model_dict[key].shape == v.shape:
                    filtered_dict[key] = v
                else:
                    unmatched_from_ckpt.append(key)

            if unmatched_from_ckpt:
                self.logger.warning(
                    f"未加载的键: {len(unmatched_from_ckpt)}, "
                    f"e.g. {unmatched_from_ckpt[:5]}"
                )

            missing, unexpected = self.model.load_state_dict(filtered_dict, strict=False)
            if missing:
                self.logger.info(f"新模块随机初始化: {len(missing)} keys")
            if unexpected:
                self.logger.info(f"忽略多余 key: {len(unexpected)} keys")

        # 损失函数
        self.matcher = HungarianMatcher(
            cost_class=self.config.cls_weight,
            cost_bbox=self.config.l1_weight,
            cost_ciou=self.config.ciou_weight,
        )

        weight_dict = {
            'loss_ce': self.config.cls_weight,
            'loss_bbox': self.config.l1_weight,
            'loss_ciou': self.config.ciou_weight,
            'loss_reid': self.config.reid_weight,
        }

        self.criterion = SetCriterion(
            num_classes=self.config.num_classes,
            matcher=self.matcher,
            weight_dict=weight_dict,
            eos_coef=0.1,
            losses=self.config.losses,
        ).to(self.device)

        # 优化器 (分层学习率)
        backbone_lr_scale = self.config.backbone_lr / max(self.config.head_lr, 1e-8)
        param_groups = self.model.get_parameter_groups(
            base_lr=self.config.head_lr,
            backbone_lr_scale=backbone_lr_scale,
        )
        self.optimizer = optim.AdamW(
            param_groups,
            weight_decay=self.config.weight_decay,
        )
        # 记录初始 lr 用于 warmup 恢复
        for pg in self.optimizer.param_groups:
            pg['init_lr'] = pg['lr']

        # 调度器 (Cosine Annealing)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.config.epochs, eta_min=self.config.min_lr,
        )

        # AMP
        self.scaler = GradScaler() if self.config.use_amp else None

        # EMA
        if self.config.use_ema:
            self.ema = EMAModel(self.model, decay=self.config.ema_decay)
            self.logger.info(f"EMA 已启用 (decay={self.config.ema_decay})")

        # TensorBoard
        self.writer = SummaryWriter(log_dir=str(self.output_dir / 'tensorboard'))

        # 统计参数量
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"模型参数: {total/1e6:.1f}M total, {trainable/1e6:.1f}M trainable")

    def apply_freeze_strategy(self, epoch: int):
        """根据 epoch 和配置应用冻结/解冻策略"""
        backbone = self.model.backbone

        if epoch < self.config.freeze_backbone_epochs:
            # 冻结 backbone
            if hasattr(backbone, 'freeze_all_backbone'):
                backbone.freeze_all_backbone()
                status = backbone.get_frozen_status()
                self.logger.info(
                    f"Backbone 冻结: {status['frozen']}/{status['total']} params "
                    f"({status['ratio']:.1%})"
                )
        else:
            # 渐进解冻最后 N 层
            if hasattr(backbone, 'unfreeze_last_n_blocks'):
                backbone.unfreeze_last_n_blocks(n=self.config.unfreeze_last_n)
                status = backbone.get_frozen_status()
                self.logger.info(
                    f"Backbone 解冻最后{self.config.unfreeze_last_n}层: "
                    f"frozen={status['frozen']}, total={status['total']}"
                )

    def _extract_model_outputs(self, outputs):
        """统一处理 model 输出 (HMAT.forward 返回 list[dict])."""
        if isinstance(outputs, list) and outputs:
            return outputs[0]
        return outputs

    def process_crowdhuman_batch(self, batch_data):
        """Phase 1: 处理 CrowdHuman 伪视频 batch — 每帧独立训练，无需记忆库"""
        # batch_data is a list of sample dicts from CrowdHumanPseudoVideoDataset
        if not batch_data or not isinstance(batch_data, list):
            return None, {}

        total_loss = None
        all_metrics = []
        total_frames = 0

        for sample in batch_data:
            frames = sample.get('frames', [])
            if not frames:
                continue

            # Phase 1 无记忆库: 每帧独立，不跨帧共享 track memory
            for frame in frames:
                image = frame['image'].unsqueeze(0).to(self.device)  # [1, 3, H, W]
                targets = [{k: v.to(self.device) for k, v in frame['targets'].items()}]
                ignore_regions = frame.get('ignore_regions')
                if ignore_regions is not None:
                    ignore_regions = ignore_regions.to(self.device)
                ignore_list = [ignore_regions] if ignore_regions is not None else None

                # 前向传播 — HMAT 返回 [outputs]，提取 dict
                raw_outputs = self.model(image, targets=targets)
                outputs = self._extract_model_outputs(raw_outputs)

                # 损失计算 — SetCriterion.forward 支持 ignore_regions 参数
                loss_dict, indices = self.criterion(outputs, targets, ignore_regions=ignore_list)

                weight_dict = self.criterion.weight_dict
                frame_loss = sum(loss_dict[k] * weight_dict.get(k, 1.0)
                                 for k in loss_dict.keys())

                if total_loss is None:
                    total_loss = frame_loss
                else:
                    total_loss = total_loss + frame_loss

                total_frames += 1
                all_metrics.append(self._compute_metrics(outputs, targets, indices, loss_dict))

        if total_loss is not None and total_frames > 0:
            total_loss = total_loss / total_frames

        # 合并指标
        merged = {}
        if all_metrics:
            for k in all_metrics[0]:
                merged[k] = sum(m.get(k, 0) for m in all_metrics) / len(all_metrics)

        return total_loss, merged

    def _process_single_frame(self, seq_item, frame_idx, img_path, img_size,
                             orig_img_size, dataset_name, teacher_force_prob=1.0):
        """处理单帧：加载 → forward → 计算 loss。

        Args:
            seq_item: MOTDataset 序列项
            frame_idx: 帧在序列中的索引 (0-based)
            img_path, img_size: 帧路径和尺寸
            orig_img_size: (W,H) 原始尺寸
            dataset_name: 'mot17','mot20','dancetrack'
            teacher_force_prob: Scheduled Sampling 概率 (1.0=全量TF)

        Returns:
            loss: 标量 loss tensor
            metrics: dict
        """
        frame_id = frame_idx + 1

        # 加载帧
        img_tensor = load_frame_tensor(img_path, img_size)
        img_tensor = img_tensor.unsqueeze(0).to(self.device)  # [1, 3, H, W]

        # 获取标注
        annot_dict = seq_item.get('annotations', {})
        raw_anns = annot_dict.get(frame_id, [])

        # 解析 targets
        if dataset_name in ('mot17', 'mot20'):
            valid_targets, ignore_regions = TargetParser.parse_mot(
                raw_anns, orig_img_size=orig_img_size)
        else:
            valid_targets, ignore_regions = TargetParser.parse_dancetrack(raw_anns)

        target_dict, ignore_tensor = _build_targets_from_parsed(
            valid_targets, ignore_regions,
            orig_img_size=orig_img_size,
            device=self.device,
        )
        targets = [target_dict]
        ignore_list = [ignore_tensor] if ignore_tensor is not None else None

        # 前向传播 (Scheduled Sampling)
        raw_outputs = self.model(img_tensor, targets=targets,
                                 teacher_force_prob=teacher_force_prob)
        outputs = self._extract_model_outputs(raw_outputs)

        # 损失计算
        loss_dict, indices = self.criterion(outputs, targets, ignore_regions=ignore_list)

        weight_dict = self.criterion.weight_dict
        loss = sum(loss_dict[k] * weight_dict.get(k, 1.0) for k in loss_dict.keys())

        metrics = self._compute_metrics(outputs, targets, indices, loss_dict)

        return loss, metrics

    def _optimizer_step(self):
        """优化器步进：梯度裁剪 + step + warmup + EMA"""
        # 梯度裁剪
        if self.config.grad_clip_norm > 0:
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.grad_clip_norm)

        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        self.optimizer.zero_grad()

        # 参数级 warmup
        if self.global_step < self.config.warmup_steps:
            progress = self.global_step / max(self.config.warmup_steps, 1)
            warmup_factor = (
                self.config.warmup_start_factor +
                (1.0 - self.config.warmup_start_factor) * progress
            )
            for pg in self.optimizer.param_groups:
                pg['lr'] = pg['init_lr'] * warmup_factor

        # EMA 更新
        if self.ema is not None:
            self.ema.update()

    def _compute_metrics(self, outputs, targets, indices, loss_dict):
        """计算训练指标"""
        metrics = {}

        for k, v in loss_dict.items():
            if isinstance(v, torch.Tensor) and v.numel() > 0:
                metrics[f'loss_{k}'] = v.item()

        # 基础检测统计
        gt_count = sum(t['boxes'].shape[0] for t in targets)
        pred_logits = outputs.get('pred_logits')
        if pred_logits is not None:
            if pred_logits.shape[-1] == 1:
                obj_score = pred_logits.sigmoid()[..., 0]
            else:
                bg_idx = self.config.num_classes
                obj_score = pred_logits.softmax(dim=-1)[..., :bg_idx].max(dim=-1)[0]
            pred_pos = int((obj_score > 0.5).sum().item())
            metrics['gt_count'] = gt_count
            metrics['pred_pos'] = pred_pos

        return metrics

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存检查点 — 流式 offload 到 CPU 避免 GPU OOM"""
        gc.collect()
        torch.cuda.empty_cache()

        # 逐参数 offload
        model_sd = {}
        for name, param in self.model.named_parameters():
            model_sd[name] = param.data.cpu()
        for name, buf in self.model.named_buffers():
            model_sd[name] = buf.data.cpu()

        checkpoint = {
            'epoch': epoch,
            'phase': self.config.phase_name,
            'global_step': self.global_step,
            'model_state_dict': model_sd,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': {k: v for k, v in self.config.__dict__.items()
                       if not k.startswith('_')},
            'history': self.history,
        }

        if self.ema is not None:
            checkpoint['ema_shadow'] = {
                k: v.cpu() for k, v in self.ema.shadow.items()
            }

        # 常规 checkpoint
        ckpt_path = self.output_dir / f'epoch_{epoch:03d}.pth'
        torch.save(checkpoint, ckpt_path)
        self.logger.info(f"Checkpoint saved: {ckpt_path}")

        # 最佳模型
        if is_best:
            best_path = self.output_dir / f'{self.config.phase_name}_best.pth'
            torch.save(checkpoint, best_path)
            self.logger.info(f"Best model saved: {best_path}")

        # 清理
        del model_sd, checkpoint
        gc.collect()
        torch.cuda.empty_cache()

    def resume_from_checkpoint(self, checkpoint_path: str):
        """从检查点恢复训练状态。

        Args:
            checkpoint_path: .pth 文件路径

        Returns:
            int: 已完成的最大 epoch 数 (下一次从该值+1开始)
        """
        self.logger.info(f"从检查点恢复: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        # 验证阶段一致性
        ckpt_phase = checkpoint.get('phase', '')
        if ckpt_phase != self.config.phase_name:
            self.logger.warning(
                f"检查点阶段 ({ckpt_phase}) 与当前阶段 ({self.config.phase_name}) 不匹配, "
                f"将仅加载模型权重"
            )

        # 恢复模型权重
        state_dict = checkpoint.get('model_state_dict', {})
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            self.logger.info(f"恢复后缺失 keys: {len(missing)}")
        if unexpected:
            self.logger.info(f"恢复后多余 keys: {len(unexpected)}")

        # 恢复优化器状态
        if 'optimizer_state_dict' in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except Exception as e:
                self.logger.warning(f"优化器状态恢复失败: {e}, 从头开始")

        # 恢复调度器状态
        if 'scheduler_state_dict' in checkpoint:
            try:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            except Exception as e:
                self.logger.warning(f"调度器状态恢复失败: {e}")

        # 恢复全局步数
        self.global_step = checkpoint.get('global_step', 0)

        # 恢复历史记录
        self.history = checkpoint.get('history', [])

        # 恢复 EMA (如果存在且当前启用了 EMA)
        if 'ema_shadow' in checkpoint and self.ema is not None:
            for k, v in checkpoint['ema_shadow'].items():
                if k in self.ema.shadow:
                    self.ema.shadow[k] = v.to(self.device)
            self.logger.info("EMA 状态已恢复")

        start_epoch = checkpoint.get('epoch', 0)
        self.logger.info(
            f"恢复完成 | epoch={start_epoch}, global_step={self.global_step}, "
            f"history={len(self.history)}条"
        )

        del checkpoint
        gc.collect()
        torch.cuda.empty_cache()

        return start_epoch

    def train(self, dataloader: torch.utils.data.DataLoader, resume_epoch: int = 0):
        """执行单阶段训练循环

        Args:
            dataloader: 数据加载器
            resume_epoch: 从该 epoch 开始训练 (0 表示从头开始, N 表示已完成 1..N epoch)
        """
        phase_type = self.config.phase_type
        self.logger.info(
            f"开始训练 Phase: {self.config.phase_name} ({self.config.epochs} epochs)"
        )

        if resume_epoch > 0:
            self.logger.info(f"↻ 从 Epoch {resume_epoch + 1} 恢复训练 (已完成 {resume_epoch}/{self.config.epochs})")
            # 手动推进 scheduler 到对应 epoch (因为 step() 在每 epoch 末尾调用)
            for _ in range(resume_epoch):
                self.scheduler.step()
            # 已完成 epoch 的冻结策略无需重复执行
            if self.config.freeze_backbone_epochs > 0 and resume_epoch > self.config.freeze_backbone_epochs:
                self.apply_freeze_strategy(resume_epoch)

        best_loss = float('inf')
        # 恢复 best_loss (从 history 中取最小值)
        if self.history:
            best_loss = min(h.get('train_loss', float('inf')) for h in self.history)

        epoch = resume_epoch
        while epoch < self.config.epochs:
            # ── 热重载 epochs: 检查是否有人工放大了 epoch 数 ──
            hot_file = self.output_dir / 'hot_epochs.txt'
            if hot_file.exists():
                try:
                    new_epochs = int(hot_file.read_text().strip())
                    if new_epochs > self.config.epochs:
                        self.logger.info(
                            f"⚡ Hot-reload: epochs {self.config.epochs} → {new_epochs}"
                        )
                        self.config.epochs = new_epochs
                except Exception:
                    pass

            epoch_start = time.time()
            self.model.train()

            # 应用冻结策略 (仅 Phase 1 需要)
            if self.config.freeze_backbone_epochs > 0:
                if epoch in [0, self.config.freeze_backbone_epochs]:
                    self.apply_freeze_strategy(epoch)

            # Scheduled Sampling 概率衰减 (仅 Phase 3)
            ss_prob = 1.0
            if self.config.scheduled_sampling:
                ss_prob = max(
                    self.config.ss_end_prob,
                    self.config.ss_start_prob - epoch * self.config.ss_prob_decay
                )
                if epoch == resume_epoch:
                    self.logger.info(
                        f"Scheduled Sampling: {ss_prob:.3f} "
                        f"({ss_prob*100:.0f}% Teacher Forcing)"
                    )

            # ReID Loss 渐进预热 (Phase 2 + Phase 3 前 N 轮)
            reid_factor = 1.0
            if self.config.reid_warmup_epochs > 0 and self.config.reid_weight > 0:
                if epoch < self.config.reid_warmup_epochs:
                    # 线性预热: 从 0.1 到 1.0
                    reid_factor = 0.1 + 0.9 * (epoch / max(self.config.reid_warmup_epochs - 1, 1))
                self.criterion.weight_dict['loss_reid'] = (
                    self.config.reid_weight * reid_factor
                )
                if epoch == resume_epoch or epoch % 5 == 0:
                    self.logger.info(
                        f"ReID warmup: epoch {epoch+1}/{self.config.reid_warmup_epochs} "
                        f"→ factor={reid_factor:.2f}, "
                        f"weight={self.criterion.weight_dict['loss_reid']:.3f}"
                    )

            epoch_loss = 0.0
            epoch_steps = 0
            nan_step_count = 0

            self.optimizer.zero_grad()
            accum_step = 0

            for batch_idx, batch in enumerate(dataloader):
                # ─── Phase 1: CrowdHuman 逐帧处理 ───
                if phase_type == 'crowdhuman' or phase_type == 'phase1':
                    loss, metrics = self.process_crowdhuman_batch(batch)
                    if loss is None or not torch.isfinite(loss):
                        nan_step_count += 1
                        continue
                    loss = loss / self.config.gradient_accumulation
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    epoch_loss += loss.item() * self.config.gradient_accumulation
                    epoch_steps += 1
                    accum_step += 1
                    if accum_step % self.config.gradient_accumulation == 0:
                        self._optimizer_step()
                        self.global_step += 1
                    if batch_idx % 50 == 0:
                        lr = self.optimizer.param_groups[0]['lr']
                        loss_ciou = metrics.get('loss_loss_ciou', 0)
                        self.logger.info(
                            f"Epoch {epoch+1}, Step {batch_idx}, "
                            f"AvgLoss: {epoch_loss/max(epoch_steps,1):.6f}, "
                            f"LR: {lr:.2e}, CIoU: {loss_ciou:.4f}"
                        )

                # ─── Phase 2/3: MOT/DanceTrack 逐帧处理 ───
                else:
                    for seq_item in batch:
                        img_paths = seq_item['img_paths']
                        img_size = seq_item['img_size']
                        orig_img_size = seq_item.get('orig_img_size', (1920, 1080))
                        seq_dir = seq_item.get('seq_dir', '')
                        dataset_name = _infer_dataset_name(seq_dir)

                        # 重置记忆库 — 每序列独立
                        if self.config.use_memory_bank and hasattr(self.model, 'memory_bank'):
                            self.model.memory_bank.reset()

                        for frame_idx, img_path in enumerate(img_paths):
                            loss, metrics = self._process_single_frame(
                                seq_item, frame_idx, img_path, img_size,
                                orig_img_size, dataset_name,
                                teacher_force_prob=ss_prob)

                            if loss is None or not torch.isfinite(loss):
                                nan_step_count += 1
                                continue

                            # 梯度累积 — 逐帧 backward
                            loss = loss / self.config.gradient_accumulation
                            if self.scaler is not None:
                                self.scaler.scale(loss).backward()
                            else:
                                loss.backward()

                            epoch_loss += loss.item() * self.config.gradient_accumulation
                            epoch_steps += 1
                            accum_step += 1

                            if accum_step % self.config.gradient_accumulation == 0:
                                self._optimizer_step()
                                self.global_step += 1

                        # 每序列结束打一次日志
                        lr = self.optimizer.param_groups[0]['lr']
                        self.logger.info(
                            f"Epoch {epoch+1}, Seq {dataset_name}/{seq_item.get('seq_name','?')}, "
                            f"Steps {epoch_steps}, AvgLoss: {epoch_loss/max(epoch_steps,1):.6f}, "
                            f"LR: {lr:.2e}"
                        )

            # ─── 处理末尾的剩余梯度累积 ───
            if accum_step % self.config.gradient_accumulation != 0:
                if self.config.grad_clip_norm > 0 and self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                if self.config.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.grad_clip_norm)
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()

            if epoch_steps == 0:
                self.logger.warning(f"Epoch {epoch+1} 没有产生任何有效优化步, 跳过")
                self.scheduler.step()
                epoch += 1
                continue

            # Epoch 后统计
            avg_loss = epoch_loss / max(epoch_steps, 1)
            self.scheduler.step()

            epoch_time = time.time() - epoch_start
            nan_rate = nan_step_count / max(epoch_steps + nan_step_count, 1)

            self.logger.info(
                f"Epoch {epoch+1} 完成 | loss={avg_loss:.6f} | "
                f"steps={epoch_steps} NaN={nan_step_count}({nan_rate:.1%}) | "
                f"time={epoch_time:.1f}s | global_step={self.global_step}"
            )

            # TensorBoard
            if self.writer:
                self.writer.add_scalar('loss/epoch', avg_loss, epoch + 1)
                self.writer.add_scalar('lr', self.optimizer.param_groups[0]['lr'], epoch + 1)

            # 历史记录
            self.history.append({
                'epoch': epoch + 1,
                'train_loss': avg_loss,
                'lr': self.optimizer.param_groups[0]['lr'],
                'epoch_time': epoch_time,
            })

            # 最佳模型
            is_best = avg_loss < best_loss
            if is_best:
                best_loss = avg_loss

            # 保存 checkpoint
            if (epoch + 1) % self.config.save_interval == 0 or is_best:
                self.save_checkpoint(epoch + 1, is_best=is_best)

            # 显存碎片清理
            gc.collect()
            torch.cuda.empty_cache()

            epoch += 1

        self.logger.info(
            f"Phase {self.config.phase_name} 训练完成, best_loss={best_loss:.6f}"
        )


def _find_latest_checkpoint(phase_dir: Path) -> Path | None:
    """在阶段目录中查找最新的 epoch checkpoint。

    Returns:
        最新 checkpoint 路径，或 None
    """
    if not phase_dir.exists():
        return None
    checkpoints = sorted(
        phase_dir.glob('epoch_*.pth'),
        key=lambda p: int(p.stem.split('_')[-1])
    )
    return checkpoints[-1] if checkpoints else None


def _is_phase_complete(phase_dir: Path, total_epochs: int) -> bool:
    """检查阶段是否已完成所有 epoch。

    判断依据（按优先级）:
    1. epoch_{total_epochs:03d}.pth 存在 → 精确完成
    2. _best.pth 存在 且 无任何 epoch checkpoint（用户手动清理）→ 视为已完成
    3. _best.pth 存在 且 最后 epoch checkpoint 接近 total（gap ≤ 10）
    """
    if not phase_dir.exists():
        return False
    if (phase_dir / f'epoch_{total_epochs:03d}.pth').exists():
        return True
    best = phase_dir / f'{phase_dir.name}_best.pth'
    if not best.exists():
        return False
    existing = sorted(phase_dir.glob('epoch_*.pth'))
    # 无 epoch checkpoint + 有 best → 用户只保留了最终权重
    if len(existing) == 0:
        return True
    # 有 epoch checkpoint 但缺最后一个 → 检查 gap
    try:
        last_epoch = int(existing[-1].stem.split('_')[-1])
    except ValueError:
        last_epoch = 0
    return last_epoch >= total_epochs - 10


def train_all_phases(yaml_path: str, gpu_id: int = 0,
                     output_dir: str = None, resume_dir: str = None):
    """主入口: 加载 YAML 配置并串行执行 Phase 1 → Phase 2 → Phase 3

    支持断点恢复:
    - 若指定 --resume, 则从已有目录恢复训练
    - 否则创建新的时间戳输出目录
    - 已完成的部分阶段/epoch 自动跳过
    """
    import yaml

    with open(yaml_path, 'r', encoding='utf-8') as f:
        full_cfg = yaml.safe_load(f)

    # 确定输出目录
    if resume_dir:
        output_base = Path(resume_dir)
        if not output_base.exists():
            raise FileNotFoundError(f"恢复目录不存在: {resume_dir}")
    elif output_dir:
        output_base = Path(output_dir)
    else:
        output_base = Path(full_cfg.get('output_dir', 'results')) / \
                      f"hamt_deep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_base.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(str(output_base))
    logger.info(f"输出目录: {output_base}")
    logger.info(f"配置文件: {yaml_path}")
    logger.info(f"GPU: {gpu_id}")
    if resume_dir:
        logger.info(f"模式: 断点恢复 ✓")

    phases = full_cfg.get('phases', {})
    phase_order = ['phase1', 'phase2', 'phase3']

    prev_weights = None

    for phase_name in phase_order:
        if phase_name not in phases:
            logger.info(f"跳过 {phase_name} (未在配置中找到)")
            continue

        phase_cfg = phases[phase_name]
        if not phase_cfg.get('enabled', True):
            logger.info(f"跳过 {phase_name} (未启用)")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"开始 {phase_name}")
        logger.info(f"{'='*60}")

        config = PhaseConfig(phase_cfg, phase_name)

        # 创建阶段输出目录
        phase_dir = output_base / phase_name
        phase_dir.mkdir(exist_ok=True)

        # ─── 检查是否已完成 ───
        if _is_phase_complete(phase_dir, config.epochs):
            logger.info(f"✓ {phase_name} 已完成 ({config.epochs}/{config.epochs} epochs), 跳过训练")
            # 为下一阶段提供权重
            best_path = phase_dir / f'{phase_name}_best.pth'
            if best_path.exists():
                prev_weights = str(best_path)
            else:
                latest = _find_latest_checkpoint(phase_dir)
                if latest:
                    prev_weights = str(latest)
                    logger.info(f"使用最新检查点作为下一阶段权重: {latest.name}")
            continue

        trainer = PhaseTrainer(config, logger, phase_dir, gpu_id=gpu_id)
        trainer.build_model(load_weights_path=prev_weights)

        # ─── 检查是否有未完成的 checkpoint ───
        latest_ckpt = _find_latest_checkpoint(phase_dir)
        resume_epoch = 0
        if latest_ckpt:
            logger.info(f"发现已有 checkpoint: {latest_ckpt.name}")
            resume_epoch = trainer.resume_from_checkpoint(str(latest_ckpt))
            if resume_epoch >= config.epochs:
                logger.info(f"✓ {phase_name} 已完成 (checkpoint epoch={resume_epoch}), 跳过训练")
                del trainer
                gc.collect()
                torch.cuda.empty_cache()
                best_path = phase_dir / f'{phase_name}_best.pth'
                if best_path.exists():
                    prev_weights = str(best_path)
                else:
                    prev_weights = str(latest_ckpt)
                continue

        # 创建 DataLoader
        dataloader = create_phase_dataloader(
            phase=config.phase_type,
            img_size=config.img_size,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            config=config,
        )

        # 训练 (支持从断点恢复)
        trainer.train(dataloader, resume_epoch=resume_epoch)

        # 清理 (必须删除 trainer 释放 GPU 显存)
        del dataloader
        del trainer
        gc.collect()
        torch.cuda.empty_cache()
        logger.info(f"Phase {phase_name} GPU 内存已释放")

        # 保存最佳权重路径供下一阶段使用
        best_path = phase_dir / f'{phase_name}_best.pth'
        if best_path.exists():
            prev_weights = str(best_path)
            logger.info(f"最佳权重: {prev_weights}")
        else:
            latest = _find_latest_checkpoint(phase_dir)
            if latest:
                prev_weights = str(latest)
                logger.info(f"使用最后检查点: {latest.name}")

    logger.info(f"\n{'='*60}")
    logger.info("三阶段训练全部完成!")
    logger.info(f"结果目录: {output_base}")
    logger.info(f"{'='*60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HAMT 三阶段训练 (新方案)')
    parser.add_argument('--yaml', type=str, required=True, help='YAML 配置文件路径')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--resume', type=str, default=None,
                        help='从已有输出目录恢复训练 (如 results/hamt_deep_20260620_101253)')
    parser.add_argument('--output', type=str, default=None,
                        help='指定输出目录 (默认自动生成时间戳目录)')
    args = parser.parse_args()

    train_all_phases(
        args.yaml,
        gpu_id=args.gpu,
        output_dir=args.output,
        resume_dir=args.resume,
    )
