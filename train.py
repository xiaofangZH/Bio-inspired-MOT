#!/usr/bin/env python3
"""HAMT 训练脚本（MOT任务）。

改进点：
1. 使用监督损失（分类CE + bbox L1 + GIoU），不再把原始输出均值当损失。
2. 记录 MOT 相关训练指标（匹配召回、匹配精度、IoU、正样本/GT数量）。
3. 校验每个 epoch 是否覆盖了数据加载器提供的全部图片帧。
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
import yaml
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter

from data_loader import create_dataloader
from hmat.modeling.hmat_model import HMAT
from hmat.modeling.loss.matcher import HungarianMatcher
from hmat.modeling.loss.set_criterion import SetCriterion

PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parent.parent


def load_yaml_config(yaml_path: str | Path) -> dict:
    """加载 YAML 训练配置文件。"""
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


class Config:
    """训练配置（支持 YAML 文件或命令行参数）。"""

    def __init__(
        self,
        yaml_config: Optional[str | Path] = None,
        dataset_name: str = "dancetrack",
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        experiment_name: Optional[str] = None,
        num_workers: int = 2,
        backbone_weights: Optional[str] = None,
        max_frames_per_seq: Optional[int] = None,
        gpu_id: int = 0,
    ):
        # 若提供了 YAML 文件，优先加载
        self._yaml = None
        if yaml_config is not None:
            self._yaml = load_yaml_config(yaml_config)

        # ─── 基础参数 ───
        self.dataset_name = dataset_name
        self.experiment_name = experiment_name or f"hamt_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.num_classes = 1
        self.hidden_dim = 256
        self.num_queries = 100

        # ─── 训练超参（YAML优先，否则命令行/默认） ───
        yt = self._yaml.get("training", {}) if self._yaml else {}
        self.epochs = epochs if epochs is not None else (yt.get("epochs", 75))
        self.batch_size = batch_size if batch_size is not None else (yt.get("batch_size", 4))
        self.gradient_accumulation = yt.get("gradient_accumulation", 2)
        self.num_workers = num_workers if num_workers is not None else (yt.get("num_workers", 4))
        self.use_amp = yt.get("use_amp", True)
        self.grad_clip_norm = yt.get("grad_clip_norm", 0.5)
        img_size = yt.get("img_size", [640, 640])
        self.img_size = tuple(img_size)

        # ─── 损失权重 ───
        yl = self._yaml.get("loss", {}) if self._yaml else {}
        self.cls_weight = yl.get("cls_weight", 0.5)
        self.l1_weight = yl.get("l1_weight", 5.0)
        self.ciou_weight = yl.get("ciou_weight", 2.0)
        self.reid_weight = yl.get("reid_weight", 1.0)
        self.mask_weight = yl.get("mask_weight", 0.0)

        # ─── 分层学习率 ───
        ylr = self._yaml.get("lr", {}) if self._yaml else {}
        self.backbone_base_lr = ylr.get("backbone_base", 2.0e-4)
        self.head_base_lr = ylr.get("head_base", 4.0e-4)
        self.backbone_min_lr = ylr.get("backbone_min", 2.0e-5)
        self.head_min_lr = ylr.get("head_min", 4.0e-5)
        self.warmup_steps = ylr.get("warmup_steps", 1000)
        self.warmup_start_factor = ylr.get("warmup_start_factor", 0.1)
        self.weight_decay = ylr.get("weight_decay", 1.0e-4)
        # 兼容旧的 learning_rate 属性
        self.learning_rate = self.head_base_lr

        # ─── 三阶段训练 ───
        ys = self._yaml.get("stages", {}) if self._yaml else {}
        self.stages = ys

        # ─── 数据集路径 ───
        yd = self._yaml.get("datasets", {}) if self._yaml else {}
        self.dataset_paths = {
            "dancetrack": {"train": yd.get("dancetrack", {}).get("train", "data/DanceTrack/train"),
                           "val": yd.get("dancetrack", {}).get("val", "data/DanceTrack/val")},
            "mot17": {"train": yd.get("mot17", {}).get("train", "/home/user/MOT17/MOT17/train"),
                      "val": yd.get("mot17", {}).get("val", "/home/user/MOT17/MOT17/test")},
            "mot20": {"train": yd.get("mot20", {}).get("train", "/home/user/MOT20/MOT20/train"),
                      "val": yd.get("mot20", {}).get("val", "/home/user/MOT20/MOT20/test")},
        }

        # ─── 验证 ───
        yv = self._yaml.get("val", {}) if self._yaml else {}
        self.val_ratio = yv.get("split_ratio", [6, 4])

        # ─── 记忆衰减 ───
        ym = self._yaml.get("memory", {}) if self._yaml else {}
        self.memory_decay_coeff = ym.get("decay_coeff", 0.95)
        self.idle_decay_start = ym.get("idle_decay_start", 5)

        # ─── 检查点 ───
        yc = self._yaml.get("checkpoint", {}) if self._yaml else {}
        self.save_interval = yc.get("save_interval", 5)
        self.eval_interval = yc.get("eval_interval", 5)
        self.checkpoint_interval_steps = yc.get("interval_steps", 200)

        # 模型参数
        ym = self._yaml.get("model", {}) if self._yaml else {}
        self.max_memory_size = ym.get("max_memory_size", 6)

        gpu_id = gpu_id if torch.cuda.is_available() else 0
        self.device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        self.max_frames_per_seq = max_frames_per_seq

        # Backbone权重
        default_backbone = PROJECT_ROOT / "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
        if backbone_weights is None:
            yb = self._yaml.get("backbone", {}) if self._yaml else {}
            bw = yb.get("weights") if self._yaml and yb.get("weights") else None
            if bw:
                candidate = Path(bw).expanduser()
                if not candidate.is_absolute():
                    candidate = (PROJECT_ROOT / candidate).resolve()
                self.backbone_weights = str(candidate) if candidate.exists() else str(default_backbone)
            else:
                self.backbone_weights = str(default_backbone) if default_backbone.exists() else None
        else:
            candidate = Path(backbone_weights).expanduser()
            if not candidate.is_absolute():
                candidate = (PROJECT_ROOT / candidate).resolve()
            self.backbone_weights = str(candidate) if candidate.exists() else None

        self.output_dir = PROJECT_ROOT / "results" / self.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.log_dir = self.output_dir / "logs"
        self.log_dir.mkdir(exist_ok=True)
        self.plots_dir = self.output_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)


def setup_logger(config: Config) -> logging.Logger:
    logger_name = f"hamt.train.{config.experiment_name}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler = logging.FileHandler(config.log_dir / "training.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)

        logger.addHandler(file_handler)

    return logger


class Trainer:
    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logger(config)
        self.writer = SummaryWriter(str(config.log_dir))

        self.model = HMAT(
            num_classes=config.num_classes,
            hidden_dim=config.hidden_dim,
            num_queries=config.num_queries,
            num_detect_queries=100,
            max_track_age=30,
            backbone_weights=config.backbone_weights,
            use_batch_memory=True,
            max_memory_size=config.max_memory_size,
        ).to(config.device)

        # 全参数训练（阶段内由 configure_model_for_stage 控制冻结/解冻）
        for param in self.model.parameters():
            param.requires_grad = True

        # ─── Hungarian Matcher + SetCriterion (替换贪心匹配) ───
        self.matcher = HungarianMatcher(
            cost_class=config.cls_weight,
            cost_bbox=config.l1_weight,
            cost_ciou=config.ciou_weight,
        )
        weight_dict = {
            'loss_ce': config.cls_weight,
            'loss_bbox': config.l1_weight,
            'loss_ciou': config.ciou_weight,
            'loss_reid': config.reid_weight,
        }
        self.criterion = SetCriterion(
            num_classes=config.num_classes,
            matcher=self.matcher,
            weight_dict=weight_dict,
            eos_coef=0.1,
            losses=['labels', 'boxes', 'reid'],
            label_smoothing=0.1,
        )

        # ─── 分层学习率：从模型获取参数分组 ───
        param_groups = self.model.get_parameter_groups(
            base_lr=config.head_base_lr,
            backbone_lr_scale=config.backbone_base_lr / config.head_base_lr,
        )
        self.optimizer = optim.AdamW(
            param_groups,
            weight_decay=config.weight_decay,
        )
        # 为 backward 兼容 scheduler 操作整个 optimizer
        self.optimizer = self.optimizer

        self.global_step = 0
        self.epoch = 0
        self.stage_name = None  # 当前阶段名称 (stage1/stage2/stage3)
        self.history = []
        self.warmup_steps = config.warmup_steps

        # ─── CosineAnnealingLR (warmup 后启用) ───
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.epochs,
            eta_min=0,  # we manually clamp later
        )

        # ─── AMP GradScaler ───
        self.scaler = torch.amp.GradScaler("cuda") if (config.use_amp and "cuda" in config.device) else None

        # ─── 启用心跳衰减 ───
        if hasattr(self.model, "memory_bank") and hasattr(self.model.memory_bank, "memory_decay_coeff"):
            self.model.memory_bank.memory_decay_coeff = config.memory_decay_coeff

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        self.logger.info("初始化训练: %s", config.experiment_name)
        self.logger.info("设备: %s", config.device)
        self.logger.info("DINOv3权重: %s", config.backbone_weights if config.backbone_weights else "未提供")
        self.logger.info("训练策略: 分层LR (backbone=%.1e head=%.1e)", config.backbone_base_lr, config.head_base_lr)
        self.logger.info("AMP: %s | GradAccum: %s | ImgSize: %s", config.use_amp, config.gradient_accumulation, config.img_size)
        self.logger.info("可训练参数: %s/%s", f"{trainable:,}", f"{total:,}")

    def configure_model_for_stage(self, stage_config: dict, stage_name: Optional[str] = None):
        """根据阶段配置冻结/解冻模型参数。"""
        if stage_name:
            self.stage_name = stage_name
        freeze_backbone = stage_config.get("freeze_backbone", False)
        if freeze_backbone:
            # 冻结 backbone.vit 参数
            for name, param in self.model.named_parameters():
                if "backbone.vit." in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            self.logger.info("阶段配置: 冻结 backbone.vit，仅训练头部模块")
        else:
            for param in self.model.parameters():
                param.requires_grad = True
            self.logger.info("阶段配置: 全参数解冻训练")

    def _apply_stage_fixed_lr(self, stage_config: dict):
        """阶段3 lr_fixed 逻辑：锁定LR到最小值。"""
        if stage_config.get("lr_fixed", False):
            for pg in self.optimizer.param_groups:
                name = pg.get("name", "")
                if name == "backbone_vit":
                    pg["lr"] = self.config.backbone_min_lr
                elif name == "new_modules":
                    pg["lr"] = self.config.head_min_lr
                else:
                    pg["lr"] = self.config.head_min_lr

        # Clamp to min lr
        for pg in self.optimizer.param_groups:
            name = pg.get("name", "")
            if name == "backbone_vit":
                pg["lr"] = max(pg["lr"], self.config.backbone_min_lr)
            elif name == "new_modules":
                pg["lr"] = max(pg["lr"], self.config.head_min_lr)

    def _restore_optimizer_scheduler(self, resume_info: dict, stage_cfg: dict, remaining_epochs: int):
        """从 resume_info 恢复 optimizer 和 scheduler 状态。

        Args:
            resume_info: 包含 optimizer_state_dict 和 scheduler_state_dict
            stage_cfg: 当前阶段配置
            remaining_epochs: 剩余 epoch 数
        """
        # ─── 从checkpoint读取当前学习率（用于初始化新optimizer）───
        resume_lr = self.config.head_base_lr  # 默认值
        opt_state = resume_info.get("optimizer_state_dict")
        if opt_state and "param_groups" in opt_state and opt_state["param_groups"]:
            resume_lr = opt_state["param_groups"][0].get("lr", self.config.head_base_lr)
            self.logger.info(f"从checkpoint恢复学习率: {resume_lr:.6e}")

        # ─── 创建optimizer（使用恢复的学习率）───
        trainable_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)

        self.optimizer = optim.AdamW(
            [{"params": trainable_params, "lr": resume_lr}],
            weight_decay=self.config.weight_decay,
        )

        # 尝试加载 optimizer 状态（如果存在且匹配）
        if opt_state:
            try:
                self.optimizer.load_state_dict(opt_state)
                self.logger.info("✓ Optimizer 状态已恢复")
            except Exception as e:
                # 参数数量不匹配时，至少保持学习率一致
                self.logger.warning("Optimizer 状态恢复失败: %s，使用恢复的学习率初始化", e)

        # 创建 scheduler（用剩余 epochs）
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=remaining_epochs,
            eta_min=0,
        )

        # 加载 scheduler 状态（如果存在）
        # 注意：load_state_dict 会覆盖 T_max，需手动恢复
        sch_state = resume_info.get("scheduler_state_dict")
        if sch_state:
            try:
                self.scheduler.load_state_dict(sch_state)
                # 恢复正确的 T_max（checkpoint中的T_max可能与剩余epoch数不同）
                self.scheduler.T_max = remaining_epochs
                self.logger.info("✓ Scheduler 状态已恢复 (T_max=%d, last_epoch=%d)",
                               remaining_epochs, self.scheduler.last_epoch)
            except Exception as e:
                self.logger.warning("Scheduler 状态恢复失败: %s，使用新 scheduler", e)
                # 手动设置 last_epoch 以保持学习率连续性
                if "last_epoch" in sch_state:
                    self.scheduler.last_epoch = sch_state["last_epoch"]
                    self.logger.info("手动恢复 last_epoch=%d", sch_state["last_epoch"])

        # 重设 scaler
        self.scaler = torch.amp.GradScaler("cuda") if self.config.use_amp and "cuda" in self.config.device else None

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        freeze_str = "(frozen backbone)" if stage_cfg.get("freeze_backbone", False) else "(all params)"
        self.logger.info(
            "恢复训练: %d trainable params %s, remaining_epochs=%d, lr=%.6e",
            trainable, freeze_str, remaining_epochs, resume_lr,
        )

    def _apply_step_lr_warmup(self, step_in_stage: int):
        """Per-step linear LR warmup."""
        warmup = self.config.warmup_steps
        if warmup <= 0 or step_in_stage >= warmup:
            return False  # warmup finished
        factor = self.config.warmup_start_factor + (1.0 - self.config.warmup_start_factor) * (step_in_stage / max(warmup, 1))
        for pg in self.optimizer.param_groups:
            name = pg.get("name", "")
            base_lr = self.config.backbone_base_lr if name == "backbone_vit" else self.config.head_base_lr
            pg["lr"] = base_lr * factor
        return True  # still in warmup

    # Removed old _apply_stage_lr — now per-step warmup handles it

    @staticmethod
    def _xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
        cx, cy, w, h = boxes.unbind(-1)
        x1 = cx - 0.5 * w
        y1 = cy - 0.5 * h
        x2 = cx + 0.5 * w
        y2 = cy + 0.5 * h
        return torch.stack([x1, y1, x2, y2], dim=-1)

    @staticmethod
    def _box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        if boxes1.numel() == 0 or boxes2.numel() == 0:
            return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))

        area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
        area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

        lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
        rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, :, 0] * wh[:, :, 1]

        union = area1[:, None] + area2[None, :] - inter
        return inter / union.clamp(min=1e-6)

    def _complete_box_iou(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """
        CIoU (Complete IoU) — 同时考虑重叠面积、中心点距离和长宽比一致性。
        相比 GIoU，CIoU 对 bbox 回归给予更精确的梯度信号：
          CIoU = IoU - ρ²(b,bgt)/c² - α·v
        其中 ρ² 是中心点欧式距离，c² 是最小包围框对角线长，
        v = 4/π²·(arctan(wgt/hgt) - arctan(w/h))²，α = v/(1-IoU+v)。
        """
        eps = 1e-7
        # boxes in xyxy format: [x1, y1, x2, y2]
        w1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=eps)
        h1 = (boxes1[:, 3] - boxes1[:, 1]).clamp(min=eps)
        w2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=eps)
        h2 = (boxes2[:, 3] - boxes2[:, 1]).clamp(min=eps)

        # IoU
        area1 = w1 * h1
        area2 = w2 * h2
        inter_l = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
        inter_t = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
        inter_r = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
        inter_b = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])
        inter_w = (inter_r - inter_l).clamp(min=0)
        inter_h = (inter_b - inter_t).clamp(min=0)
        inter = inter_w * inter_h
        union = area1[:, None] + area2[None, :] - inter
        iou = inter / union.clamp(min=eps)

        # center distance penalty: ρ² / c²
        c1x = (boxes1[:, 0] + boxes1[:, 2]) / 2
        c1y = (boxes1[:, 1] + boxes1[:, 3]) / 2
        c2x = (boxes2[:, 0] + boxes2[:, 2]) / 2
        c2y = (boxes2[:, 1] + boxes2[:, 3]) / 2
        rho2 = (c1x[:, None] - c2x[None, :]) ** 2 + (c1y[:, None] - c2y[None, :]) ** 2

        enclose_l = torch.min(boxes1[:, None, 0], boxes2[None, :, 0])
        enclose_t = torch.min(boxes1[:, None, 1], boxes2[None, :, 1])
        enclose_r = torch.max(boxes1[:, None, 2], boxes2[None, :, 2])
        enclose_b = torch.max(boxes1[:, None, 3], boxes2[None, :, 3])
        c2 = (enclose_r - enclose_l).clamp(min=eps) ** 2 + (enclose_b - enclose_t).clamp(min=eps) ** 2

        # aspect ratio penalty v
        atan1 = torch.atan(w1 / h1)
        atan2 = torch.atan(w2 / h2)
        v = (4.0 / (math.pi ** 2)) * (atan2[None, :] - atan1[:, None]) ** 2

        # α: trade-off (no grad needed for α)
        with torch.no_grad():
            alpha = v / ((1.0 - iou) + v + eps)

        ciou = iou - rho2 / c2.clamp(min=eps) - alpha * v
        return ciou

    def _generalized_box_iou(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        iou = self._box_iou(boxes1, boxes2)
        if boxes1.numel() == 0 or boxes2.numel() == 0:
            return iou

        lt = torch.min(boxes1[:, None, :2], boxes2[None, :, :2])
        rb = torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:])
        wh = (rb - lt).clamp(min=0)
        area_c = wh[:, :, 0] * wh[:, :, 1]

        area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
        area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

        lt_inter = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
        rb_inter = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
        wh_inter = (rb_inter - lt_inter).clamp(min=0)
        inter = wh_inter[:, :, 0] * wh_inter[:, :, 1]
        union = area1[:, None] + area2[None, :] - inter

        return iou - (area_c - union) / area_c.clamp(min=1e-6)



    def _build_targets(self, frames):
        targets = []
        total_gt = 0
        for frame in frames:
            anns = frame.get("annotations", [])
            boxes = []
            labels = []
            img = frame.get("image")
            if img is not None:
                _, h_img, w_img = img.shape
            else:
                h_img, w_img = 1, 1

            for ann in anns:
                if "bbox_norm" in ann:
                    box = ann["bbox_norm"]
                    cx, cy, w, h = map(float, box)
                else:
                    x, y, w_abs, h_abs = map(float, ann.get("bbox", [0, 0, 0, 0]))
                    cx = (x + 0.5 * w_abs) / max(w_img, 1)
                    cy = (y + 0.5 * h_abs) / max(h_img, 1)
                    w = w_abs / max(w_img, 1)
                    h = h_abs / max(h_img, 1)

                cx = min(max(cx, 0.0), 1.0)
                cy = min(max(cy, 0.0), 1.0)
                w = min(max(w, 1e-6), 1.0)
                h = min(max(h, 1e-6), 1.0)

                class_id = int(ann.get("class", 1)) - 1
                class_id = min(max(class_id, 0), self.config.num_classes - 1)

                boxes.append([cx, cy, w, h])
                labels.append(class_id)

            if boxes:
                boxes_t = torch.tensor(boxes, dtype=torch.float32, device=self.config.device)
                labels_t = torch.tensor(labels, dtype=torch.long, device=self.config.device)
            else:
                boxes_t = torch.zeros((0, 4), dtype=torch.float32, device=self.config.device)
                labels_t = torch.zeros((0,), dtype=torch.long, device=self.config.device)

            total_gt += boxes_t.shape[0]
            targets.append({"boxes": boxes_t, "labels": labels_t})

        return targets, total_gt

    def _compute_loss_and_metrics(self, outputs, targets):
        """使用 HungarianMatcher + SetCriterion 计算损失及指标."""
        if isinstance(outputs, list) and outputs:
            outputs = outputs[0]
        if not isinstance(outputs, dict):
            self.logger.warning("输出不是 dict 类型: %s", type(outputs))
            return None, {}

        pred_logits = outputs.get("pred_logits")
        pred_boxes = outputs.get("pred_boxes")
        if pred_logits is None or pred_boxes is None:
            self.logger.warning("输出缺少 pred_logits 或 pred_boxes: keys=%s", list(outputs.keys()))
            return None, {}

        # NaN/Inf 检测: 模型输出异常 → 跳过该步（不更新权重）
        logits_finite = torch.isfinite(pred_logits).all()
        boxes_finite = torch.isfinite(pred_boxes).all()
        if not logits_finite or not boxes_finite:
            self.logger.warning(
                "模型输出含 NaN/Inf: logits_finite=%s boxes_finite=%s, "
                "logits_range=[%.4f, %.4f] boxes_range=[%.4f, %.4f]",
                logits_finite, boxes_finite,
                float(pred_logits.min()), float(pred_logits.max()),
                float(pred_boxes.min()), float(pred_boxes.max()),
            )
            return None, {}

        # 使用 SetCriterion (Hungarian matching + Focal loss + GIoU + L1)
        loss_dict = self.criterion(outputs, targets)

        total_loss = sum(loss_dict.values())
        if isinstance(total_loss, torch.Tensor) and total_loss.numel() == 0:
            self.logger.warning("total_loss 为空张量")
            return None, {}

        # 额外计算指标
        with torch.no_grad():
            bsz = pred_logits.shape[0]
            bg_idx = self.config.num_classes
            gt_total = sum(t["boxes"].shape[0] for t in targets)

            # 使用 Hungarian matcher 获取匹配结果
            indices = self.matcher(outputs, targets)

            matched_total = sum(len(src) for src, _ in indices)
            iou_sum = 0.0
            tp_iou50 = 0
            if matched_total > 0:
                for b, (src_idx, tgt_idx) in enumerate(indices):
                    if len(src_idx) == 0:
                        continue
                    pred_b = pred_boxes[b][src_idx]
                    tgt_b = targets[b]["boxes"][tgt_idx]
                    pred_xyxy = self._xywh_to_xyxy(pred_b)
                    tgt_xyxy = self._xywh_to_xyxy(tgt_b)
                    iou_vals = self._box_iou(pred_xyxy, tgt_xyxy).diag()
                    iou_sum += float(iou_vals.sum().item())
                    tp_iou50 += int((iou_vals >= 0.5).sum().item())

            obj_score = pred_logits.softmax(dim=-1)[..., 0]
            pred_pos_total = int((obj_score > 0.5).sum().item())

            precision = tp_iou50 / max(pred_pos_total, 1)
            recall = tp_iou50 / max(gt_total, 1)
            mean_iou = iou_sum / max(matched_total, 1)
            match_recall = matched_total / max(gt_total, 1)

        metrics = {
            "loss_total": float(total_loss.item()) if isinstance(total_loss, torch.Tensor) else float(total_loss),
            "loss_cls": float(loss_dict.get("loss_ce", 0)),
            "loss_l1": float(loss_dict.get("loss_bbox", 0)),
            "loss_ciou": float(loss_dict.get("loss_ciou", 0)),
            "loss_reid": float(loss_dict.get("loss_reid", 0)),
            "precision_iou50": float(precision),
            "recall_iou50": float(recall),
            "match_recall": float(match_recall),
            "mean_matched_iou": float(mean_iou),
            "gt_count": int(gt_total),
            "matched_count": int(matched_total),
            "pred_pos_count": int(pred_pos_total),
        }
        return total_loss, metrics

    def _forward_pass(self, frames):
        if not frames:
            return None, {}

        images = [frame_data["image"].to(self.config.device) for frame_data in frames]
        images = torch.stack(images)
        targets, total_gt = self._build_targets(frames)

        outputs = self.model(images, targets=targets)
        loss, metrics = self._compute_loss_and_metrics(outputs, targets)
        metrics["gt_count"] = int(total_gt)
        metrics["frame_count"] = len(frames)
        return loss, metrics

    def save_checkpoint(self, epoch_index: int, step_tag: Optional[int] = None):
        """Save checkpoint. Moves state dicts to CPU first to avoid GPU OOM."""
        import torch

        # Move model state to CPU to avoid GPU memory spike during torch.save
        model_sd = {k: v.cpu() for k, v in self.model.state_dict().items()}

        opt_state = self.optimizer.state_dict()
        opt_sd = {}
        for k, v in opt_state.items():
            if k == 'state':
                opt_sd[k] = {}
                for pk, pv in v.items():
                    opt_sd[k][pk] = {sk: sv.cpu() if isinstance(sv, torch.Tensor) else sv
                                     for sk, sv in pv.items()}
            elif isinstance(v, torch.Tensor):
                opt_sd[k] = v.cpu()
            else:
                opt_sd[k] = v

        checkpoint = {
            "epoch": epoch_index,
            "model_state_dict": model_sd,
            "optimizer_state_dict": opt_sd,
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "stage_name": self.stage_name,
            "config": vars(self.config),
        }

        torch.cuda.empty_cache()

        latest_path = self.config.checkpoint_dir / "latest.pth"
        torch.save(checkpoint, latest_path)

        if step_tag is not None:
            step_path = self.config.checkpoint_dir / f"step_{step_tag:07d}.pth"
            torch.save(checkpoint, step_path)

        if (epoch_index + 1) % self.config.save_interval == 0:
            epoch_path = self.config.checkpoint_dir / f"epoch_{epoch_index + 1:03d}.pth"
            torch.save(checkpoint, epoch_path)

        self.logger.info("检查点已保存: %s", latest_path)

    def _save_training_plots(self):
        if not self.history:
            return

        epochs = [item["epoch"] for item in self.history]
        train_loss = [item["train_loss"] for item in self.history]
        lr_values = [item["lr"] for item in self.history]
        precision = [item["precision_iou50"] for item in self.history]
        recall = [item["recall_iou50"] for item in self.history]

        fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        axes[0].plot(epochs, train_loss, marker="o", color="#1f77b4", label="Train Loss")
        axes[0].set_ylabel("Loss")
        axes[0].set_title(f"{self.config.dataset_name.upper()} Training Loss")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        axes[1].plot(epochs, precision, marker="o", color="#2ca02c", label="Precision@IoU50")
        axes[1].plot(epochs, recall, marker="o", color="#d62728", label="Recall@IoU50")
        axes[1].set_ylabel("Score")
        axes[1].set_title("MOT Detection Surrogate Metrics")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        axes[2].plot(epochs, lr_values, marker="o", color="#ff7f0e", label="Learning Rate")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("LR")
        axes[2].set_title("Learning Rate Schedule")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()

        fig.tight_layout()
        plot_path = self.config.plots_dir / "training_curves.png"
        fig.savefig(plot_path, dpi=160)
        plt.close(fig)
        self.logger.info("训练曲线已保存: %s", plot_path)

        summary_path = self.config.output_dir / "training_history.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)
        self.logger.info("训练历史已保存: %s", summary_path)

    def train(self, train_dataloader, val_dataloader=None, stage_config: Optional[dict] = None, global_epoch_offset: int = 0, step_in_stage_start: int = 0):
        self.logger.info("开始训练: %s", self.config.dataset_name)
        self.logger.info("共 %s 个 epoch", self.config.epochs)
        self.logger.info("训练批次数: %s", len(train_dataloader))
        self.logger.info("Per-step warmup: %s steps (start factor: %.2f)", self.config.warmup_steps, self.config.warmup_start_factor)

        step_in_stage = step_in_stage_start
        warmup_active = step_in_stage < self.config.warmup_steps

        for epoch in range(self.config.epochs):
            global_epoch = global_epoch_offset + epoch
            epoch_start = time.time()
            self.epoch = epoch
            self.model.train()

            # ─── 每轮检查 lr_fixed (阶段3) ───
            if stage_config is not None:
                self._apply_stage_fixed_lr(stage_config)

            total_loss = 0.0
            num_steps = 0
            skipped_no_frames = 0
            failed_steps = 0
            expected_frames = 0
            processed_frames = 0
            accum_step = 0

            metric_sums = {
                "loss_cls": 0.0,
                "loss_l1": 0.0,
                "loss_ciou": 0.0,
                "loss_reid": 0.0,
                "precision_iou50": 0.0,
                "recall_iou50": 0.0,
                "match_recall": 0.0,
                "mean_matched_iou": 0.0,
                "gt_count": 0.0,
                "matched_count": 0.0,
                "pred_pos_count": 0.0,
            }

            for batch_idx, batch in enumerate(train_dataloader):
                sequences = batch if isinstance(batch, list) else [batch]
                for seq_idx, sequence in enumerate(sequences):
                    frames = sequence.get("frames", [])
                    if not frames:
                        skipped_no_frames += 1
                        continue

                    # ─── 跨序列时重置记忆库，释放 GPU 碎片 ───
                    if hasattr(self.model, 'memory_bank') and self.model.memory_bank is not None:
                        self.model.memory_bank.reset()
                        torch.cuda.empty_cache()

                    expected_frames += len(frames)

                    # 首次迭代时清零梯度
                    if accum_step == 0 and batch_idx == 0 and seq_idx == 0:
                        self.optimizer.zero_grad()

                    for i in range(0, len(frames), self.config.batch_size):
                        frame_chunk = frames[i:i + self.config.batch_size]
                        try:
                            # ─── AMP 前向 ───
                            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=self.config.use_amp and "cuda" in self.config.device):
                                loss, metrics = self._forward_pass(frame_chunk)
                            if loss is None:
                                failed_steps += 1
                                self.logger.warning(
                                    "loss=None (step=%s, seq=%s, chunk=%s, failed_total=%s)",
                                    self.global_step, seq_idx, i, failed_steps,
                                )
                                if failed_steps <= 3:
                                    self.logger.warning(
                                        "  → 可能原因: 模型输出含NaN/Inf, 请检查上一行日志"
                                    )
                                continue

                            # ─── NaN/Inf 防护 ───
                            if torch.isnan(loss) or torch.isinf(loss):
                                self.logger.warning("检测到 NaN/Inf loss，跳过该步")
                                failed_steps += 1
                                continue

                            # ─── AMP backward + grad accum ───
                            # zero_grad 仅在累积完成时调用，不在此处清空
                            loss_scaled = loss / self.config.gradient_accumulation
                            self.scaler.scale(loss_scaled).backward()
                            accum_step += 1

                            if accum_step % self.config.gradient_accumulation == 0:
                                self.scaler.unscale_(self.optimizer)

                                # ─── Per-step LR warmup ───
                                step_in_stage += 1
                                if warmup_active:
                                    warmup_active = self._apply_step_lr_warmup(step_in_stage)

                                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
                                self.scaler.step(self.optimizer)
                                self.scaler.update()
                                self.optimizer.zero_grad()

                            total_loss += float(loss.item())
                            num_steps += 1
                            self.global_step += 1
                            processed_frames += metrics.get("frame_count", len(frame_chunk))

                            for k in metric_sums.keys():
                                metric_sums[k] += float(metrics.get(k, 0.0))

                            if self.global_step % 20 == 0:
                                avg_loss = total_loss / max(num_steps, 1)
                                current_lr = self.optimizer.param_groups[0]["lr"]
                                self.logger.info(
                                    "Epoch %s, Step %s, AvgLoss: %.6f, LR: %.2e, ProcessedFrames: %s/%s",
                                    epoch + 1,
                                    self.global_step,
                                    avg_loss,
                                    current_lr,
                                    processed_frames,
                                    expected_frames,
                                )

                            if self.global_step % self.config.checkpoint_interval_steps == 0:
                                self.save_checkpoint(epoch, step_tag=self.global_step)

                            # ─── 定期回收 GPU/CPU 内存碎片 ───
                            if self.global_step % 1000 == 0:
                                import gc
                                gc.collect()
                                torch.cuda.empty_cache()
                        except Exception as exc:
                            failed_steps += 1
                            self.logger.warning(
                                "训练步失败 (batch=%s seq=%s chunk=%s): %s",
                                batch_idx,
                                seq_idx,
                                i,
                                exc,
                            )
                            continue

            if num_steps == 0:
                raise RuntimeError(
                    f"Epoch {epoch + 1} 没有产生任何有效优化步。"
                    f" skipped_no_frames={skipped_no_frames}, failed_steps={failed_steps}, expected_frames={expected_frames}"
                )

            coverage = processed_frames / max(expected_frames, 1)
            if processed_frames != expected_frames:
                self.logger.warning(
                    "Epoch %d 帧覆盖不完整: processed=%d expected=%d (%.1f%%), skipped_no_frames=%d, failed=%d",
                    epoch + 1, processed_frames, expected_frames,
                    coverage * 100, skipped_no_frames, failed_steps,
                )

            # ─── 处理末尾的剩余梯度累积 ───
            if accum_step % self.config.gradient_accumulation != 0:
                self.scaler.unscale_(self.optimizer)

                # ─── Per-step LR warmup (final accumulate) ───
                step_in_stage += 1
                if warmup_active:
                    warmup_active = self._apply_step_lr_warmup(step_in_stage)

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            avg_loss = total_loss / max(num_steps, 1)

            # ─── Scheduler: cosine annealing (per-epoch, skip when lr_fixed) ───
            if stage_config is None or not stage_config.get("lr_fixed", False):
                self.scheduler.step()

            epoch_seconds = time.time() - epoch_start
            steps_per_second = num_steps / max(epoch_seconds, 1e-6)
            frames_per_second = processed_frames / max(epoch_seconds, 1e-6)

            avg_metrics = {k: (v / max(num_steps, 1)) for k, v in metric_sums.items()}
            current_lr = self.optimizer.param_groups[0]["lr"]

            self.logger.info(
                "Epoch %s 完成 | loss=%.6f cls=%.6f l1=%.6f ciou=%.6f reid=%.6f | P@50=%.4f R@50=%.4f MatchR=%.4f mIoU=%.4f | steps=%s failed=%s skipped_seq=%s | frames=%s/%s(%.3f) | %.2fs %.3f step/s %.3f frame/s",
                epoch + 1,
                avg_loss,
                avg_metrics.get("loss_cls", 0.0),
                avg_metrics.get("loss_l1", 0.0),
                avg_metrics.get("loss_ciou", 0.0),
                avg_metrics.get("loss_reid", 0.0),
                avg_metrics.get("precision_iou50", 0.0),
                avg_metrics.get("recall_iou50", 0.0),
                avg_metrics.get("match_recall", 0.0),
                avg_metrics.get("mean_matched_iou", 0.0),
                num_steps,
                failed_steps,
                skipped_no_frames,
                processed_frames,
                expected_frames,
                coverage,
                epoch_seconds,
                steps_per_second,
                frames_per_second,
            )

            self.writer.add_scalar("loss/epoch", avg_loss, epoch + 1)
            for k in avg_metrics:
                self.writer.add_scalar(f"loss/{k}", avg_metrics[k], epoch + 1)
            self.writer.add_scalar("train/processed_frames", processed_frames, epoch + 1)
            self.writer.add_scalar("train/expected_frames", expected_frames, epoch + 1)
            self.writer.add_scalar("train/frame_coverage", coverage, epoch + 1)
            self.writer.add_scalar("train/failed_steps", failed_steps, epoch + 1)
            self.writer.add_scalar("train/epoch_seconds", epoch_seconds, epoch + 1)
            self.writer.add_scalar("train/steps_per_second", steps_per_second, epoch + 1)
            self.writer.add_scalar("train/frames_per_second", frames_per_second, epoch + 1)
            self.writer.add_scalar("lr", current_lr, epoch + 1)

            self.history.append({
                "epoch": epoch + 1,
                "train_loss": avg_loss,
                "lr": current_lr,
                **avg_metrics,
                "processed_frames": processed_frames,
                "expected_frames": expected_frames,
                "frame_coverage": coverage,
                "failed_steps": failed_steps,
                "epoch_seconds": epoch_seconds,
                "steps_per_second": steps_per_second,
                "frames_per_second": frames_per_second,
            })

            if (epoch + 1) % self.config.save_interval == 0:
                self.save_checkpoint(epoch)

            if val_dataloader is not None and (epoch + 1) % self.config.eval_interval == 0:
                self.logger.info("验证集已提供，后续可扩展MOTA/IDF1评估")

            # ─── 释放显存碎片 ───
            torch.cuda.empty_cache()

        self.save_checkpoint(self.config.epochs - 1)
        self._save_training_plots()
        # writer.close() 由 train_three_stage 统一管理，避免跨阶段关闭导致崩溃
        self.logger.info("训练完成")


def train_three_stage(
    yaml_path: str,
    backbone_weights: Optional[str] = None,
    max_frames_per_seq: Optional[int] = None,
    resume_checkpoint: Optional[str] = None,
    start_stage: int = 1,
    gpu_id: int = 0,
):
    """三阶段训练入口：加载 YAML 配置并依次执行阶段1→阶段2→阶段3。

    Args:
        yaml_path: YAML 配置文件路径
        backbone_weights: DINOv3 权重路径
        max_frames_per_seq: 每序列最大帧数
        resume_checkpoint: 续训检查点路径（加载权重后从指定阶段开始）
        start_stage: 起始阶段编号 (1-based, 1/2/3)
        gpu_id: GPU 编号
    """
    config = Config(
        yaml_config=yaml_path,
        backbone_weights=backbone_weights,
        max_frames_per_seq=max_frames_per_seq,
        gpu_id=gpu_id,
    )

    print("\n" + "=" * 70)
    print(f"HAMT 三阶段训练启动")
    print(f"实验名称: {config.experiment_name}")
    print(f"总 Epochs: {config.epochs}")
    print(f"Batch Size: {config.batch_size} x{config.gradient_accumulation} = {config.batch_size * config.gradient_accumulation}")
    print(f"AMP: {config.use_amp} | ImgSize: {config.img_size}")
    print(f"分层LR: backbone={config.backbone_base_lr:.1e} head={config.head_base_lr:.1e}")
    if resume_checkpoint:
        print(f"续训检查点: {resume_checkpoint}")
    if start_stage > 1:
        print(f"起始阶段: stage{start_stage}")
    print("=" * 70 + "\n")

    # 创建一个 Trainer 实例
    trainer = Trainer(config)

    # ─── 加载续训检查点 ───
    resume_info = None
    if resume_checkpoint:
        resume_info = _load_checkpoint_full(trainer, resume_checkpoint, config)

    stages = config.stages
    stage_names = ["stage1", "stage2", "stage3"]
    global_epoch = 0

    # ─── 确定需要跳过的阶段 ───
    resume_stage_idx = 0
    if resume_info and resume_info.get("stage_name"):
        resume_stage_idx = int(resume_info["stage_name"][-1])  # 1, 2, 3

    for stage_name in stage_names:
        stage_idx = int(stage_name[-1])  # 1, 2, 3
        if stage_name not in stages:
            continue

        stage_cfg = stages[stage_name]
        ep_range = stage_cfg.get("epochs", [0, 0])
        stage_start_ep, stage_end_ep = ep_range[0], ep_range[1]
        n_epochs = stage_end_ep - stage_start_ep + 1
        if n_epochs <= 0:
            continue

        # ─── 判断是否需要跳过此阶段 ───
        if resume_info and stage_idx < resume_stage_idx:
            # 此阶段在断点之前已完成，跳过
            global_epoch = stage_end_ep + 1
            continue

        # ─── 判断是否需要恢复此阶段 ───
        if resume_info and resume_info.get("stage_name") == stage_name:
            # 当前阶段需要恢复
            resume_epoch = resume_info.get("epoch", 0)
            resume_global_step = resume_info.get("global_step", 0)
            resume_stage_epoch = resume_epoch  # 阶段内的epoch编号

            print("\n" + "-" * 50)
            print(f"恢复阶段: {stage_name} (从 Epoch {resume_stage_epoch + 1} 继续)")
            print(f"已完成: {resume_stage_epoch + 1} epochs, global_step={resume_global_step}")
            print("-" * 50 + "\n")

            # 计算剩余epoch数
            remaining_epochs = n_epochs - resume_stage_epoch - 1
            if remaining_epochs <= 0:
                print(f"阶段 {stage_name} 已完成，跳过")
                global_epoch = stage_end_ep + 1
                resume_info = None
                continue

            # 创建数据加载器
            sample_ratio = stage_cfg.get("sample_ratio", [4, 3, 3])
            augment_mode = stage_cfg.get("augment", "basic")
            combined_loader = create_combined_dataloader(
                config=config,
                sample_ratio=sample_ratio,
                augment=augment_mode,
                batch_size=1,
            )

            # 配置模型状态
            trainer.configure_model_for_stage(stage_cfg, stage_name)

            # 恢复 optimizer 和 scheduler 状态
            trainer._restore_optimizer_scheduler(
                resume_info,
                stage_cfg,
                remaining_epochs,
            )

            # 设置 global_step 和 epoch
            trainer.global_step = resume_global_step
            trainer.epoch = resume_epoch

            # 临时 config epochs
            orig_epochs = config.epochs
            config.epochs = remaining_epochs
            trainer.config = config

            # 计算 step_in_stage_start (用于 warmup)
            # 从断点恢复时，warmup 已经完成，所以设为 warmup_steps
            step_in_stage_start = max(trainer.config.warmup_steps, resume_global_step)

            trainer.train(
                combined_loader,
                val_dataloader=None,
                stage_config=stage_cfg,
                global_epoch_offset=stage_start_ep + resume_stage_epoch + 1,
                step_in_stage_start=step_in_stage_start,
            )

            config.epochs = orig_epochs
            global_epoch = stage_end_ep + 1
            resume_info = None  # 清除 resume_info，后续阶段正常开始

        elif stage_idx < start_stage:
            # 跳过已完成的阶段
            global_epoch = stage_end_ep + 1
            continue

        else:
            # 正常开始新阶段
            sample_ratio = stage_cfg.get("sample_ratio", [4, 3, 3])
            augment_mode = stage_cfg.get("augment", "basic")

            print("\n" + "-" * 50)
            print(f"阶段: {stage_name} (Epoch {stage_start_ep}-{stage_end_ep}, {n_epochs} epochs)")
            print(f"采样配比 MOT17:MOT20:DanceTrack = {sample_ratio[0]}:{sample_ratio[1]}:{sample_ratio[2]}")
            print(f"增强: {augment_mode} | 冻结主骨: {stage_cfg.get('freeze_backbone', False)}")
            print("-" * 50 + "\n")

            combined_loader = create_combined_dataloader(
                config=config,
                sample_ratio=sample_ratio,
                augment=augment_mode,
                batch_size=1,
            )

            # 配置模型状态
            trainer.configure_model_for_stage(stage_cfg, stage_name)

            # 阶段切换时重置记忆库
            if hasattr(trainer.model, 'memory_bank'):
                trainer.model.memory_bank.reset()
                trainer.logger.info("记忆库已重置（阶段切换）")

            # 创建新 optimizer
            trainable_params = []
            for name, param in trainer.model.named_parameters():
                if param.requires_grad:
                    trainable_params.append(param)
            trainer.optimizer = optim.AdamW(
                [{"params": trainable_params, "lr": config.head_base_lr}],
                weight_decay=config.weight_decay,
            )
            freeze_str = "(frozen backbone)" if stage_cfg.get("freeze_backbone", False) else "(all params)"
            trainer.logger.info(
                "阶段优化器重置: %s, %d trainable params %s",
                stage_name, len(trainable_params), freeze_str,
            )

            # 创建 scheduler
            trainer.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                trainer.optimizer,
                T_max=n_epochs,
                eta_min=0,
            )
            trainer.scaler = torch.amp.GradScaler("cuda") if config.use_amp and "cuda" in config.device else None

            # 临时 config epochs
            orig_epochs = config.epochs
            config.epochs = n_epochs
            trainer.config = config

            trainer.train(
                combined_loader,
                val_dataloader=None,
                stage_config=stage_cfg,
                global_epoch_offset=stage_start_ep,
                step_in_stage_start=0,
            )

            config.epochs = orig_epochs
            global_epoch = stage_end_ep + 1

    # 保存最终模型
    trainer.save_checkpoint(config.epochs - 1)
    trainer._save_training_plots()
    trainer.writer.close()
    trainer.logger.info("三阶段训练完成")
    return trainer


def _load_checkpoint_weights(trainer, ckpt_path: str, config):
    """加载检查点权重到 trainer.model（strict=False 兼容架构变更）。

    自动跳过形状不匹配的参数/缓冲区（如 1D→2D PE、3→5 通道 gate）。
    """
    import torch

    ckpt = torch.load(ckpt_path, map_location=config.device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))

    # 获取当前模型的完整 shape 字典（含参数和缓冲区）
    model_shapes = {k: v.shape for k, v in trainer.model.state_dict().items()}

    # 处理 DataParallel 'module.' 前缀，跳过形状不匹配的 key
    new_state = {}
    skip_count = 0
    for k, v in state.items():
        key = k.replace("module.", "")
        target_shape = model_shapes.get(key)
        if target_shape is not None and target_shape != v.shape:
            skip_count += 1
            continue
        if target_shape is None:
            # 当前模型中不存在的 key 也跳过（旧架构残留）
            skip_count += 1
            continue
        new_state[key] = v

    if skip_count > 0:
        trainer.logger.info("加载检查点: %d 个 key 因形状不匹配/不存在跳过 (架构变更)", skip_count)

    missing, unexpected = trainer.model.load_state_dict(new_state, strict=False)
    if missing:
        trainer.logger.info("加载检查点: 新模块随机初始化 (%d keys)", len(missing))
        for mk in missing[:6]:
            trainer.logger.info("  new: %s", mk)
        if len(missing) > 6:
            trainer.logger.info("  ... +%d more", len(missing) - 6)
    if unexpected:
        trainer.logger.info("加载检查点: %d 个多余 key 已忽略 (旧架构残留)", len(unexpected))

    trainer.logger.info("✓ 检查点加载完成: epoch=%s, global_step=%s",
                        ckpt.get("epoch", "?"), ckpt.get("global_step", "?"))


def _load_checkpoint_full(trainer, ckpt_path: str, config):
    """完整加载检查点：模型权重 + optimizer/scheduler 状态 + epoch/step。

    返回 resume_info dict，包含：
        - epoch: 阶段内 epoch 编号
        - global_step: 全局步数
        - stage_name: 当前阶段名称（若缺失则根据 epoch 范围推断）
        - optimizer_state_dict
        - scheduler_state_dict
    """
    import torch

    ckpt = torch.load(ckpt_path, map_location=config.device, weights_only=False)

    # 加载模型权重
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model_shapes = {k: v.shape for k, v in trainer.model.state_dict().items()}

    new_state = {}
    skip_count = 0
    for k, v in state.items():
        key = k.replace("module.", "")
        target_shape = model_shapes.get(key)
        if target_shape is not None and target_shape != v.shape:
            skip_count += 1
            continue
        if target_shape is None:
            skip_count += 1
            continue
        new_state[key] = v

    if skip_count > 0:
        trainer.logger.info("加载检查点: %d 个 key 因形状不匹配跳过", skip_count)

    missing, unexpected = trainer.model.load_state_dict(new_state, strict=False)
    if missing:
        trainer.logger.info("加载检查点: 新模块随机初始化 (%d keys)", len(missing))
    if unexpected:
        trainer.logger.info("加载检查点: %d 个多余 key 已忽略", len(unexpected))

    epoch = ckpt.get("epoch", 0)
    global_step = ckpt.get("global_step", 0)
    stage_name = ckpt.get("stage_name", None)

    # 若 checkpoint 缺少 stage_name，根据 epoch 范围 + global_step 推断
    if stage_name is None:
        stage_name = _infer_stage_from_epoch(epoch, config.stages, global_step)
        if stage_name is None:
            trainer.logger.warning(
                "无法推断阶段 (epoch=%s global_step=%s)，将使用 start_stage 参数",
                epoch, global_step,
            )
        else:
            trainer.logger.info("推断阶段: epoch=%s global_step=%s → stage=%s",
                               epoch, global_step, stage_name)

    resume_info = {
        "epoch": epoch,
        "global_step": global_step,
        "stage_name": stage_name,
        "optimizer_state_dict": ckpt.get("optimizer_state_dict", None),
        "scheduler_state_dict": ckpt.get("scheduler_state_dict", None),
    }

    trainer.logger.info(
        "✓ 检查点完整加载: epoch=%s, global_step=%s, stage=%s",
        resume_info["epoch"], resume_info["global_step"], resume_info["stage_name"],
    )

    return resume_info


def _infer_stage_from_epoch(epoch: int, stages: dict, global_step: int = 0) -> Optional[str]:
    """根据 epoch 编号推断所属阶段。

    注意：checkpoint 中的 epoch 是阶段内编号（从 0 开始），
    可能多个阶段的 epoch 范围重叠（如 stage1 epoch=4 和 stage2 epoch=4 无法区分），
    此时用 global_step 辅助判断。

    Returns None 表示无法确定。
    """
    possible = []
    for stage_name, stage_cfg in stages.items():
        ep_range = stage_cfg.get("epochs", [0, 0])
        start_ep, end_ep = ep_range[0], ep_range[1]
        n_epochs = end_ep - start_ep + 1
        if 0 <= epoch < n_epochs:
            possible.append((stage_name, n_epochs, start_ep))

    if len(possible) == 0:
        return "stage3" if "stage3" in stages else None
    if len(possible) == 1:
        return possible[0][0]

    # 歧义情况：多个阶段可能包含此 epoch（如 epoch=4 在两个阶段都 < 阶段 epoch 数）
    # 使用 global_step 辅助判断（stage1 最多约 150000 steps）
    if global_step > 200000:
        # 远超 stage1 预期步数，优先返回后续阶段
        for sn, _, _ in possible:
            if sn != "stage1":
                return sn
    # 无法确定，返回 None 让调用方根据 start_stage 参数决定
    return None


def create_combined_dataloader(
    config: Config,
    sample_ratio: list,
    augment: str = "basic",
    batch_size: int = 1,
):
    """创建多数据集混合 DataLoader。

    sample_ratio = [MOT17, MOT20, DanceTrack] 比例（控制采样权重，非副本数）。
    augment: basic / full / minimal
    """
    from data_loader import MOTDataset, collate_sequences

    datasets_info = [
        ("mot17", config.dataset_paths["mot17"]["train"]),
        ("mot20", config.dataset_paths["mot20"]["train"]),
        ("dancetrack", config.dataset_paths["dancetrack"]["train"]),
    ]

    # 创建各数据集实例（每份仅一个，不复制）
    ds_list = []
    ds_weights = []
    for idx, ((ds_name, ds_path), ratio) in enumerate(zip(datasets_info, sample_ratio)):
        if ratio <= 0:
            continue
        root_path = Path(ds_path)
        if not root_path.exists():
            print(f"[WARN] 数据集路径不存在跳过: {root_path}")
            continue
        dataset = MOTDataset(
            root_dir=str(root_path),
            dataset_name=ds_name,
            split="train",
            img_size=config.img_size,
            max_frames_per_seq=config.max_frames_per_seq,
        )
        ds_list.append(dataset)
        ds_weights.extend([idx] * len(dataset))

    if not ds_list:
        raise RuntimeError("没有可用的数据集！")

    combined = ConcatDataset(ds_list)
    total_seqs = sum(len(d) for d in ds_list)

    # 加权采样：按 sample_ratio 分配每个序列被采样的概率
    weights = []
    active_ratios = [r for r in sample_ratio if r > 0]
    for idx, dataset in enumerate(ds_list):
        w = active_ratios[idx] / sum(active_ratios)
        weights.extend([w / len(dataset)] * len(dataset))

    sampler = torch.utils.data.WeightedRandomSampler(
        weights=weights,
        num_samples=total_seqs,
        replacement=True,
    )

    print(f"[CombinedData] 合并 {len(ds_list)} 个数据集, 共 {total_seqs} 序列, "
          f"采样权重 MOT17:MOT20:DanceTrack={sample_ratio}")

    # WeightedRandomSampler + multi-worker 不稳定，强制 num_workers=0
    return DataLoader(
        combined,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        collate_fn=collate_sequences,
        pin_memory=torch.cuda.is_available(),
    )


def train_on_dataset(
    dataset_name: str,
    train_root,
    val_root=None,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    num_workers: int = 2,
    backbone_weights: Optional[str] = None,
    max_frames_per_seq: Optional[int] = None,
):
    """单数据集训练（兼容旧版调用）."""
    config = Config(
        dataset_name=dataset_name,
        epochs=epochs,
        batch_size=batch_size,
        num_workers=num_workers,
        backbone_weights=backbone_weights,
        max_frames_per_seq=max_frames_per_seq,
    )

    print("\n" + "=" * 70)
    print(f"在 {dataset_name.upper()} 上开始训练")
    print(f"训练集: {train_root}")
    if val_root:
        print(f"验证集: {val_root}")
    print(f"推荐配置: epochs={config.epochs}, batch_size={config.batch_size}")
    print(f"max_frames_per_seq: {config.max_frames_per_seq if config.max_frames_per_seq else 'ALL'}")
    print("=" * 70 + "\n")

    train_loader = create_dataloader(
        root_dir=str(train_root),
        dataset_name=dataset_name,
        split="train",
        batch_size=1,
        num_workers=config.num_workers,
        img_size=config.img_size,
        max_frames_per_seq=config.max_frames_per_seq,
    )

    val_loader = None
    if val_root:
        val_loader = create_dataloader(
            root_dir=str(val_root),
            dataset_name=dataset_name,
            split="val",
            batch_size=1,
            num_workers=config.num_workers,
            img_size=config.img_size,
            max_frames_per_seq=config.max_frames_per_seq,
        )

    trainer = Trainer(config)
    trainer.train(train_loader, val_loader)
    return trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HAMT训练脚本")
    parser.add_argument("--yaml", type=str, default="configs/dancetrack_full.yaml", help="YAML配置文件路径（三阶段训练）")
    parser.add_argument("--dataset", type=str, default=None, help="只训练单个数据集（兼容旧版）")
    parser.add_argument("--epochs", type=int, default=None, help="覆盖训练轮次")
    parser.add_argument("--batch-size", type=int, default=None, help="覆盖帧微批大小")
    parser.add_argument("--num-workers", type=int, default=4, help="数据加载线程数")
    parser.add_argument("--backbone-weights", type=str, default=None, help="DINOv3权重路径")
    parser.add_argument("--resume", type=str, default=None, help="续训检查点路径 (e.g. results/.../checkpoints/step_*.pth)")
    parser.add_argument("--gpu", type=int, default=0, help="使用的GPU编号 (0-7)")
    parser.add_argument("--start-stage", type=int, default=1, choices=[1, 2, 3], help="从指定阶段开始训练 (1/2/3)")
    parser.add_argument(
        "--max-frames-per-seq",
        type=int,
        default=0,
        help="每个序列最多读取帧数，0表示读取全部",
    )
    args = parser.parse_args()

    max_frames = args.max_frames_per_seq if args.max_frames_per_seq and args.max_frames_per_seq > 0 else None

    if args.dataset:
        # ─── 单数据集训练（兼容旧版） ───
        config = Config(
            dataset_name=args.dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            backbone_weights=args.backbone_weights,
            max_frames_per_seq=max_frames,
            gpu_id=args.gpu,
        )
        print("\n" + "=" * 70)
        print(f"单数据集训练: {args.dataset.upper()}")
        print("=" * 70 + "\n")
        train_loader = create_dataloader(
            root_dir=config.dataset_paths[args.dataset]["train"],
            dataset_name=args.dataset,
            split="train",
            batch_size=1,
            num_workers=config.num_workers,
            img_size=config.img_size,
            max_frames_per_seq=max_frames,
        )
        trainer = Trainer(config)
        trainer.train(train_loader)
        trainer.save_checkpoint(config.epochs - 1)
        trainer._save_training_plots()
        trainer.writer.close()
    else:
        # ─── 三阶段训练（默认） ───
        yaml_path = Path(args.yaml)
        if not yaml_path.is_absolute():
            yaml_path = PROJECT_ROOT / yaml_path
        if not yaml_path.exists():
            print(f"[ERROR] YAML配置文件不存在: {yaml_path}")
            exit(1)
        train_three_stage(
            yaml_path=str(yaml_path),
            backbone_weights=args.backbone_weights,
            max_frames_per_seq=max_frames,
            resume_checkpoint=args.resume,
            start_stage=args.start_stage,
            gpu_id=args.gpu,
        )

    if torch.cuda.is_available():
        print(f"GPU可用: {torch.cuda.get_device_name(args.gpu)}")
        print(f"显存: {torch.cuda.get_device_properties(args.gpu).total_memory / 1e9:.1f} GB")
    else:
        print("GPU不可用，将使用CPU训练")
