#!/usr/bin/env python3
"""训练监控脚本：实时从日志提取指标并记录到 JSON，训练结束后生成可视化图表."""

import re
import json
import time
import sys
import os
from pathlib import Path
from datetime import datetime

LOG_DIR = Path(__file__).resolve().parent / "results"
METRICS_FILE = LOG_DIR / "training_metrics_v2.json"

# Epoch log pattern matching
EPOCH_PATTERN = re.compile(
    r"Epoch\s+(\d+)\s+完成\s+\|\s+"
    r"loss=([\d.]+)\s+cls=([\d.]+)\s+l1=([\d.]+)\s+ciou=([\d.]+)\s+\|\s+"
    r"P@50=([\d.]+)\s+R@50=([\d.]+)\s+MatchR=([\d.]+)\s+mIoU=([\d.]+)\s+\|\s+"
    r"steps=(\d+)\s+failed=(\d+)\s+skipped_seq=(\d+)\s+\|\s+"
    r"frames=(\d+)/(\d+).*?\|\s+"
    r"([\d.]+)s\s+([\d.]+)\s+step/s\s+([\d.]+)\s+frame/s"
)

STAGE_PATTERN = re.compile(r"阶段:\s+(\w+)\s+\(Epoch\s+(\d+)-(\d+)")


def extract_metrics(log_path):
    """Extract all epoch metrics from a log file."""
    if not os.path.exists(log_path):
        return []

    metrics = []
    current_stage = "unknown"

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            # Detect stage transitions
            stage_match = STAGE_PATTERN.search(line)
            if stage_match:
                current_stage = stage_match.group(1)

            # Extract epoch metrics
            match = EPOCH_PATTERN.search(line)
            if match:
                epoch = int(match.group(1))
                d = {
                    "epoch": epoch,
                    "stage": current_stage,
                    "timestamp": datetime.now().isoformat(),
                    "loss_total": float(match.group(2)),
                    "loss_cls": float(match.group(3)),
                    "loss_l1": float(match.group(4)),
                    "loss_ciou": float(match.group(5)),
                    "precision_iou50": float(match.group(6)),
                    "recall_iou50": float(match.group(7)),
                    "match_recall": float(match.group(8)),
                    "mean_matched_iou": float(match.group(9)),
                    "steps": int(match.group(10)),
                    "failed_steps": int(match.group(11)),
                    "skipped_sequences": int(match.group(12)),
                    "processed_frames": int(match.group(13)),
                    "expected_frames": int(match.group(14)),
                    "epoch_seconds": float(match.group(15)),
                    "steps_per_second": float(match.group(16)),
                    "frames_per_second": float(match.group(17)),
                }
                metrics.append(d)
    return metrics


if __name__ == "__main__":
    # Find latest log
    logs = sorted(LOG_DIR.glob("train_v[0-9]_*.log"))
    if not logs:
        print("No training log found.")
        sys.exit(1)

    log_path = str(logs[-1])
    print(f"Monitoring: {log_path}")

    seen_epochs = set()
    while True:
        metrics = extract_metrics(log_path)

        if metrics:
            new_metrics = [m for m in metrics if m["epoch"] not in seen_epochs]
            for m in new_metrics:
                seen_epochs.add(m["epoch"])
                print(
                    f"[{m['stage']:7s}] Epoch {m['epoch']:3d} | "
                    f"Loss={m['loss_total']:.6f} cls={m['loss_cls']:.6f} l1={m['loss_l1']:.6f} ciou={m['loss_ciou']:.6f} | "
                    f"P@50={m['precision_iou50']:.4f} R@50={m['recall_iou50']:.4f} "
                    f"MatchR={m['match_recall']:.4f} mIoU={m['mean_matched_iou']:.4f} | "
                    f"{m['epoch_seconds']:.1f}s"
                )

            # Save to JSON
            with open(METRICS_FILE, "w") as f:
                json.dump(metrics, f, indent=2)

        time.sleep(60)  # Check every 60s
