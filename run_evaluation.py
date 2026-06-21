#!/usr/bin/env python3
"""HAMT 评估流水线 — 在多数据集上评估并生成结构化 JSON 供可视化使用.

用法:
  python run_evaluation.py --checkpoint results/hamt_full_*/checkpoints/latest.pth \
                           --dataset dancetrack --conf 0.3

依赖 eval.py 中的 MOTEvaluator 和 compute_mot_metrics().
注意: eval.py 当前使用硬编码的模型参数，若架构有变更需同步更新。
"""
import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def evaluate_single(dataset_name, checkpoint_path, config_path, device='cuda', conf=0.3):
    """Run evaluation on one dataset.

    Returns dict of MOT metrics or None on failure.
    """
    from eval import MOTEvaluator
    from eval import create_dataloader
    import yaml

    # Load config for dataset paths
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    ds_cfg = config.get('datasets', {}).get(dataset_name, {})
    test_root = ds_cfg.get('val') or ds_cfg.get('test')
    if not test_root:
        print(f"  ⚠ No val/test path found for dataset '{dataset_name}' in config")
        return None

    print(f"  Dataset: {dataset_name}, test_root: {test_root}")

    # Build evaluator
    evaluator = MOTEvaluator(
        dataset_name=dataset_name,
        device=device,
        checkpoint_path=checkpoint_path,
        conf_threshold=conf,
    )

    # Build dataloader
    dataloader = create_dataloader(
        dataset_name=dataset_name,
        root=test_root,
        split='val',
        batch_size=1,
        img_size=config.get('img_size', 640),
        augment=False,
    )

    metrics = evaluator.evaluate(dataloader)

    if metrics:
        print(f"    MOTA={metrics.get('MOTA',0):.2f} MOTP={metrics.get('MOTP',0):.2f} "
              f"IDF1={metrics.get('IDF1',0):.2f} | "
              f"FP={metrics.get('FP',0)} FN={metrics.get('FN',0)} IDSW={metrics.get('IDSW',0)}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="HAMT Evaluation Pipeline")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pth)')
    parser.add_argument('--config', type=str, default='configs/dancetrack_full.yaml',
                        help='Path to training config YAML')
    parser.add_argument('--dataset', type=str, default='dancetrack',
                        help='Dataset name (dancetrack/mot17/mot20), or "all"')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--conf', type=float, default=0.3,
                        help='Confidence threshold')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON path')
    args = parser.parse_args()

    # Determine datasets
    if args.dataset == 'all':
        datasets = ['dancetrack', 'mot17', 'mot20']
    else:
        datasets = [args.dataset]

    # Output path
    if args.output:
        output_path = Path(args.output)
    else:
        ckpt_stem = Path(args.checkpoint).stem
        output_path = PROJECT_ROOT / "results" / f"eval_results_{ckpt_stem}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"HAMT Evaluation Pipeline")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Config:     {args.config}")
    print(f"  Datasets:   {datasets}")
    print(f"  Device:     {args.device}")
    print(f"  Conf:       {args.conf}")
    print(f"  Output:     {output_path}")
    print(f"{'='*60}")

    all_results = {}
    for ds_name in datasets:
        try:
            metrics = evaluate_single(ds_name, args.checkpoint, args.config,
                                      args.device, args.conf)
            if metrics:
                all_results[ds_name] = metrics
        except Exception as e:
            print(f"  ✗ {ds_name} evaluation failed: {e}")

    # Save
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Evaluation complete. Results saved to: {output_path}")
    print(f"{'='*60}")

    # Summary table
    if all_results:
        print(f"\n{'Dataset':<15} {'MOTA':>8} {'MOTP':>8} {'IDF1':>8} {'Prec':>8} {'Rec':>8} {'IDSW':>6}")
        print(f"{'-'*60}")
        for ds_name, m in sorted(all_results.items()):
            print(f"  {ds_name:<13} {m.get('MOTA',0):>8.2f} {m.get('MOTP',0):>8.2f} "
                  f"{m.get('IDF1',0):>8.2f} {m.get('Precision',0):>8.2f} "
                  f"{m.get('Recall',0):>8.2f} {m.get('IDSW',0):>6d}")


if __name__ == "__main__":
    main()
