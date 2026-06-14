#!/usr/bin/env python3
"""完整的训练和评估流程"""

from train import train_on_dataset
from eval import evaluate_on_dataset
import json
from datetime import datetime

class Pipeline:
    def __init__(self):
        self.datasets = {
            'dancetrack': '/home/user/MOT项目/HAMT/data/DanceTrack/train1',
            'mot17': '/home/user/MOT项目/HAMT/data/MOT17/train',
            'mot20': '/home/user/MOT项目/HAMT/data/MOT20/train'
        }
        self.test_paths = {
            'dancetrack': '/home/user/test',
            'mot17': '/home/user/MOT17/MOT17/test',
            'mot20': '/home/user/MOT20/MOT20/test'
        }
        self.results = {}
    
    def run_all(self, epochs=50):
        print("=" * 70)
        print("开始完整的训练和评估流程")
        print("=" * 70)
        
        for dataset_name in ['dancetrack', 'mot17', 'mot20']:
            print(f"\n处理数据集: {dataset_name.upper()}")
            
            # 训练
            try:
                trainer = train_on_dataset(dataset_name, self.datasets[dataset_name], epochs)
                print(f"[✓] {dataset_name} 训练完成")
            except Exception as e:
                print(f"[✗] {dataset_name} 训练失败: {e}")
                continue
            
            # 评估
            try:
                metrics = evaluate_on_dataset(dataset_name, self.test_paths[dataset_name])
                self.results[dataset_name] = metrics
                print(f"[✓] {dataset_name} 评估完成")
            except Exception as e:
                print(f"[✗] {dataset_name} 评估失败: {e}")
        
        self.print_summary()
    
    def print_summary(self):
        print("\n" + "=" * 70)
        print("总结")
        print("=" * 70)
        for dataset, metrics in self.results.items():
            print(f"{dataset.upper()}: {metrics}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()
    
    pipeline = Pipeline()
    pipeline.run_all(args.epochs)
