#!/bin/bash
# HAMT 三阶段训练启动脚本
# 执行: bash /root/MOT项目/HAMT/start_training.sh

cd /root/MOT项目/HAMT

echo "============================================"
echo "HAMT 三阶段训练"
echo "Phase 1: CrowdHuman 检测筑基 (15 epochs)"
echo "Phase 2: MOT17+MOT20 运动关联 (20 epochs)"
echo "Phase 3: DanceTrack 极难域微调 (20 epochs)"
echo "============================================"
echo ""

python train_phase.py --yaml configs/three_phase.yaml --gpu 0
