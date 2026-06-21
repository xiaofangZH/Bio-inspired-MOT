#!/bin/bash
# HAMT 快速启动脚本

cd /home/user/MOT项目/HAMT

echo ""
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║         HAMT 训练和评估系统 - 快速启动                     ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""

# 检查GPU
python3 << 'EOF'
import torch
if torch.cuda.is_available():
    print(f"✓ GPU 可用: {torch.cuda.get_device_name(0)}")
    print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("⚠ GPU 不可用，将使用CPU (可能很慢)")
EOF

echo ""
echo "选择操作:"
echo "1. 完整训练和评估 (所有数据集)"
echo "2. 仅训练"
echo "3. 仅评估"
echo "0. 退出"
echo ""
read -p "请选择 (0-3): " choice

case $choice in
    1)
        read -p "输入训练轮次 (默认: 50): " epochs
        epochs=${epochs:-50}
        python3 run_pipeline.py --epochs $epochs
        ;;
    2)
        read -p "输入训练轮次 (默认: 50): " epochs
        epochs=${epochs:-50}
        read -p "选择数据集 (1=dancetrack, 2=mot17, 3=mot20): " dataset_choice
        case $dataset_choice in
            1) dataset="dancetrack" ;;
            2) dataset="mot17" ;;
            3) dataset="mot20" ;;
            *) dataset="dancetrack" ;;
        esac
        python3 train.py --dataset $dataset --epochs $epochs
        ;;
    3)
        read -p "选择数据集 (1=dancetrack, 2=mot17, 3=mot20): " dataset_choice
        case $dataset_choice in
            1) dataset="dancetrack" ;;
            2) dataset="mot17" ;;
            3) dataset="mot20" ;;
            *) dataset="dancetrack" ;;
        esac
        python3 eval.py --dataset $dataset
        ;;
    0)
        echo "退出"
        exit 0
        ;;
    *)
        echo "无效选择"
        ;;
esac
