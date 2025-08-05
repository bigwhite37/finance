#!/bin/bash

echo "========================================================="
echo "强化学习投资策略回测脚本使用示例"
echo "========================================================="

# 检查是否存在模型文件
MODEL_PATH="models/final_model_20250805_083702.zip"
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ 模型文件不存在: $MODEL_PATH"
    echo "请先训练模型或使用其他可用的模型文件："
    find . -name "*.zip" -type f
    exit 1
fi

echo "✅ 使用模型: $MODEL_PATH"
echo ""

# 示例1: 基础回测
echo "示例1: 基础回测"
echo "命令: python run_backtest.py --model-path $MODEL_PATH"
echo "-------------------------------------------------------"
python run_backtest.py --model-path "$MODEL_PATH" --output-dir "results/example1_basic"
echo ""

# 示例2: 自定义测试期间
echo "示例2: 自定义测试期间"
echo "命令: python run_backtest.py --model-path $MODEL_PATH --test-start 2023-11-01 --test-end 2023-12-31"
echo "-------------------------------------------------------"
python run_backtest.py \
    --model-path "$MODEL_PATH" \
    --test-start "2023-11-01" \
    --test-end "2023-12-31" \
    --output-dir "results/example2_custom_period"
echo ""

# 示例3: 启用Qlib可视化
echo "示例3: 启用Qlib专业可视化"
echo "命令: python run_backtest.py --model-path $MODEL_PATH --enable-qlib-viz"
echo "-------------------------------------------------------"
python run_backtest.py \
    --model-path "$MODEL_PATH" \
    --enable-qlib-viz \
    --output-dir "results/example3_qlib_viz"
echo ""

# 示例4: 完整参数配置
echo "示例4: 完整参数配置"
echo "命令: python run_backtest.py --model-path $MODEL_PATH --initial-cash 2000000 --stocks 30 --enable-qlib-viz"
echo "-------------------------------------------------------"
python run_backtest.py \
    --model-path "$MODEL_PATH" \
    --initial-cash 2000000 \
    --stocks 30 \
    --enable-qlib-viz \
    --log-level DEBUG \
    --output-dir "results/example4_full_config"
echo ""

# 示例5: 仅生成报告，不绘图
echo "示例5: 仅生成报告，不绘图"
echo "命令: python run_backtest.py --model-path $MODEL_PATH --no-plot --no-export"
echo "-------------------------------------------------------"
python run_backtest.py \
    --model-path "$MODEL_PATH" \
    --no-plot \
    --no-export \
    --output-dir "results/example5_report_only"
echo ""

echo "========================================================="
echo "所有示例运行完成！"
echo "结果保存在 results/ 目录下的各个子目录中"
echo "========================================================="