#!/bin/bash

# 设置脚本在遇到错误时立即退出
set -e

# --- 配置 ---
# 数据集的基础目录
BASE_DATA_DIR="datasets/medical_consult"
# 生成答案的输出目录
OUTPUT_DIR="test_infer/generate_answer"
# 训练完成后的MeMR检查点目录，需包含 checkpoint_info.json 和 state_dict.pt
CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"
# 基座模型路径
BASE_MODEL_PATH="${BASE_MODEL_PATH:-chinese-alpaca-plus-7b-hf}"
# 科室元数据嵌入路径
META_EMBEDDINGS_PATH="${META_EMBEDDINGS_PATH:-metadata_embeddings/keshi_meta_embeddings.pt}"
# 定义要处理的科室列表
DEPARTMENTS=("neike" "waike" "erke" "fuchanke" "nanke" "zhongliuke")
# Python脚本的名称
PYTHON_SCRIPT="test_infer/generate_answer.py"

# --- 主逻辑 ---

# 检查Python脚本是否存在
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "错误: Python脚本 '$PYTHON_SCRIPT' 未找到。"
    echo "请确保此脚本与 'generate_answer.py' 在同一目录下。"
    exit 1
fi

echo "开始批量生成医疗问答..."
echo "========================================"

if [ -z "$CHECKPOINT_DIR" ]; then
    echo "错误: 请通过环境变量 CHECKPOINT_DIR 指定检查点目录。"
    echo "示例: CHECKPOINT_DIR=checkpoints/... bash test_infer/generate_answer.sh"
    exit 1
fi

# 创建输出目录（如果不存在）
mkdir -p "$OUTPUT_DIR"
echo "输出目录 '$OUTPUT_DIR' 已准备就绪。"

# 循环处理每个科室的数据集
for dept in "${DEPARTMENTS[@]}"; do
    INPUT_FILE="$BASE_DATA_DIR/$dept/test.json"
    OUTPUT_FILE="$OUTPUT_DIR/$dept.json"

    echo ""
    echo "--- 开始处理科室: $dept ---"
    
    # 检查输入文件是否存在
    if [ ! -f "$INPUT_FILE" ]; then
        echo "警告: 未找到输入文件 '$INPUT_FILE'，跳过此科室。"
        continue
    fi
    
    echo "输入文件: $INPUT_FILE"
    echo "输出文件: $OUTPUT_FILE"

    # 执行Python脚本进行推理和生成
    python "$PYTHON_SCRIPT" \
        --input_path "$INPUT_FILE" \
        --output_path "$OUTPUT_FILE" \
        --checkpoint_dir "$CHECKPOINT_DIR" \
        --base_model_path "$BASE_MODEL_PATH" \
        --meta_embeddings_path "$META_EMBEDDINGS_PATH"
    
    echo "--- 科室 '$dept' 处理完成 ---"
done

echo ""
echo "========================================"
echo "🎉 所有数据集处理完毕！"
echo "生成的结果已保存在 '$OUTPUT_DIR' 目录下。"
