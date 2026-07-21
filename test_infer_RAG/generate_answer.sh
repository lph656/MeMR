#!/bin/bash

# 设置脚本在遇到错误时立即退出
set -e

export CUDA_VISIBLE_DEVICES="0"

# --- 配置 ---
# 数据集的基础目录
BASE_DATA_DIR="datasets/medical_consult"
# 生成答案的输出目录
OUTPUT_DIR="test_infer_RAG/generate_answer_rag_v2" # 建议为RAG结果使用新目录
# 训练完成后的MeMR检查点目录，需包含 checkpoint_info.json 和 state_dict.pt
CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"
# 基座模型路径
BASE_MODEL_PATH="${BASE_MODEL_PATH:-chinese-alpaca-plus-7b-hf}"
# 科室元数据嵌入路径
META_EMBEDDINGS_PATH="${META_EMBEDDINGS_PATH:-metadata_embeddings/keshi_meta_embeddings.pt}"
# 定义要处理的科室列表
DEPARTMENTS=("neike" "waike" "erke" "fuchanke" "nanke" "zhongliuke")
#DEPARTMENTS=("neike" "waike" "erke" "fuchanke")

# --- RAG 配置 (新增) ---
# 知识库文件路径 (请确保此文件存在)
KNOWLEDGE_BASE_PATH="datasets/knowledge_base/normalized_knowledge_base.json"
# FAISS索引的保存目录
FAISS_INDEX_PATH="datasets/faiss_index_medical"
# 用于生成文本向量的嵌入模型
EMBEDDING_MODEL="shibing624/text2vec-base-chinese"

# --- 脚本路径配置 ---
INDEX_BUILDER_SCRIPT="test_infer_RAG/build_rag_index.py"
ANSWER_GENERATOR_SCRIPT="test_infer_RAG/generate_answer.py"

# --- 主逻辑 ---

# 检查Python脚本是否存在
if [ ! -f "$ANSWER_GENERATOR_SCRIPT" ]; then
    echo "错误: Python脚本 '$ANSWER_GENERATOR_SCRIPT' 未找到。"
    exit 1
fi
if [ ! -f "$INDEX_BUILDER_SCRIPT" ]; then
    echo "错误: RAG索引构建脚本 '$INDEX_BUILDER_SCRIPT' 未找到。"
    exit 1
fi

# --- RAG索引准备 (新增) ---
if [ -z "$CHECKPOINT_DIR" ]; then
    echo "错误: 请通过环境变量 CHECKPOINT_DIR 指定检查点目录。"
    echo "示例: CHECKPOINT_DIR=checkpoints/... bash test_infer_RAG/generate_answer.sh"
    exit 1
fi

echo "--- 检查RAG索引 ---"
if [ ! -d "$FAISS_INDEX_PATH" ]; then
    echo "警告: 未找到FAISS索引目录 '$FAISS_INDEX_PATH'。"
    echo "将开始自动构建索引，这可能需要几分钟时间..."

    if [ ! -f "$KNOWLEDGE_BASE_PATH" ]; then
        echo "错误: 知识库文件 '$KNOWLEDGE_BASE_PATH' 不存在，无法构建索引！"
        exit 1
    fi
    
    python "$INDEX_BUILDER_SCRIPT" \
        --kb_path "$KNOWLEDGE_BASE_PATH" \
        --index_path "$FAISS_INDEX_PATH" \
        --embedding_model "$EMBEDDING_MODEL"
    
    echo "✅ RAG索引构建完成。"
else
    echo "发现已存在的FAISS索引 '$FAISS_INDEX_PATH'，将直接使用。"
fi
echo "----------------------"


echo ""
echo "开始批量生成RAG增强的医疗问答..."
echo "========================================"

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

    # 执行Python脚本进行推理和生成 (已更新参数)
    python "$ANSWER_GENERATOR_SCRIPT" \
        --input_path "$INPUT_FILE" \
        --output_path "$OUTPUT_FILE" \
        --checkpoint_dir "$CHECKPOINT_DIR" \
        --index_path "$FAISS_INDEX_PATH" \
        --embedding_model "$EMBEDDING_MODEL" \
        --base_model_path "$BASE_MODEL_PATH" \
        --meta_embeddings_path "$META_EMBEDDINGS_PATH"
    
    echo "--- 科室 '$dept' 处理完成 ---"
done

echo ""
echo "========================================"
echo "🎉 所有数据集处理完毕！"
echo "基于RAG生成的结果已保存在 '$OUTPUT_DIR' 目录下。"
