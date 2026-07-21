#!/bin/bash

# 定义固定路径
INPUT_DIR="./test_infer_RAG/generate_answer_rag_v2"
OUTPUT_DIR="./test_infer_RAG/generate_scores_DS_v2" # 修改输出目录名称
PYTHON_SCRIPT="./test_infer_RAG/generate_score_DS.py" # 修改 Python 脚本名称

# 输入 JSON 文件列表
INPUT_JSONS=(
    "${INPUT_DIR}/neike.json"
    "${INPUT_DIR}/waike.json"
    "${INPUT_DIR}/erke.json"
    "${INPUT_DIR}/fuchanke.json"
    "${INPUT_DIR}/nanke.json"
    "${INPUT_DIR}/zhongliuke.json"
)

# 检查 Python 脚本存在
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "错误：Python 脚本 $PYTHON_SCRIPT 不存在"
    exit 1
fi

# 创建输出主目录
mkdir -p "$OUTPUT_DIR"

# 遍历输入 JSON 文件，生成评分
for INPUT_JSON in "${INPUT_JSONS[@]}"; do
    # 提取文件名（不含路径和扩展名）
    FILENAME=$(basename "$INPUT_JSON" .json)
    # 构造输出 JSON 路径
    OUTPUT_JSON="${OUTPUT_DIR}/${FILENAME}/${FILENAME}.json"

    # 检查输入文件存在
    if [ ! -f "$INPUT_JSON" ]; then
        echo "错误：输入文件 $INPUT_JSON 不存在"
        continue
    fi

    echo "处理 $INPUT_JSON，输出到 $OUTPUT_JSON"

    # 创建输出子目录
    mkdir -p "${OUTPUT_DIR}/${FILENAME}"

    # 调用 Python 脚本生成评分
    python3 "$PYTHON_SCRIPT" \
        --input_json "$INPUT_JSON" \
        --output_json "$OUTPUT_JSON"

    # 检查 Python 脚本执行状态
    if [ $? -ne 0 ]; then
        echo "错误：处理 $INPUT_JSON 失败"
    else
        echo "完成：评分生成，结果保存在 $OUTPUT_JSON"
    fi
done

# 自动计算所有科室的均分
echo "开始计算所有科室的评分均分..."
# 均分模式使用新的输出目录
python3 "$PYTHON_SCRIPT" --average --score_dir "$OUTPUT_DIR" 
if [ $? -ne 0 ]; then
    echo "错误：计算均分失败"
else
    echo "完成：均分已成功保存"
fi

echo "所有任务完成"
