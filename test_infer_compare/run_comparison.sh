#!/bin/bash

# 文件夹A的路径
FOLDER_A="test_infer_compare/generate_answer_v1" 

# 文件夹B的路径
FOLDER_B="test_infer_compare/generate_answer_contin_finetuning" 

# DeepSeek API Key must be provided via environment variable.
if [ -z "$DEEPSEEK_API_KEY" ]; then
  echo "Error: DEEPSEEK_API_KEY is not set."
  echo "Example: export DEEPSEEK_API_KEY=\"<your_deepseek_api_key>\""
  exit 1
fi

# --- 调用Python脚本 ---
echo "正在运行医疗问答对比脚本..."
echo "文件夹A: $FOLDER_A"
echo "文件夹B: $FOLDER_B"

# 执行Python脚本，并将配置作为命令行参数传递
python3 test_infer_compare/compare_medical_qa.py "$FOLDER_A" "$FOLDER_B" "$DEEPSEEK_API_KEY"

echo "脚本执行完毕。"
