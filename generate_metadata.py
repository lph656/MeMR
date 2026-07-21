import json
import torch
from transformers import LlamaTokenizer, LlamaModel, BitsAndBytesConfig
import os
import numpy as np

# --- 1. 配置 ---
# 元数据JSON文件的路径
# 请确保您的merged_metadata.json文件位于这个路径下
METADATA_JSON_PATH = "./metadata_embeddings/merged_metadata.json"
# 生成的元数据嵌入文件的保存路径
META_EMBEDDINGS_SAVE_PATH = "./metadata_embeddings/keshi_meta_embeddings.pt"

# 脚本中指定的科室顺序 (保持拼音，因为 script.sh 依赖它)
TASK_LIST_ORDER_PINYIN = ["neike", "waike", "erke", "fuchanke", "nanke", "zhongliuke"]
#TASK_LIST_ORDER_PINYIN = ["neike", "waike", "erke", "fuchanke"]


# 用于编码元数据的预训练大模型路径
LLAMA_MODEL_PATH = "chinese-alpaca-plus-7b-hf"

# --- MDTM 改进：新增拼音到中文的映射 ---
# 这个映射用于将 TASK_LIST_ORDER_PINYIN 中的拼音名称
# 转换为 merged_metadata.json 文件中的中文键名
PINYIN_TO_CHINESE_MAP = {
    "neike": "内科",
    "waike": "外科",
    "erke": "儿科",
    "fuchanke": "妇产科",
    "nanke": "男科",
    "zhongliuke": "肿瘤科",
}
# --- MDTM 改进结束 ---

# 2. 加载 Llama Tokenizer 和 Llama Model
print(f"Loading Llama Tokenizer and Model from: {LLAMA_MODEL_PATH}")

tokenizer = LlamaTokenizer.from_pretrained(LLAMA_MODEL_PATH, add_prefix_space=True)
# 确保特殊 token ID 设置正确，与训练时保持一致
tokenizer.bos_token_id = 1
tokenizer.eos_token_id = 2
tokenizer.pad_token_id = 1 # Llama 通常使用 BOS 或 EOS 作为 PAD

# 提前定义 BitsAndBytesConfig (与训练时保持一致，以防模型加载有量化需求)
compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
)

# 加载 LlamaModel (注意这里是 LlamaModel，而不是 LlamaForCausalLM)
# 并且将其冻结，只用于特征提取
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LlamaModel.from_pretrained(
    LLAMA_MODEL_PATH,
    quantization_config=quantization_config,
    device_map="auto", # 自动分配到 GPU
    torch_dtype=compute_dtype,
    trust_remote_code=True
)
# 冻结模型参数
for param in model.parameters():
    param.requires_grad = False
model.eval() # 设置为评估模式

# 获取模型输出的嵌入维度 (即 hidden_size)，这将是 key_dim
KEY_DIM = model.config.hidden_size
print(f"LlamaModel loaded. Embedding dimension (key_dim): {KEY_DIM}")
print(f"Ensure that KeyEncoderConfig.key_dim is set to {KEY_DIM} in your training script.")


# 3. 加载元数据 JSON 文件
print(f"Loading metadata from: {METADATA_JSON_PATH}")
if not os.path.exists(METADATA_JSON_PATH):
    raise FileNotFoundError(f"Metadata JSON file not found at {METADATA_JSON_PATH}")

# full_metadata 的键将是 "儿科", "内科", "外科" 等中文名称
with open(METADATA_JSON_PATH, 'r', encoding='utf-8') as f:
    # 假设 JSON 文件顶级结构是一个列表，每个元素是科室对象
    # 例如：[{"department": "儿科", ...}, {"department": "内科", ...}]
    # 我们需要将其转换为一个以中文科室名为键的字典，以便按名称查找
    loaded_data_list = json.load(f)
    full_metadata_dict = {item['department']: item for item in loaded_data_list}


# 4. 按照指定顺序处理和编码元数据
all_meta_embeddings = []
num_processed_tasks = 0

print("Processing and encoding metadata for each department...")
# 循环使用拼音名称，通过映射查找对应的中文名称
for pinyin_task_name in TASK_LIST_ORDER_PINYIN:
    # --- MDTM 改进：使用映射转换科室名称 ---
    chinese_task_name = PINYIN_TO_CHINESE_MAP.get(pinyin_task_name)
    if chinese_task_name is None:
        raise ValueError(f"No Chinese mapping found for pinyin task name: '{pinyin_task_name}'. "
                         f"Please update PINYIN_TO_CHINESE_MAP in the script.")

    if chinese_task_name not in full_metadata_dict: # 使用字典查找
        raise ValueError(f"Metadata for task '{chinese_task_name}' (pinyin: '{pinyin_task_name}') "
                         f"is missing in the JSON file at '{METADATA_JSON_PATH}'. "
                         f"Please ensure all tasks in PINYIN_TO_CHINESE_MAP are present in the JSON.")
    # --- MDTM 改进结束 ---
        
    metadata = full_metadata_dict[chinese_task_name] # 使用转换后的中文名称查找元数据
    
    # 拼接元数据文本
    keywords_str = "。关键词：" + "，".join(metadata['keywords']) if metadata.get('keywords') else ""
    symptoms_str = "。症状：" + "，".join(metadata['symptoms']) if metadata.get('symptoms') else ""
    
    text_to_encode = f"{metadata['description']}{keywords_str}{symptoms_str}"
    
    print(f"  Encoding '{chinese_task_name}' (pinyin: '{pinyin_task_name}'): {text_to_encode[:100]}...")

    inputs = tokenizer(
        text_to_encode, 
        return_tensors="pt", 
        padding='longest', 
        truncation=True, 
        max_length=2048, # 假设元数据文本长度不超过512
        add_special_tokens=True 
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden_state = outputs.last_hidden_state
        attention_mask = inputs['attention_mask']

        masked_embeddings = last_hidden_state * attention_mask.unsqueeze(-1) 
        sum_embeddings = torch.sum(masked_embeddings, dim=1) 
        num_tokens = torch.sum(attention_mask, dim=1).unsqueeze(-1) 
        
        mean_pooling_embeddings = sum_embeddings / torch.clamp(num_tokens, min=1e-9)
    
    if mean_pooling_embeddings.shape[1] != KEY_DIM:
        raise ValueError(f"Embedding dimension mismatch for task '{chinese_task_name}'. "
                         f"Expected {KEY_DIM}, got {mean_pooling_embeddings.shape[1]}. Check LlamaModel config.")
    
    all_meta_embeddings.append(mean_pooling_embeddings.squeeze(0).cpu())

    num_processed_tasks += 1

if num_processed_tasks != len(TASK_LIST_ORDER_PINYIN):
    print(f"Error: Number of processed tasks ({num_processed_tasks}) does not match expected tasks in order list ({len(TASK_LIST_ORDER_PINYIN)}).")
    exit()

# 5. 堆叠所有向量并保存
final_meta_embeddings = torch.stack(all_meta_embeddings)
print(f"Successfully generated final meta embeddings with shape: {final_meta_embeddings.shape}")

os.makedirs(os.path.dirname(META_EMBEDDINGS_SAVE_PATH), exist_ok=True)
torch.save(final_meta_embeddings, META_EMBEDDINGS_SAVE_PATH)
print(f"Meta embeddings saved to: {META_EMBEDDINGS_SAVE_PATH}")
