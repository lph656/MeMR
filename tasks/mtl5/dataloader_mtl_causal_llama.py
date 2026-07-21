"""
为医疗指令跟随任务的持续学习提供一个自定义数据加载器
"""

import os
from datasets import load_dataset, Dataset
import torch
from torch.utils.data import DataLoader
from transformers import default_data_collator, DataCollatorWithPadding

# 指令微调的模板。这有助于模型理解任务。
PROMPT_TEMPLATE = "你是一位专业的AI医生，请根据患者的提问给出建议。\n\n患者提问：{instruction}\n\nAI医生回答："


def custom_data_collator(features, tokenizer):
    """
    一个自定义的数据整理器，它能正确处理 'targets' 字段。
    """
    # 从特征中分离出 'targets' 字段
    targets_list = [f.pop("targets", None) for f in features]

    # 这会正确地填充 input_ids, attention_mask, labels 等字段
    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    batch = data_collator(features)

    # 将之前分离出的 'targets' 字段添加回批次中
    batch['targets'] = targets_list

    # 如果 loss_mask 字段已经是tensor，则不需要额外处理。如果不是，需要手动堆叠。
    if "loss_mask" in features[0] and isinstance(features[0]["loss_mask"], list):
         loss_masks = [f["loss_mask"] for f in features]
         batch['loss_mask'] = torch.tensor(loss_masks, dtype=torch.long)

    return batch


def preprocess_function(examples, tokenizer, max_seq_length, is_eval=False):
    """
    为医疗问诊指令微调数据进行预处理。
    这将把 instruction, output 格式化为 "prompt -> response" 的形式,
    并创建正确的 loss_mask ，确保模型只学习预测 response 部分。
    """
    instructions = examples['instruction']
    outputs = examples['output']
    
    sources = []
    targets = []

    # 格式化提示和回答文本
    for instruction, output in zip(instructions, outputs):
        # 使用 PROMPT_TEMPLATE 格式化指令，生成提示文本
        source = PROMPT_TEMPLATE.format(instruction=instruction)
        # 格式化回答文本，在开头添加一个空格，并在结尾添加分词器的结束标记
        target = f" {output}{tokenizer.eos_token}"
        # 将格式化后的提示和回答添加到 sources 和 targets 列表
        sources.append(source)
        targets.append(target)

    # 对 sources 和 targets 列表中的文本进行分词，生成 token ID。
    tokenized_sources = tokenizer(sources, add_special_tokens=False)
    tokenized_targets = tokenizer(targets, add_special_tokens=False)

    all_input_ids = [] # 训练时：起始符+问题文本的token ID+回答文本的token ID；验证时：起始符+问题文本的token ID
    all_loss_masks = [] # 训练时：问题部分的mask为0，回答部分的mask为1；验证时：全部为0
    all_labels_for_eval = [] # 存储了未经填充和截断的、原始的答案序列的token ID
    all_padded_labels = [] # 存储了用于模型计算损失的目标标签
    all_query_input_ids = [] # 存储了专为MPEFT的 Query Encoder 准备的输入序列的 token ID
    all_query_attention_masks = [] # 存储了与 all_query_input_ids 对应的注意力掩码

    for source_ids, target_ids in zip(tokenized_sources['input_ids'], tokenized_targets['input_ids']):
        if is_eval:
            # 验证集：input_ids 只包含提示部分
            input_ids = [tokenizer.bos_token_id] + source_ids
            labels = target_ids  # 验证集的 labels 直接使用 target_ids
        else:
            # 训练集：input_ids 包含提示 + 回答
            input_ids = [tokenizer.bos_token_id] + source_ids + target_ids
            labels = input_ids  # 训练集的 labels 与 input_ids 相同
        
        # 创建损失掩码，提示部分为0，回答部分为1
        source_len = len(source_ids) + 1
        loss_mask = [0] * source_len + [1] * len(target_ids) if not is_eval else [0] * len(input_ids)
        
        # 截断与填充
        if len(input_ids) > max_seq_length:
            # 截断时从右侧截断，保留开头的 token
            input_ids = input_ids[:max_seq_length]
            loss_mask = loss_mask[:max_seq_length]
        else:
            padding_length = max_seq_length - len(input_ids)
            # 将填充符加在序列的左边
            input_ids = [tokenizer.pad_token_id] * padding_length + input_ids
            # loss_mask 同样在左边填充 0
            loss_mask = [0] * padding_length + loss_mask
            
        # 填充或截断 labels（验证集和训练集均需统一长度）
        if len(labels) > max_seq_length:
            labels = labels[:max_seq_length]
        else:
            padding_length = max_seq_length - len(labels)
            # 将填充符加在 labels 序列的左边
            labels = [tokenizer.pad_token_id] * padding_length + labels
        
        all_input_ids.append(input_ids)
        all_loss_masks.append(loss_mask)
        all_labels_for_eval.append(target_ids)
        all_padded_labels.append(labels)

        # --- 【新增】为 Query Encoder 准备输入 (无论训练或评估，都只用 source) ---
        query_ids = [tokenizer.bos_token_id] + source_ids
        # 截断与填充
        if len(query_ids) > max_seq_length:
            query_ids = query_ids[:max_seq_length]
        else:
            padding_length = max_seq_length - len(query_ids)
            query_ids = [tokenizer.pad_token_id] * padding_length + query_ids
        
        # 创建注意力掩码
        query_attention_mask = [0 if token_id == tokenizer.pad_token_id else 1 for token_id in query_ids]
        
        all_query_input_ids.append(query_ids)
        all_query_attention_masks.append(query_attention_mask)

    # 准备模型输入字典
    model_inputs = {
        'input_ids': torch.tensor(all_input_ids, dtype=torch.long),
        'attention_mask': torch.tensor(
            [[0] * sum(1 for x in ids if x == tokenizer.pad_token_id) + [1] * (len(ids) - sum(1 for x in ids if x == tokenizer.pad_token_id)) for ids in all_input_ids],
            dtype=torch.long
        ),
        'labels': torch.tensor(all_padded_labels, dtype=torch.long),
        'loss_mask': torch.tensor(all_loss_masks, dtype=torch.long),
        'targets': all_labels_for_eval,
        # 【新增】将 query_encoder 的专用输入添加到批次中
        'query_input_ids': torch.tensor(all_query_input_ids, dtype=torch.long),
        'query_attention_mask': torch.tensor(all_query_attention_masks, dtype=torch.long),
    }

    return model_inputs


def DataLoaderMTL(
    data_args,
    training_args,
    task_list,
    tokenizer,
    max_seq_length=None,
    overwrite_cache=False,
):
    """
    为持续学习场景创建一系列的 DataLoader。
    这个函数会加载每个任务的数据，进行训练/验证集划分，然后应用预处理。
    """
    # 数据集根目录
    dataset_root_dir = "datasets/medical_consult"
    
    # 验证集划分比例
    validation_split_percentage = data_args.validation_split_percentage if hasattr(data_args, 'validation_split_percentage') else 0.1

    # 初始化 dataloader 字典
    dataloaders = {}

    # 设置分词器的填充方向为左填充
    tokenizer.padding_side = 'left'
    # 如果分词器没有定义pad_token，则将eos_token作为填充token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # custom_collator_fn 是一个“绑定了 tokenizer 的 custom_data_collator 函数”的快捷方式
    # 它将 tokenizer 作为固定的参数嵌入到函数中，使得后续调用 custom_collator_fn 时，只需传入 features 即可
    custom_collator_fn = lambda features: custom_data_collator(features, tokenizer)

    for task in task_list:
        print(f"--- Processing task: {task} ---")
        
        # 构建文件路径并加载数据集
        train_file_path = os.path.join(dataset_root_dir, task, 'train.json')
        if not os.path.exists(train_file_path):
            raise FileNotFoundError(f"Data file not found for task '{task}' at path: {train_file_path}")

        raw_dataset = load_dataset('json', data_files=train_file_path, split='train')

        # 划分训练集和验证集
        split_dataset = raw_dataset.train_test_split(test_size=validation_split_percentage, seed=training_args.seed)
        train_dataset = split_dataset['train']
        eval_dataset = split_dataset['test']
        print(f"Task '{task}': {len(train_dataset)} training samples, {len(eval_dataset)} validation samples.")

        train_dataset = train_dataset.map(
            lambda examples: preprocess_function(examples, tokenizer, max_seq_length, is_eval=False),   # 预处理函数
            batched=True,   # 启用批量处理
            remove_columns=raw_dataset.column_names, # 移除原始数据集的列（如 instruction 和 output ）,因为预处理函数会生成新的列
            load_from_cache_file=not overwrite_cache,
            desc=f"Running tokenizer on {task} train dataset",  # 进度条的描述信息
        )
        
        eval_dataset = eval_dataset.map(
            lambda examples: preprocess_function(examples, tokenizer, max_seq_length, is_eval=True),
            batched=True,
            remove_columns=raw_dataset.column_names,
            load_from_cache_file=not overwrite_cache,
            desc=f"Running tokenizer on {task} eval dataset",
        )

        # 创建 DataLoader
        dataloaders[task] = {
            'train': DataLoader(
                train_dataset,  # 指定训练数据集
                batch_size=training_args.per_device_train_batch_size,   # 设置训练批次大小
                sampler=torch.utils.data.RandomSampler(train_dataset),  # 指定训练数据的采样器为随机采样器
                collate_fn=custom_collator_fn,  # 指定批次数据的整理函数为 custom_collator_fn
                drop_last=training_args.dataloader_drop_last,   # 控制是否丢弃最后一个不完整的批次
                num_workers=training_args.dataloader_num_workers,   # 设置数据加载的并行工作进程数
                pin_memory=training_args.dataloader_pin_memory, # 启用内存锁定，加速数据传输到GPU
            ),
            'dev': DataLoader(
                eval_dataset,
                batch_size=training_args.per_device_eval_batch_size,
                sampler=torch.utils.data.SequentialSampler(eval_dataset),
                collate_fn=custom_collator_fn,
                drop_last=False,
                num_workers=training_args.dataloader_num_workers,
                pin_memory=training_args.dataloader_pin_memory,
            )
        }
        
    print("--- All dataloaders created successfully! ---")
    return dataloaders