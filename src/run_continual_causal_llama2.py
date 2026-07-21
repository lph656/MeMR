import os
import sys
sys.path.append(".")
sys.path.append("../")
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_dir)
sys.path.insert(0, project_root)

import nltk
import numpy as np
from evaluate import load

import json
import shutil
from transformers import LlamaTokenizer, HfArgumentParser, TrainingArguments, set_seed, AutoConfig
from mpeft import LoraConfig, KeyEncoderConfig, TaskType
from arguments import DataTrainingArguments, ModelArguments, OSLArguments
import torch

# 修改4
# from model.modeling_llama import LlamaModel
from transformers import BitsAndBytesConfig, LlamaModel

from utils.set_logger import set_logger
from model.causal_lm_llama import LlamaContinualForCausalLM
from training.trainer_continual_causal_llama_lora import ContinualTrainerMTL
from tasks.mtl5.dataloader_mtl_causal_llama import DataLoaderMTL

if __name__ == "__main__":
    
    ### 1. 加载所有需要的配置
    # 解析命令行参数，分为三类，将解析结果存储在training_args, data_args, model_args中
    parser = HfArgumentParser((TrainingArguments, DataTrainingArguments, ModelArguments, OSLArguments))
    training_args, data_args, model_args, osl_args = parser.parse_args_into_dataclasses()
    
    # 设置种子
    seed = training_args.seed
    set_seed(seed)
    
    # overwrite_output_dir should reset prior artifacts for the current run.
    if training_args.overwrite_output_dir and os.path.isdir(training_args.output_dir):
        shutil.rmtree(training_args.output_dir)

    # 创建输出文件夹
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir, exist_ok=True)
    lora_output_dir = os.path.join(training_args.output_dir, 'lora')
    if not os.path.exists(lora_output_dir):
        os.mkdir(lora_output_dir)
    
    # 加载peft和mpeft配置
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=4, lora_alpha=16, lora_dropout=0.1, bias="none", 
        output_dir=lora_output_dir,
        target_modules=[
            "q_proj",
            "v_proj",
        ],
        mpeft_enabled=model_args.mpeft_enabled,
        )
    # if model_args.mpeft_enabled:
        # --- MDTM 改进：将 meta_embeddings_path 和 key_dim 传递给 mpeft_config ---
    #    mpeft_config = KeyEncoderConfig(
    #        seed=seed,
    #        query_encoder_type=model_args.query_encoder_type,
    #        task_list=data_args.task_list.split('_'),
    #        meta_embeddings_path=model_args.meta_embeddings_path, # MDTM 新增参数
    #        key_dim=model.config.hidden_size, # 确保 key_dim 与 LlamaModel 的 hidden_size 匹配
    #        )
    #    # --- MDTM 改进结束 ---
    #    peft_config.mpeft_config = mpeft_config
    
    logfile = os.path.join(training_args.output_dir, "log.txt")
    logger = set_logger(logfile)
    
    # 创建配置路径并保存配置
    config_path = os.path.join(training_args.output_dir, f"configs.json")
    with open(config_path, "w", newline='\n') as f:
        f.write(f"\n(m)peft_args:\n {peft_config}\n")
        f.write(f"\ntraining_args:\n {training_args}\n")
        f.write(f"\nosl_args:\n {osl_args}\n")

    ### 2. 加载数据集
    task_list = data_args.task_list.split('_')
    
    # 加载Llama2分词器
    model_name_or_path = model_args.model_name_or_path
    tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path,
                                            add_prefix_space=True,
                                            padding_side='left',  # 左填充
                                            trust_remote_code=True  # 支持某些自定义模型的tokenizer
                                            )
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2
    tokenizer.pad_token_id = 1
    
    if data_args.max_seq_length_list is not None:
        max_seq_length = [int(sql) for sql in data_args.max_seq_length_list.split('_')]
    else:
        max_seq_length = data_args.max_seq_length
    max_target_length = data_args.max_target_length

    # --- 提前定义 BitsAndBytesConfig ---
    # 如果 GPU 不支持 bf16，可以使用 torch.float16
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    logger.info(f"Using compute dtype: {compute_dtype}")

    # 创建量化配置
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",      # NF4 是推荐的4位量化得具体类型或算法
        bnb_4bit_compute_dtype=compute_dtype,
    )
    logger.info(f"Using quantization config: {quantization_config}")

    # 加载多任务数据加载器
    dataloaders = DataLoaderMTL(
        data_args=data_args,
        training_args=training_args,
        task_list=task_list,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        overwrite_cache=data_args.overwrite_cache,
    )
    train_dataloaders = {task: dataloaders[task]['train'] for task in task_list}
    dev_dataloaders = {task: dataloaders[task]['dev'] for task in task_list}
    test_dataloaders = {task: dataloaders[task]['dev'] for task in task_list}
    
    
    ### 3. 加载基模型
    # 修改2
    """
    model = LlamaContinualForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name_or_path,
        device_map="auto",
        offload_folder="offload",
        trust_remote_code=True
        )
    """
    model = LlamaContinualForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name_or_path,
        quantization_config=quantization_config, # <<< --- 应用预定义的量化配置
        device_map="auto",
        offload_folder="offload",
        trust_remote_code=True,
    )
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # 将 model_args 对象的非私有属性且不是可调用对象（例如方法）的属性值复制到 model.config 对象中
    for arg in dir(model_args):
        if not arg.startswith("__") and not callable(getattr(model_args, arg)):
            setattr(model.config, arg, getattr(model_args, arg))

    # 初始化 query encoder
    # 修改3
    """
    query_encoder = LlamaModel(model.config)
    for param in query_encoder.parameters():
        param.requires_grad = False
    """
    query_encoder = None
    if model_args.mpeft_enabled:
        logger.info(f"Loading query encoder (LlamaModel) from {model_name_or_path} with quantization...")
        try:
            query_encoder = LlamaModel.from_pretrained(
                model_name_or_path,
                quantization_config=quantization_config, # <<< --- 应用相同的量化配置
                device_map="auto",                       # <<< --- 尝试直接加载到 GPU
                offload_folder="offload_query",          # <<< --- 使用不同的卸载文件夹
                torch_dtype=compute_dtype,               # <<< --- 保持 dtype 一致
                trust_remote_code=True
            )

            # 冻结 query_encoder 的参数
            for param in query_encoder.parameters():
                param.requires_grad = False
            logger.info("Froze query encoder parameters.")

        except Exception as e:
            logger.error(f"Failed to load quantized query encoder (LlamaModel): {e}")
            raise e
    else:
        logger.info("MPEFT not enabled, skipping query encoder loading.")

    # --- MDTM 改进：mpeft_config 的实例化和赋值移动到 model 定义之后 ---
    if model_args.mpeft_enabled:
        mpeft_config = KeyEncoderConfig(
            seed=seed,
            query_encoder_type=model_args.query_encoder_type,
            task_list=data_args.task_list.split('_'),
            meta_embeddings_path=model_args.meta_embeddings_path,
            key_dim=model.config.hidden_size, # <--- 现在 model 已经定义，可以访问它的配置了
            matching_loss_v2=model_args.matching_loss_v2,
            matching_loss_coeff=model_args.matching_loss_coeff,
            ortho_loss_key=osl_args.atr_enable_key_ortho,
            ortho_loss_coeff=osl_args.atr_key_ortho_coeff,
            key_ortho_mode=osl_args.atr_key_ortho_mode,
            key_ortho_threshold=osl_args.atr_key_ortho_threshold,
            lamda_keys_L2=osl_args.atr_key_l2_lambda,
            )
        peft_config.mpeft_config = mpeft_config
        # 此时mpeft_config已完整，可以重新记录peft_config到文件
        # 注意：这会覆盖之前的peft_config记录，如果需要累加，请调整文件写入模式
        with open(config_path, "a", newline='\n') as f:
            f.write(f"\n(m)peft_args_after_model_init:\n {peft_config}\n")
    # --- ----------------------------------------------------------- ---

    ### 4. 设置训练器
    # 初始化多任务持续学习训练器
        trainer = ContinualTrainerMTL(
        args=training_args,
        model=model,
        query_encoder=query_encoder,
        logger=logger,
        task_list=task_list,
        label_list=None, # <--- 显式传递None
        peft_config=peft_config,
        lora_save_dir=os.path.join(training_args.output_dir, 'checkpoint_loras'),
        early_stopping_patience=data_args.early_stopping_patience if data_args.early_stop else -1,
        tokenizer=tokenizer,
        max_target_length = data_args.max_target_length,
        learning_rate_list=data_args.learning_rate_list,
        max_train_batches_per_epoch=data_args.max_train_batches_per_epoch,
        max_eval_batches=data_args.max_eval_batches,
        max_final_test_batches=data_args.max_final_test_batches,
        lamda_1=osl_args.lamda_1,
        lamda_2=osl_args.lamda_2,
        orthogonal_threshold=osl_args.orthogonal_threshold,
        lm_loss_weight=osl_args.lm_loss_weight,
    )
    
    ### 5. 开始训练
    trainer.train(
        train_dataloaders,
        dev_dataloaders,
        test_dataloaders,
    )
