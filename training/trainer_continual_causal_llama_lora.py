import os
import json
import math
import shutil
import numpy as np
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from collections import defaultdict
from tqdm import tqdm

from transformers import GenerationConfig
from transformers.trainer_pt_utils import get_parameter_names
from transformers.optimization import get_scheduler
from training.early_stopping import EarlyStopping

from mpeft import get_peft_model
from mpeft.tuners.lora.layer import LoraLayer

from utils.compute_metrics import compute_metrics
from utils.utilities import mahalanobis

import time
import logging

class ContinualTrainerMTL(nn.Module):
    # 初始化函数
    def __init__(self,
                 # 训练超参数
                 args,
                 # 主模型实例
                 model,
                 # 辅助编码器
                 query_encoder,
                 # 日志记录器
                 logger,
                 # 任务列表
                 task_list,
                 # PEFT配置
                 peft_config,
                 # 保存LoRA权重或其他适配器权重的目录
                 lora_save_dir,
                 # 标签列表
                 label_list=None,
                 # 早停耐心值
                 early_stopping_patience=-1,
                 # 分词器
                 tokenizer=None,
                 # 生成文本的最大目标长度
                 max_target_length=20,
                 # 学习率列表
                 learning_rate_list=None,
                 max_train_batches_per_epoch=None,
                 max_eval_batches=None,
                 max_final_test_batches=None,
                 lamda_1: float = 0.1,
                 lamda_2: float = 0.01,
                 orthogonal_threshold: float = 0.1,
                 lm_loss_weight: float = 1.0,
                 ):
        super(ContinualTrainerMTL, self).__init__()
        
        self.args = args
        self.seed = args.seed
        # 修改1
        # self.model = model.to(self.args.device)
        self.model = model
        # Quantized models loaded with device_map/accelerate cannot be moved with `.to(...)`.
        self.query_encoder = query_encoder
        self.logger = logger
        self.task_list = task_list
        self.num_tasks = len(task_list)
        self.peft_config = peft_config
        
        # 存储训练轮数
        self.num_train_epochs = math.ceil(args.num_train_epochs)
        # 存储早停耐心值
        self.early_stopping_patience = early_stopping_patience
        # 初始化早停对象
        self.early_stopping = EarlyStopping(
            save_path=os.path.join(args.output_dir, 'best_checkpoint'),
            logger=self.logger,
            patience = early_stopping_patience
        )
        self.tokenizer = tokenizer
        self.max_target_length=max_target_length
        self.metric = 'rougeL'
        self.max_train_batches_per_epoch = max_train_batches_per_epoch
        self.max_eval_batches = max_eval_batches
        self.max_final_test_batches = max_final_test_batches
        
        try:
            self.learning_rate_list = [float(x) for x in learning_rate_list.split('_')]
        except:
            self.learning_rate_list = [self.args.learning_rate for _ in range(self.num_tasks)]

        self.logger.info(f"***********************************seed: {self.seed}***********************************")
        self.logger.info(f"***********************************lr: {self.learning_rate_list}***********************************")
        self.logger.info(f"***********************************lr_scheduler_type: {self.args.lr_scheduler_type}***********************************")

        self.lamda_1 = lamda_1
        self.lamda_2 = lamda_2
        self.orthogonal_threshold = orthogonal_threshold # 存储新的阈值
        self.lm_loss_weight = lm_loss_weight
        self.current_adapter_name = None 
        self.previous_adapter_names = []
        self._logged_lora_discovery = set()

        self.logger.info("LoRA layers will be discovered dynamically when needed.")

        # --- OSL Integration: 配置单独的 OSL 损失日志文件 ---
        osl_logfile_path = os.path.join(args.output_dir, "osl_losses.log")
        self.osl_loss_logger = logging.getLogger("osl_loss_logger")
        self.osl_loss_logger.setLevel(logging.INFO)
        # 避免重复添加handler
        if not self.osl_loss_logger.handlers:
            file_handler = logging.FileHandler(osl_logfile_path)
            formatter = logging.Formatter('%(asctime)s - %(message)s')
            file_handler.setFormatter(formatter)
            self.osl_loss_logger.addHandler(file_handler)
            # 也可以选择不将 OSL 日志输出到控制台
            # stream_handler = logging.StreamHandler()
            # stream_handler.setFormatter(formatter)
            # self.osl_loss_logger.addHandler(stream_handler)
        self.osl_loss_logger.propagate = False # 避免将日志消息传递给根日志器
        self.logger.info(f"OSL losses will be logged to: {osl_logfile_path}")
    
    # 为训练配置优化器
    def _prepare_optimizer(self, task_id=None):
        # decay_parameters 是一个包含参数名称的列表，表示需要应用权重衰减的参数。
        decay_parameters = get_parameter_names(self.model, [nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if p.requires_grad and n in decay_parameters],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if p.requires_grad and n not in decay_parameters],
                "weight_decay": 0.0,
            },
        ]

        learning_rate = self.learning_rate_list[task_id] if self.learning_rate_list is not None else self.args.learning_rate
        optimizer_kwargs = {"lr": learning_rate, "betas": (self.args.adam_beta1, self.args.adam_beta2), "eps": self.args.adam_epsilon}
        # 初始化优化器
        self.optimizer = AdamW(optimizer_grouped_parameters, **optimizer_kwargs)
    
    # 为训练配置学习率调度器，支持动态调整学习率以优化模型收敛
    def _prepare_scheduler(self, num_training_steps, optimizer):
        self.lr_scheduler = get_scheduler(
            self.args.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
            num_training_steps=num_training_steps,
        )

    # 从数据加载器中获取一批数据，将其移动到指定设备
    def _process_data(self, loader, mode):
        try:
            data_batch = next(loader[1])
        except:
            loader[1] = iter(loader[0])
            data_batch = next(loader[1])
        
        # 将所有张量数据移动到正确的设备
        data_batch = {k: v.to(self.args.device) if isinstance(v, torch.Tensor) else v for k, v in data_batch.items()}

        return data_batch

    # 根据模式调用模型进行前向传播或文本生成
    def compute_loss(self, model, inputs, task, mode, query_embed=None, final=False):
        if mode == 'train':
            # 在训练模式下，调用模型进行前向传播
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                loss_mask=inputs["loss_mask"],
                labels=inputs["labels"],
                active_adapter=task,
                query_embed=query_embed,
                disentangle_modules=self.model.config.disentangle_modules,
                train=True,
                final=False,
            )
            return outputs
        else:
            # 配置生成参数
            generation_config = GenerationConfig(
                max_new_tokens=self.max_target_length,
                repetition_penalty=1.0,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=False,
                use_cache=False,
            )
            
            generated_tokens = model.generate(
                input_ids=inputs["input_ids"],
                active_adapter=task,
                query_embed=query_embed,
                disentangle_modules=self.model.config.disentangle_modules,
                generation_config=generation_config,
                train=False,
                final=final,
                attention_mask=inputs["attention_mask"],
            )
            return generated_tokens
    
    # 准备dataloader
    def _prepare_dataloaders(self, dataloaders):
        loader = {}
        batch_num = []
        for task in dataloaders.keys():
            loader[task] = [dataloaders[task], iter(dataloaders[task])]
            batch_num.append(len(dataloaders[task]))
        return loader, batch_num
    
    def _iter_lora_layers(self):
        for module_name, module in self.model.base_model.named_modules():
            if isinstance(module, LoraLayer):
                yield module_name, module

    def _get_lora_params_for_adapter(self, adapter_name: str, get_trainable_only: bool = False):
        all_params = []
        for lora_layer_name, lora_layer_module in self._iter_lora_layers():
            param_pair = {}
            if adapter_name in lora_layer_module.lora_A: # 线性层的 LoRA A/B 矩阵
                lora_A_weight = lora_layer_module.lora_A[adapter_name].weight
                lora_B_weight = lora_layer_module.lora_B[adapter_name].weight
                
                if not get_trainable_only or (get_trainable_only and lora_A_weight.requires_grad):
                    param_pair['lora_A'] = lora_A_weight
                    param_pair['lora_B'] = lora_B_weight
                    param_pair['layer_module'] = lora_layer_module # 存储模块引用，便于后续直接访问
            elif adapter_name in lora_layer_module.lora_embedding_A: # 嵌入层的 LoRA A/B 矩阵
                lora_embedding_A = lora_layer_module.lora_embedding_A[adapter_name]
                lora_embedding_B = lora_layer_module.lora_embedding_B[adapter_name]
                if not get_trainable_only or (get_trainable_only and lora_embedding_A.requires_grad):
                    param_pair['lora_embedding_A'] = lora_embedding_A
                    param_pair['lora_embedding_B'] = lora_embedding_B
                    param_pair['layer_module'] = lora_layer_module # 存储模块引用
            
            if param_pair: # 如果找到了参数对
                all_params.append(param_pair)
        discovery_key = (adapter_name, get_trainable_only)
        if discovery_key not in self._logged_lora_discovery:
            self.logger.info(
                f"Discovered {len(all_params)} LoRA parameter groups for adapter '{adapter_name}' "
                f"(trainable_only={get_trainable_only})."
            )
            self._logged_lora_discovery.add(discovery_key)
        return all_params

    def _calculate_orthogonal_loss(self, current_adapter_name: str, previous_adapter_names: List[str]):
        if not previous_adapter_names or self.lamda_1 == 0:
            return torch.tensor(0.0, device=self.model.device)

        orthogonal_loss = torch.tensor(0.0, device=self.model.device)
        
        current_lora_layer_params = self._get_lora_params_for_adapter(current_adapter_name, get_trainable_only=True)

        if not current_lora_layer_params:
            return orthogonal_loss

        prev_lora_params_by_layer = defaultdict(lambda: {'A': [], 'B': []})
        for prev_adapter_name in previous_adapter_names:
            prev_params_for_adapter = self._get_lora_params_for_adapter(prev_adapter_name, get_trainable_only=False)
            for param_entry in prev_params_for_adapter:
                layer_module_ref = param_entry['layer_module']
                if 'lora_A' in param_entry:
                    prev_lora_params_by_layer[layer_module_ref]['A'].append(param_entry['lora_A'])
                    prev_lora_params_by_layer[layer_module_ref]['B'].append(param_entry['lora_B'])
                elif 'lora_embedding_A' in param_entry:
                    prev_lora_params_by_layer[layer_module_ref]['A'].append(param_entry['lora_embedding_A'])
                    prev_lora_params_by_layer[layer_module_ref]['B'].append(param_entry['lora_embedding_B'])

        for current_param_entry in current_lora_layer_params:
            current_layer_module_ref = current_param_entry['layer_module']
            
            if current_layer_module_ref not in prev_lora_params_by_layer:
                continue
            
            prev_A_list_for_layer = prev_lora_params_by_layer[current_layer_module_ref]['A']
            prev_B_list_for_layer = prev_lora_params_by_layer[current_layer_module_ref]['B']

            if not prev_A_list_for_layer: 
                continue

            if 'lora_A' in current_param_entry:
                current_A = current_param_entry['lora_A']
                current_B = current_param_entry['lora_B']
                
                valid_prev_A_list = [p for p in prev_A_list_for_layer if p.shape[1] == current_A.shape[1]]
                valid_prev_B_list = [p for p in prev_B_list_for_layer if p.shape[0] == current_B.shape[0]]

                if not valid_prev_A_list or not valid_prev_B_list:
                    continue

                prev_A_stacked = torch.stack(valid_prev_A_list, dim=0)
                prev_B_stacked = torch.stack(valid_prev_B_list, dim=0)

                # --- 软正交性计算：只惩罚超出阈值的部分 ---
                # 计算点积的绝对值
                dot_products_A = torch.abs(torch.matmul(current_A.unsqueeze(0), prev_A_stacked.transpose(-1, -2)))
                dot_products_B = torch.abs(torch.matmul(current_B.T.unsqueeze(0), prev_B_stacked))
                
                # 惩罚超出阈值的部分
                orthogonal_loss += F.relu(dot_products_A - self.orthogonal_threshold).sum()
                orthogonal_loss += F.relu(dot_products_B - self.orthogonal_threshold).sum()
                # --- End 软正交性 ---
                
            elif 'lora_embedding_A' in current_param_entry:
                current_A = current_param_entry['lora_embedding_A']
                current_B = current_param_entry['lora_embedding_B']

                valid_prev_A_list = [p for p in prev_A_list_for_layer if p.shape[1] == current_A.shape[1]]
                valid_prev_B_list = [p for p in prev_B_list_for_layer if p.shape[0] == current_B.shape[0]]

                if not valid_prev_A_list or not valid_prev_B_list:
                    continue
                
                prev_A_stacked = torch.stack(valid_prev_A_list, dim=0)
                prev_B_stacked = torch.stack(valid_prev_B_list, dim=0)

                # --- 软正交性计算 ---
                dot_products_A = torch.abs(torch.matmul(current_A.unsqueeze(0), prev_A_stacked.transpose(-1, -2)))
                dot_products_B = torch.abs(torch.matmul(current_B.T.unsqueeze(0), prev_B_stacked))
                
                orthogonal_loss += F.relu(dot_products_A - self.orthogonal_threshold).sum()
                orthogonal_loss += F.relu(dot_products_B - self.orthogonal_threshold).sum()
                # --- End 软正交性 ---

        return orthogonal_loss

    # 在指定阶段保存模型的特定参数状态字典为.pt文件，并将任务相关元数据保存为JSON文件，同时记录日志
    def _save_model_checkpoint_info(self, task, task_id, stage, save_dir="snapshots"):
        """在指定阶段保存模型的状态字典和元数据。"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        checkpoint_dir = os.path.join(self.args.output_dir, save_dir, f"task_{task_id}_{task}_{stage}_{timestamp}")
        os.makedirs(checkpoint_dir, exist_ok=True)

        state_dict_to_save = {}
        for k, v in self.model.state_dict().items():
            if 'key_encoder' in k or 'lora' in k or 'lm_head' in k:
                state_dict_to_save[k] = v.detach().cpu()
        
        lora_adapters = list(self.model.peft_config.keys()) if hasattr(self.model, 'peft_config') else []
        lora_count = len(lora_adapters)
        
        key_encoder_info = {}
        if hasattr(self.model, 'key_encoder') and self.model.key_encoder is not None:
            keys = self.model.key_encoder.keys
            attn_weights_logs = []
            for weights in self.model.key_encoder.attn_weights:
                if weights is not None:
                    if isinstance(weights, torch.Tensor):
                        attn_weights_logs.append([round(w.item(), 4) for w in weights])
                    else: 
                        attn_weights_logs.append(weights)
                else:
                    attn_weights_logs.append(None)

            key_encoder_info = {
                'keys_shape': list(keys.shape),
                'keys_norm': round(keys.norm().item(), 4),
                'keys_mean': round(keys.mean().item(), 4),
                'is_orthogonal': torch.allclose(keys @ keys.T, torch.eye(keys.shape[0]).to(keys.device), atol=1e-4) if keys.shape[0] > 1 else True,
                'attn_weights': attn_weights_logs,
                'task_count': self.model.key_encoder.task_count,
            }
        
        checkpoint_info = {
            'task_id': task_id,
            'task_name': task,
            'stage': stage,
            'timestamp': timestamp,
            'lora_count': lora_count,
            'lora_adapters': lora_adapters,
            'key_encoder': key_encoder_info,
            'state_dict_keys': list(state_dict_to_save.keys()),
        }

        json_path = os.path.join(checkpoint_dir, 'checkpoint_info.json')
        with open(json_path, 'w') as f:
            json.dump(checkpoint_info, f, indent=2, ensure_ascii=False)

        torch.save(state_dict_to_save, os.path.join(checkpoint_dir, 'state_dict.pt'))
        self.logger.info(f"Saved checkpoint info for {task} at {stage} to {json_path}")
    
    def train(self, train_dataloaders, val_dataloaders, test_dataloaders):
        # 准备训练、验证、测试数据加载器
        train_loader, train_batch = self._prepare_dataloaders(train_dataloaders)
        val_loader, val_batch = self._prepare_dataloaders(val_dataloaders)
        test_loader, test_batch = self._prepare_dataloaders(test_dataloaders)
        
        if self.args.do_train:
            for task_id, task in enumerate(self.task_list):
                self.current_adapter_name = task
                if task_id == 0 or self.model.config.multi_peft_modules:
                    # self.model.config.multi_peft_modules: True
                    # 配置LoRA适配器
                    # model: PeftModelForCausalLM
                    self.model = get_peft_model(self.model, self.peft_config, adapter_name=task)
                    # 打印可训练参数
                    self.model.print_trainable_parameters()
                # 激活任务适配器
                self.model.set_adapter(adapter_name=task)
                
                # Save model state at training start
                # self._save_model_checkpoint_info(task, task_id, stage="train_start")

                # 获取训练批次数量
                num_train_batch = train_batch[task_id]
                if self.max_train_batches_per_epoch:
                    num_train_batch = min(num_train_batch, self.max_train_batches_per_epoch)
                # 计算数据加载器长度
                len_dataloader = num_train_batch
                # 计算每轮更新步数同时确保更新步数至少为1
                num_update_steps_per_epoch = max(len_dataloader // self.args.gradient_accumulation_steps, 1)
                # 计算总优化步数
                num_training_steps_all_epochs = math.ceil(self.num_train_epochs * num_update_steps_per_epoch)
                
                # 准备优化器
                self._prepare_optimizer(task_id)
                # 准备学习率调度器
                self._prepare_scheduler(num_training_steps_all_epochs, self.optimizer)

                # 设置模型总轮次
                self.model.epochs = self.num_train_epochs
                # 设置模型任务ID
                self.model.task_id = task_id
                # 重置早停状态
                self.early_stopping.reinit()
                # 初始化验证结果字典
                eval_res = defaultdict(list)

                self.logger.info(f"***** 开始训练 - 任务 {task_id}: {task} *****")
                self.logger.info(f"  总轮次 (Num Epochs) = {self.num_train_epochs}")
                self.logger.info(f"  单设备批大小 (Batch size per device) = {self.args.per_device_train_batch_size}")
                self.logger.info(f"  梯度累积步数 (Gradient Accumulation steps) = {self.args.gradient_accumulation_steps}")
                self.logger.info(f"  总优化步数 (Total optimization steps) = {num_training_steps_all_epochs}")
                self.logger.info(f"  每轮训练批上限 (Max train batches/epoch) = {self.max_train_batches_per_epoch}")
                self.logger.info(f"  验证批上限 (Max eval batches) = {self.max_eval_batches}")
                self.logger.info(f"  最终测试批上限 (Max final test batches) = {self.max_final_test_batches}")
                self.logger.info(f"  OSL lamda_1 (Orthogonal Loss weight) = {self.lamda_1}")
                self.logger.info(f"  OSL lamda_2 (L2 Loss weight) = {self.lamda_2}")
                self.logger.info(f"  OSL Orthogonal Threshold = {self.orthogonal_threshold}") # 记录新的阈值
                
                for epoch in range(self.num_train_epochs):
                    self.model.epoch = epoch
                    self.model.train()

                    # 梯度清零
                    self.model.zero_grad()

                    for batch_index in tqdm(range(num_train_batch), desc=f"轮次 {epoch+1}/{self.num_train_epochs}"):
                        # 从train_leader[task]获取一批数据，转换为适合训练的格式，返回输入train_input
                        train_input = self._process_data(train_loader[task], mode='train')

                        """修改开始
                        # 获取 train_input 的内容
                        print("train_input keys:", train_input.keys())
                        for key, value in train_input.items():
                            if isinstance(value, torch.Tensor):
                                print(f"{key}: shape {value.shape}")
                            else:
                                print(f"{key}: {value}")
            
                        # 获取解码后的输入文本
                        if "input_ids" in train_input:
                            decoded_inputs = self.tokenizer.batch_decode(
                                train_input["input_ids"], 
                                skip_special_tokens=True, 
                                clean_up_tokenization_spaces=True
                            )
                            print("Decoded inputs:", decoded_inputs)
                        修改结束"""

                        # 获取任务嵌入
                        query_embed = self._get_query_embed(train_input)
                        
                        # 获取模型输出
                        outputs = self.compute_loss(self.model, train_input, task, mode='train', query_embed=query_embed)
                        
                        # --- OSL Integration: 添加 OSL 正则化损失 ---
                        base_loss = outputs.loss if isinstance(outputs, dict) else outputs[0]
                        
                        orthogonal_loss = self._calculate_orthogonal_loss(self.current_adapter_name, self.previous_adapter_names)
                        
                        l2_loss = torch.tensor(0.0, device=self.model.device)
                        if self.lamda_2 > 0:
                            current_trainable_lora_params = self._get_lora_params_for_adapter(self.current_adapter_name, get_trainable_only=True)
                            for param_group in current_trainable_lora_params:
                                for param_name, param_value in param_group.items():
                                    if param_name == "layer_module":
                                        continue
                                    l2_loss = l2_loss + torch.norm(param_value, p=2)

                        total_loss = base_loss * self.lm_loss_weight + orthogonal_loss * self.lamda_1 + l2_loss * self.lamda_2
                        
                        # 记录所有损失项，方便调试和分析
                        # self.logger.info(f"orthogonal_loss: {orthogonal_loss.item():.4f}; l2_loss: {l2_loss.item():.4f}; base_loss: {base_loss.item():.4f}; total_loss: {total_loss.item():.4f}; λ1: {self.lamda_1}; λ2: {self.lamda_2}")
                        
                        # --- OSL Integration: 记录 OSL 损失到单独的日志文件 ---
                        self.osl_loss_logger.info(
                            f"Task {task_id} ({self.current_adapter_name}), Epoch {epoch+1}, Batch {batch_index+1}: "
                            f"orthogonal_loss={orthogonal_loss.item():.6f}, l2_loss={l2_loss.item():.6f}, "
                            f"weighted_orthogonal_loss={(orthogonal_loss * self.lamda_1).item():.6f}, weighted_l2_loss={(l2_loss * self.lamda_2).item():.6f}, "
                            f"base_loss={base_loss.item():.6f}, lm_loss_weight={self.lm_loss_weight:.6f}, total_loss={total_loss.item():.6f}"
                        )
                        # --- End OSL Integration ---
                        
                        train_loss = total_loss
                        # --- End OSL Integration 损失 ---

                        # 缩放损失
                        train_loss = train_loss / self.args.gradient_accumulation_steps
                        
                        # 反向传播
                        train_loss.backward()
                        # 检查是否执行优化步骤
                        if (batch_index + 1) % self.args.gradient_accumulation_steps == 0:
                            # 更新参数
                            self.optimizer.step()
                            # 更新学习率
                            self.lr_scheduler.step()
                            # 梯度清零
                            self.optimizer.zero_grad()
                    # Training-only mode: skip per-epoch validation / early stopping.
                    self.clean_gpu_memory()

                # Save model state at the end of training for the current task.
                self._save_model_checkpoint_info(task, task_id, stage="train_end")

                if self.model.config.multi_peft_modules and not self.model.config.disentangle_modules and hasattr(self.model, 'key_encoder'):
                    self.model.key_encoder.process_task_count()
                
                self.previous_adapter_names.append(self.current_adapter_name)

        if os.path.isdir(self.early_stopping.save_path):
            shutil.rmtree(self.early_stopping.save_path)

    # 根据配置的查询编码器类型，从模型输入中生成查询嵌入
    def _get_query_embed(self, model_inputs):
        if self.query_encoder is None:
            return None
        input_ids = model_inputs['query_input_ids']
        attention_mask = model_inputs['query_attention_mask']
        with torch.no_grad():
            # 获取 hidden_states，表示每个 token 的嵌入向量
            hidden_states = self.query_encoder(input_ids, attention_mask=attention_mask)[0]
        
        # 对hidden_states沿着维度1取平均值，生成查询嵌入。
        if self.model.config.query_encoder_type == 'avg_all_embed':
            return hidden_states.mean(dim=1)
        # 计算仅对有效token的加权平均嵌入。
        elif self.model.config.query_encoder_type == 'avg_word_embed':
            masked_sum = torch.sum(hidden_states * attention_mask.unsqueeze(-1), dim=1)
            num_tokens = torch.sum(attention_mask, dim=1).unsqueeze(-1)
            # 添加一个小的 epsilon 防止除以零
            return masked_sum / (num_tokens + 1e-9)
        return None
    
    # 对指定任务的数据进行批量预测，生成文本，计算评估指标，记录平均结果并返回。
    def eval(self, loader, batch, task, mode='test', final=False):
        self.model.eval()
        self.clean_gpu_memory()
        if final:
            batch_cap = self.max_final_test_batches
        else:
            batch_cap = self.max_eval_batches
        if batch_cap:
            batch = min(batch, batch_cap)
        results = defaultdict(list)
        with torch.no_grad():
            for _ in tqdm(range(batch), desc=f"评估 {task} ({mode})"):
                self.clean_gpu_memory()
                input_data = self._process_data(loader, mode)

                """修改开始
                # 获取 input_data 的内容
                print("input_data keys:", input_data.keys())
                for key, value in input_data.items():
                    if isinstance(value, torch.Tensor):
                        print(f"{key}: shape {value.shape}")
                    else:
                        print(f"{key}: {value}")
            
                # 获取解码后的输入文本
                if "input_ids" in input_data:
                    decoded_inputs = self.tokenizer.batch_decode(
                        input_data["input_ids"], 
                        skip_special_tokens=True, 
                        clean_up_tokenization_spaces=True
                    )
                    print("Decoded inputs:", decoded_inputs)
                修改结束"""

                query_embed = self._get_query_embed(input_data)
                # 生成预测 token 序列
                generated_tokens = self.compute_loss(self.model, input_data, task, mode=mode, query_embed=query_embed, final=final)
                # 获取 input_data 的 targets 字段
                references = input_data["targets"]
                # 将输入的长度传递给后处理函数
                input_len = input_data["input_ids"].shape[1]
                # 后处理生成 token, 转换为文本预测
                predictions = self._postprocess_generated_tokens(generated_tokens, input_len)
                # 解码生成参考文本
                decoded_references = self.tokenizer.batch_decode(references, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                
                metrics = compute_metrics(predictions, decoded_references)
                for key, value in metrics.items():
                    results[key].append(value)
        
        if final:
            self.logger.info(f"***** 任务 {task} - 最终评估: {mode} *****")
        else:
            self.logger.info(f"***** 任务 {task} - 轮次 {self.model.epoch+1}: {mode} (seed {self.seed}) *****")
        
        # 计算并记录每个指标的平均值
        avg_results = {}
        for key, value in results.items(): 
            avg_value = round(sum(value)/len(value), 4)
            avg_results[key] = avg_value
            self.logger.info(f"  {key}: {avg_value}")
        
        return avg_results

    # 后处理模型生成的 token 序列，将其转换为可读的文本
    def _postprocess_generated_tokens(self, generated_tokens, input_length):
        # 转换为 CPU 和 numpy 数组以便操作
        if isinstance(generated_tokens, torch.Tensor):
            generated_tokens = generated_tokens.cpu().numpy()
        # 从 generated_tokens 中提取生成部分，排除输入部分
        generated_part = generated_tokens[:, input_length:]
        # 将 generated_part 中值为-100的 token 替换为 self.tokenizer.pad_token_id
        generated_part = np.where(generated_part == -100, self.tokenizer.pad_token_id, generated_part)
        
        # 将处理后的 token 序列解码为可读的文本字符串
        predictions = self.tokenizer.batch_decode(
            generated_part, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=True
        )
        # 移除每条文本首尾的空白字符，并返回结果
        return [pred.strip() for pred in predictions]

    # 记录最终测试结果，包括即时测试和最终测试的平均指标，并打印任务顺序
    def _log_final_results(self, test_loader, test_results, fwd_pass_results=None, key='rougeL'):
        self.logger.info(f"********************** 最终 {key.upper()} 结果 (seed: {self.seed}) **********************")
        
        # 获取测试数据加载器中所有任务的名称
        task_names = list(test_loader.keys())
        self.logger.info(f"任务顺序: {task_names}")

        if fwd_pass_results:
            self.logger.info(f"即时测试 {key}: {fwd_pass_results}")
            # 计算即使测试结果的平均值，并保留4位小数
            avg_fwd = round(sum(fwd_pass_results)/len(fwd_pass_results), 4)
            self.logger.info(f"平均即时测试 {key}: {avg_fwd}")
        
        self.logger.info(f"最终测试 {key} (所有训练后): {test_results}")
        # 计算最终测试结果的平均值，并保留4位小数
        avg_final = round(sum(test_results)/len(test_results), 4)
        self.logger.info(f"平均最终测试 {key}: {avg_final}")
        self.logger.info("******************************************************************")

    # 清理gpu内存
    def clean_gpu_memory(self):
        """清理GPU缓存的工具函数"""
        torch.cuda.empty_cache()
    
    # --- 省略了与 MPEFT 框架的 epizodic memory 相关的函数，它们在此次修改中保持不变 ---
    # _initialize_epi_variables, _statistics_task_mean_cov, _get_epi_ids,等函数无需修改
