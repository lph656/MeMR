""" Utility classes and functions related to MoCL (NAACL 2024).
Copyright (c) 2024 Robert Bosch GmbH

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.
You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter

# --- MDTM 新增代码：DynamicAttentionLayer 类 ---
class DynamicAttentionLayer(nn.Module):
    def __init__(self, key_dim):
        super().__init__()
        self.key_dim = key_dim
        
    def forward(self, E_query, V_meta_k, E_task_k):
        # E_query: [batch_size, key_dim] (问诊编码)
        # V_meta_k: [num_tasks, key_dim] (所有科室的元数据向量，静态)
        # E_task_k: [num_tasks, key_dim] (可训练的科室嵌入)

        device = E_task_k.device
        target_dtype = E_task_k.dtype
        E_query = E_query.to(device).to(target_dtype)
        V_meta_k = V_meta_k.to(device).to(target_dtype) # 确保 V_meta_k 也是目标 dtype

        norm_query = nn.functional.normalize(E_query, dim=-1)
        norm_meta_k = nn.functional.normalize(V_meta_k, dim=-1)

        attn_scores = torch.einsum('bd,td->bt', norm_query, norm_meta_k) 
        
        alpha_k = torch.sigmoid(attn_scores)
        
        alpha_k_expanded = alpha_k.unsqueeze(-1)

        E_task_k_expanded = E_task_k.unsqueeze(0).expand(E_query.shape[0], -1, -1)
        V_meta_k_expanded = V_meta_k.unsqueeze(0).expand(E_query.shape[0], -1, -1)

        E_dynamic_k = alpha_k_expanded * V_meta_k_expanded + (1 - alpha_k_expanded) * E_task_k_expanded
        
        return E_dynamic_k # [batch_size, num_tasks, key_dim]

# --- MDTM 新增代码结束 ---

class TaskKeyEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.seed = config.seed
        
        self.task_list = config.task_list
        self.num_tasks = len(self.task_list)
        self.task_count = 0

        self.steps = {t_id: 0 for t_id in range(self.num_tasks)}
        self.steps_val = {t_id: 0 for t_id in range(self.num_tasks)}
        self.steps_final = {t_id: 0 for t_id in range(self.num_tasks)}
        
        # --- MDTM 改进点 1.1：加载所有科室的元数据向量 ---
        if self.config.meta_embeddings_path is None:
            raise ValueError("config.meta_embeddings_path must be provided for MDTM initialization and dynamic matching.")
        if not os.path.exists(self.config.meta_embeddings_path):
            raise FileNotFoundError(f"Metadata embeddings file not found at {self.config.meta_embeddings_path}")
        
        loaded_meta_keys = torch.load(self.config.meta_embeddings_path, map_location="cpu")
        assert loaded_meta_keys.shape[0] == self.num_tasks, \
            f"Loaded meta_embeddings count {loaded_meta_keys.shape[0]} does not match num_tasks {self.num_tasks}."
        assert loaded_meta_keys.shape[1] == self.config.key_dim, \
            f"Loaded meta_embeddings dim {loaded_meta_keys.shape[1]} does not match config.key_dim {self.config.key_dim}."
        self.register_buffer("all_meta_keys", loaded_meta_keys)
        print(f"Loaded {self.num_tasks} metadata embeddings from {self.config.meta_embeddings_path} with dim {self.config.key_dim}")
        # --- ----------------------------------------- ---

        # --- MDTM 改进点 2.1：实例化动态注意力层 ---
        self.dynamic_attn_layer = DynamicAttentionLayer(self.config.key_dim)
        # --- ---------------------------------- ---
        
        # --- OSL Integration: 定义新的内部超参数 ---
        # 任务键的软正交阈值，默认值可根据实验调整
        self.key_ortho_threshold = getattr(config, 'key_ortho_threshold', 0.1) 
        # 任务键的 L2 正则化权重，默认值可根据实验调整
        self.lamda_keys_L2 = getattr(config, 'lamda_keys_L2', 0.005) 
        print(f"TaskKeyEncoder - Key Ortho Threshold: {self.key_ortho_threshold}, Key L2 Lambda: {self.lamda_keys_L2}")
        # --- End OSL Integration ---

        self._initialize_keys()
        if hasattr(self.config, 'bnb_4bit_compute_dtype') and self.config.bnb_4bit_compute_dtype is not None:
             self.keys.data = self.keys.data.to(self.config.bnb_4bit_compute_dtype)
             # 同时，由于 self.all_meta_keys 是静态加载的，确保它也具有相同的 dtype
             self.all_meta_keys = self.all_meta_keys.to(self.config.bnb_4bit_compute_dtype)
             print(f"TaskKeyEncoder keys and meta_keys converted to {self.config.bnb_4bit_compute_dtype}")
        
        self.log_weights = True
        tb_log_dir = os.path.join(self.config.output_dir, f"tensorboard_seed{self.seed}")
        if not os.path.exists(tb_log_dir):
            os.makedirs(tb_log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=tb_log_dir)
        
    def forward(self, x_query, adapter_name, train=True, final=False):
        task_id = self.task_list.index(adapter_name)
        
        if train and not final:
            self.steps[task_id] += 1
        elif not train and not final:
            self.steps_val[task_id] += 1
        else:
            self.steps_final[task_id] += 1
        
        batch_size = x_query.shape[0]
        loss = torch.tensor(0.0, device=x_query.device)
        
        # --- MDTM 改进点 2.2：动态匹配增强 ---
        # 1. 获取当前任务及之前所有任务的 V_meta_k 和 E_task_k
        target_device = x_query.device
        target_dtype = x_query.dtype
        K_meta = self.all_meta_keys[:task_id+1].to(device=target_device, dtype=target_dtype)
        K_task = self.keys[:task_id+1].to(device=target_device, dtype=target_dtype)
        
        # 2. 调用动态注意力层生成动态科室嵌入 E_dynamic_k
        E_dynamic_k = self.dynamic_attn_layer(x_query, K_meta, K_task)

        # 3. 计算查询向量 x_query 与动态科室嵌入 E_dynamic_k 的匹配分数 (余弦相似度)
        n_q = nn.functional.normalize(x_query.to(E_dynamic_k.dtype), dim=-1).detach().unsqueeze(1) 
        n_dynamic_k = nn.functional.normalize(E_dynamic_k, dim=-1)

        cos_sim = torch.bmm(n_q, n_dynamic_k.transpose(1, 2)).squeeze(1)
        
        # 保留原有的 keys_mask 逻辑
        for i in range(cos_sim.shape[-1]): 
            if i < len(self.keys_mask) and self.keys_mask[i]: 
                cos_sim = cos_sim.clone()
                cos_sim[:, i] = -1e8
        # --- ------------------------------- ---

        w = nn.functional.softmax(cos_sim*self.config.softmax_match_scale, dim=-1) if not self.config.direct_compose else cos_sim
        
        if train:
            loss, ortho_loss_key, matching_loss, l2_loss_keys = self._add_kp_losses(K_task, w, cos_sim, loss, batch_size, task_id, x_query)
        
        if not train and not final:
            w_avg = torch.mean(w.detach().clone(), dim=0)
            if self.attn_weights[task_id] is None:
                self.attn_weights[task_id] = w_avg
            else:
                self.attn_weights[task_id] = ((self.attn_weights[task_id] * self.steps_val[task_id]) + w_avg) / (self.steps_val[task_id]+1)
        
        self._log_weights(w, task_id, train, final)
                    
        if train:
            self._log_losses(task_id, ortho_loss_key, matching_loss, l2_loss_keys)
            
        return w, loss
    
                
    def _initialize_keys(self):
        self.attn_weights = [None for _ in range(self.num_tasks)]
        
        # --- MDTM 改进点 1.2：用元数据向量初始化 self.keys (E_task_k) ---
        self.keys = nn.Parameter(self.all_meta_keys.clone().detach(), requires_grad=True) 
        print(f"Task keys (E_task_k) initialized from loaded metadata embeddings.")
        # --- -------------------------------------------- ---

        self.keys_mask = [False for _ in range(self.num_tasks)]
        
        # --- MDTM 改进点 1.3：根据元数据初始化跳过或调整其他初始化方式 ---
        if self.config.key_init_func == 'uniform':
            print("Uniform initialization skipped as metadata init is used.")
        
        elif self.config.key_init_func == 'orthogonal':
            self.keys = self.gram_schmidt(self.keys)
            print("Task keys (E_task_k) further orthogonalized after metadata initialization.")

        # Key-level orthogonality is an ATR training option and should not depend on
        # whether we additionally orthogonalize the initialization.
        self.ortho_loss_key = self.config.ortho_loss_key
        self.ortho_loss_coeff = self.config.ortho_loss_coeff
        

    def _add_kp_losses(self, K_task_for_loss, w, cos_sim, loss, batch_size, task_id, x_query):
        ortho_loss_key = 0 
        if self.ortho_loss_key:
            # OSL Integration: 使用新的 soft_ortho_penalty
            ortho_loss_key_val = self.ortho_loss_coeff * self.ortho_penalty(K_task_for_loss)
            loss = loss + ortho_loss_key_val
            ortho_loss_key = ortho_loss_key_val.item() 
        
        # OSL Integration: 添加任务键的 L2 正则化损失
        l2_loss_keys = 0
        if self.lamda_keys_L2 > 0:
            l2_loss_keys_val = self.lamda_keys_L2 * torch.norm(self.keys, p=2)
            loss = loss + l2_loss_keys_val
            l2_loss_keys = l2_loss_keys_val.item()

        matching_loss = 0
        if self.config.matching_loss:
            matching_loss_val = (1.0-w[:, -1]).mean() * self.config.matching_loss_coeff
            loss = loss + matching_loss_val
            matching_loss = matching_loss_val.item()
            
        if self.config.matching_loss_v2: 
            matching_loss_val = (1.0-cos_sim[:, -1]).mean() * self.config.matching_loss_coeff
            loss = loss + matching_loss_val
            matching_loss = matching_loss_val.item()
            
        if self.config.matching_loss_cls:
            gt_w = torch.zeros_like(w, device=w.device) 
            gt_w[:, task_id] = 1.
            gt_w = gt_w[:, :cos_sim.shape[-1]] 
            
            matching_loss_val = torch.abs(gt_w - w).mean() * self.config.matching_loss_coeff
            loss = loss + matching_loss_val
            matching_loss = matching_loss_val.item()
                
        if self.config.matching_loss_cls_all:
            gt_w = torch.zeros_like(w, device=w.device)
            gt_w[:, task_id] = 1.
            
            matching_loss_val = torch.abs(gt_w - w).mean() * self.config.matching_loss_coeff
            loss = loss + matching_loss_val
            matching_loss = matching_loss_val.item()
            
        return loss, ortho_loss_key, matching_loss, l2_loss_keys # OSL Integration: 返回 l2_loss_keys
    
    def _log_weights(self, w, task_id, train, final):
        if self.log_weights:
            w_avg = torch.mean(w.detach().clone(), dim=0)
            
            for t in range(w_avg.shape[-1]):
                if train and not final:
                    step = self.steps[task_id]
                    log_name = f'training_weight_{task_id}/{t}'
                elif not train and not final:
                    step = self.steps_val[task_id]
                    log_name = f'validation_weight_{task_id}/{t}'
                elif final:
                    step = self.steps_final[task_id]
                    log_name = f'final_weight_{task_id}/{t}'

                self.writer.add_scalar(log_name, w_avg[t], step)

    # OSL Integration: 修改 _log_losses 的签名以接收 L2 损失
    def _log_losses(self, task_id, ortho_loss_key, matching_loss, l2_loss_keys):
        if self.ortho_loss_key and not isinstance(ortho_loss_key, (tuple, list)): 
            self.writer.add_scalar(f'ortho_loss/task_{task_id}', ortho_loss_key, self.steps[task_id])
        if self.config.matching_loss or self.config.matching_loss_v2 or self.config.matching_loss_cls or self.config.matching_loss_cls_all:
            self.writer.add_scalar(f'matching_loss/task_{task_id}', matching_loss, self.steps[task_id])
        # OSL Integration: 记录 L2 损失
        if self.lamda_keys_L2 > 0:
            self.writer.add_scalar(f'l2_keys_loss/task_{task_id}', l2_loss_keys, self.steps[task_id])    
    # 对输入张量执行正交化，生成一组正交且归一化的向量。
    # 正交化
    # 归一化
    def gram_schmidt(self, vv):

        def projection(u, v):
            denominator = (u * u).sum()

            if denominator < 1e-8:
                return None
            else:
                return (v * u).sum() / denominator * u
        
        is_nd = len(vv.shape) > 2
        if is_nd:
            shape_nd = copy.deepcopy(vv.shape)
            vv = vv.view(vv.shape[0], -1)
            
        vv = vv.T

        nk = vv.size(1)
        uu = torch.zeros_like(vv, device=vv.device)
        
        s = self.task_count
        if s>0:
            uu[:, 0:s] = vv[:, 0:s].clone()
            
        for k in range(s, nk):
            redo = True
            while redo:
                redo = False
                vk = torch.randn_like(vv[:,k]).to(vv.device)
                uk = 0
                for j in range(0, k):
                    if not redo:
                        uj = uu[:, j].clone()
                        proj = projection(uj, vk)
                        if proj is None:
                            redo = True
                            print('restarting!')
                        else:
                            uk = uk + proj
                if not redo:
                    uu[:, k] = vk - uk
                    
        for k in range(s, nk):
            uk = uu[:, k].clone()
            uu[:, k] = uk / uk.norm()
            
        uu = uu.T 
        
        if is_nd:
            uu = uu.view(shape_nd)
        
        return nn.Parameter(uu) 
    
    
    # OSL Integration: 修改 ortho_penalty 以实现软正交
    def ortho_penalty(self, t):
        mode = getattr(self.config, "key_ortho_mode", "unnormalized_soft")
        if mode == "cosine_soft":
            t_for_gram = F.normalize(t, dim=-1)
            gram_matrix = t_for_gram @ t_for_gram.T
            off_diagonal_elements = gram_matrix - torch.eye(t.shape[0], dtype=t.dtype, device=t.device)
            mask = ~torch.eye(t.shape[0], dtype=torch.bool, device=t.device)
            abs_off_diagonal = torch.abs(off_diagonal_elements[mask])
            if self.key_ortho_threshold > 0:
                return F.relu(abs_off_diagonal - self.key_ortho_threshold).mean()
            return abs_off_diagonal.mean()

        gram_matrix = t @ t.T
        off_diagonal_elements = gram_matrix - torch.eye(t.shape[0], dtype=t.dtype, device=t.device)
        mask = ~torch.eye(t.shape[0], dtype=torch.bool, device=t.device)
        abs_off_diagonal = torch.abs(off_diagonal_elements[mask])

        if mode == "hard":
            return abs_off_diagonal.mean()

        if self.key_ortho_threshold > 0:
            return F.relu(abs_off_diagonal - self.key_ortho_threshold).mean()
        return abs_off_diagonal.mean()
    
    def process_task_count(self):
        self.task_count += 1
        
        if self.config.key_init_func == 'orthogonal':
            self.keys = self.gram_schmidt(self.keys)
            print(f"Task keys orthogonalized after adding task {self.task_count-1}.")
