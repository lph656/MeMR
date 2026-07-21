# Adapter Composition Implementation Audit

## 1. Executive Conclusion
- Actual module type: each paper task module `P_k` is implemented as one task-named LoRA adapter, concretely a per-target-layer pair of `lora_A[k]` and `lora_B[k]`.
- Actual composition level: the active path composes adapter output increments, not LoRA `A` matrices, not `B` matrices, and not precomputed `ΔW`.
- Modules included during training: when training task index `n` (0-based), routing and composition both use tasks `0..n`, including the current task.
- Modules included during inference: the standalone inference utility loads all learned adapters and routes over all adapters listed in the final checkpoint.
- Current-task module role: the current adapter is one term inside the same weighted sum as old adapters; there is no separate unweighted current-task branch in the active global-composition path.
- Whether old modules are frozen: Yes for LoRA parameters of inactive adapters.
- Whether old task vectors are frozen: No static evidence of freezing; old rows of `self.keys` remain trainable.
- Whether metadata embeddings are trainable: No. They are loaded as a buffer.
- Whether routing parameters are trainable: Yes, primarily `TaskKeyEncoder.keys`.
- Need for new experiments: No for clarifying the reported mechanism, but the manuscript formula and some implementation details must be corrected to match code. If any reported result used `utils/inference_utils.py`, that utility has a query-embedding mismatch relative to the trainer path and should be documented separately.

## 2. Code Entry Points

| Functionality | File | Class/Function | Lines | Description |
| --- | --- | --- | --- | --- |
| Training entry | `src/run_continual_causal_llama2.py` | `__main__` | 33-60, 109-146, 163-200, 205-228 | Builds LoRA config, quantized base model, quantized query encoder, and `KeyEncoderConfig`. |
| LoRA wrapper creation | `training/trainer_continual_causal_llama_lora.py` | `ContinualTrainerMTL.train` | 399-410 | Adds one named adapter per task via `get_peft_model(..., adapter_name=task)` and activates it with `set_adapter`. |
| Global key encoder install | `mpeft/peft_model.py` | `PeftModel.__init__` | 113-135 | Creates `TaskKeyEncoder` for global composition and attaches it to the base model. |
| LoRA module definition | `mpeft/tuners/lora/layer.py` | `LoraLayer.update_layer` | 77-113 | Creates `lora_A[adapter]`, `lora_B[adapter]`, dropout, and stored scaling. |
| Actual composition | `mpeft/tuners/lora/layer.py` | `Linear.forward` | 307-345 | Adds weighted LoRA outputs to the base linear output. |
| Routing weights | `model/key_encoder.py` | `TaskKeyEncoder.forward` | 112-165 | Computes task-visible routing weights from query embeddings, metadata, and trainable task keys. |
| LM forward | `model/causal_lm_llama.py` | `LlamaContinualForCausalLM.forward` | 108-165 | Calls `key_encoder`, passes `adapter_weights` to the decoder, and adds `match_loss` to LM loss. |
| Adapter weights propagation | `model/modeling_llama.py` | `LlamaSdpaAttention.forward`, `LlamaDecoderLayer.forward`, `LlamaModel.forward` | 668-746, 767-828, 980-1135 | Forwards the same `adapter_weights` through all decoder layers. |
| Optimizer scope | `training/trainer_continual_causal_llama_lora.py` | `_prepare_optimizer` | 131-149 | Optimizes only `model.named_parameters()` with `requires_grad=True`. |
| Query embedding build in trainer | `training/trainer_continual_causal_llama_lora.py` | `_get_query_embed` | 547-565 | Computes query embedding under `torch.no_grad()`, using averaged token embeddings. |
| Standalone inference restore | `utils/inference_utils.py` | `load_memr_model`, `generate_memr_response` | 43-121, 131-177 | Rebuilds all adapters from checkpoint metadata and performs generation. |

## 3. Definition of a Task-Specific Module
- Paper symbol: `P_k`
- Code object: one task-named LoRA adapter distributed across all targeted linear layers.
- Module type: task-specific LoRA parameter pairs `lora_A[k]` and `lora_B[k]`, whose forward contribution is an additive hidden-state increment.
- Parameter structure:
  - `mpeft/tuners/lora/layer.py:91-92` creates `self.lora_A[adapter_name] = nn.Linear(self.in_features, r, bias=False)` and `self.lora_B[adapter_name] = nn.Linear(r, self.out_features, bias=False)`.
  - `mpeft/tuners/lora/layer.py:324-331` applies those modules and adds the result to the base output.
- Mapping from manuscript to code:
  - `P_k` is not a standalone merged tensor object.
  - The closest faithful code-level mapping is: “the collection of all task-named LoRA submodules for adapter `k` across every targeted projection”.
  - In the active forward path, the object actually being aggregated is the output increment of each such adapter.

Evidence:

File: `mpeft/tuners/lora/layer.py`  
Function: `LoraLayer.update_layer`  
Lines: 82-97  
Snippet:
```python
self.lora_A[adapter_name] = nn.Linear(self.in_features, r, bias=False)
self.lora_B[adapter_name] = nn.Linear(r, self.out_features, bias=False)
self.scaling[adapter_name] = lora_alpha / r
```
Conclusion: each task adapter is represented by a LoRA A/B pair plus stored scaling and dropout.

File: `mpeft/tuners/lora/layer.py`  
Function: `Linear.forward`  
Lines: 324-331  
Snippet:
```python
for i, adapter in enumerate(self.lora_A.keys()):
    lora_A = self.lora_A[adapter]
    lora_B = self.lora_B[adapter]
    scaling = adapter_weights.mean(dim=0)[i]
    result += lora_B(lora_A(dropout(x))) * scaling
```
Conclusion: the aggregated object is the adapter output increment.

## 4. LoRA Configuration

| Item | Actual Value | Source File and Lines | Notes |
| --- | --- | --- | --- |
| Rank `r` | `4` | `src/run_continual_causal_llama2.py:52-60` | Also reconstructed as `4` in `utils/inference_utils.py:94-102`. |
| `lora_alpha` | `16` | `src/run_continual_causal_llama2.py:52-60` | Stored per adapter in `mpeft/tuners/lora/layer.py:82-83`. |
| Stored LoRA scaling formula | `alpha / r` | `mpeft/tuners/lora/layer.py:93-97` | `use_rslora=False` by default, so stored scaling is `16 / 4 = 4`. |
| Effective scaling in active global path | routing weight only | `mpeft/tuners/lora/layer.py:323-331` | The active composition path does not multiply by `self.scaling[adapter]`. |
| `lora_dropout` | `0.1` | `src/run_continual_causal_llama2.py:52-60` | Stored as per-adapter dropout module in `mpeft/tuners/lora/layer.py:84-89`. |
| Target modules | `["q_proj", "v_proj"]` | `src/run_continual_causal_llama2.py:55-58` | No `k_proj`, `o_proj`, or MLP target listed. |
| Where targeting is applied | every matching target layer | `mpeft/tuners/lora/model.py:74-128` | Standard LoRA injection path. |
| Bias | `"none"` | `src/run_continual_causal_llama2.py:52-60` | Therefore bias parameters are not intentionally made trainable. |
| `use_rslora` | `False` default | `mpeft/tuners/lora/config.py:252-260` | No override found in run script. |
| `fan_in_fan_out` | `False` default | `mpeft/tuners/lora/config.py:245-248` | No override found. |
| LoRA initialization | `True` default | `mpeft/tuners/lora/config.py:284-288`, `mpeft/tuners/lora/layer.py:98-101, 119-128` | `A` initialized, `B` zero-initialized. |
| Quantization | 4-bit NF4 base model and query encoder | `src/run_continual_causal_llama2.py:103-113, 140-146, 167-174` | QLoRA-style backbone loading via `BitsAndBytesConfig(load_in_4bit=True)`. |
| Global composition switch | `True` default | `mpeft/tuners/lora/config.py:63-68` | No override found. |
| `softmax_match_scale` | `8` | `mpeft/tuners/lora/config.py:99-104` | Used as a fixed config scalar, not a trainable parameter. |
| Task architecture consistency | Yes | final checkpoint shapes plus config | All six adapters have identical `A/B` shapes and parameter counts. |

Architecture consistency evidence:

File: final snapshot `checkpoint_info.json` and `state_dict.pt` under `checkpoints_continual_keshi_llama/order1_compose_peft/snapshots/task_5_zhongliuke_train_end_20260626_123739/`  
Observed facts:
- `lora_adapters = ['neike', 'waike', 'erke', 'fuchanke', 'nanke', 'zhongliuke']`
- Each adapter has:
  - `A` shapes only `(4, 4096)`
  - `B` shapes only `(4096, 4)`
  - `128` LoRA tensors total
  - `2,097,152` LoRA parameters total
- This implies identical structure across all tasks. The `64` target sites are inferred from `128` tensors = `64` pairs of `A/B`, consistent with `32` decoder layers times `q_proj` and `v_proj`.

## 5. Exact Adapter Composition Mechanism
- Composition category: Case C, adapter-output composition.
- Weight shape: `w` from `TaskKeyEncoder.forward` has shape `[batch_size, num_visible_tasks]`.
- Composition location: `mpeft/tuners/lora/layer.py:323-331`.
- Layer-wise behavior: the same `adapter_weights` tensor is passed to every decoder layer and to every LoRA-wrapped target projection.
- Batch behavior: weights are first sample-specific, then averaged across the batch inside each LoRA target layer via `adapter_weights.mean(dim=0)[i]`.
- Token behavior: weights are not token-specific in the active path.
- Layer-specific behavior: weights are not layer-specific in the active global path.
- Weight normalization: `softmax(cos_sim * softmax_match_scale, dim=-1)` unless `direct_compose=True`; default is `False`.
- Masking / truncation: optional `keys_mask` exists, but static analysis found only initialization to all `False` and no update site in searched project code.

### Exact mathematical formula

For one targeted linear layer during global composition, the actual code implements:

\[
y = W_0 x + \sum_{k=0}^{n} \bar{w}_k \, B_k(A_k(D_k(x)))
\]

where:
- `n` is the current task index during task-`n` training, or the final visible task index during final inference.
- `D_k` is the task-specific LoRA dropout module.
- `\bar{w}_k = mean_batch(w[:, k])`.
- `w = softmax(8 \cdot cos_sim(query, dynamic_keys), dim=-1)` under default config.

Important code-backed caveat:
- The active global-composition path does not multiply by the stored LoRA scaling factor `self.scaling[k] = alpha / r`.
- Therefore the implemented forward is not:
\[
W_0 x + \sum_k \bar{w}_k \frac{\alpha}{r} B_k A_k x
\]
- It is exactly the unscaled LoRA output mixture shown above, plus base output.

### Exact code evidence

File: `model/key_encoder.py`  
Function: `TaskKeyEncoder.forward`  
Lines: 129-149  
Snippet:
```python
K_meta = self.all_meta_keys[:task_id+1]
K_task = self.keys[:task_id+1]
E_dynamic_k = self.dynamic_attn_layer(x_query, K_meta, K_task)
cos_sim = torch.bmm(n_q, n_dynamic_k.transpose(1, 2)).squeeze(1)
w = nn.functional.softmax(cos_sim*self.config.softmax_match_scale, dim=-1)
```
Conclusion: routing weights are computed over all visible tasks up to and including the current task.

File: `mpeft/tuners/lora/layer.py`  
Function: `Linear.forward`  
Lines: 323-331  
Snippet:
```python
for i, adapter in enumerate(self.lora_A.keys()):
    if i < adapter_weights.shape[-1]:
        scaling = adapter_weights.mean(dim=0)[i]
        result += lora_B(lora_A(dropout(x))) * scaling
```
Conclusion: the routing weight is applied after `A/B` transformation, directly on adapter output increments.

File: `mpeft/tuners/lora/layer.py`  
Function: `Linear.forward`  
Lines: 333-342  
Snippet:
```python
for active_adapter in self.active_adapters:
    scaling = self.scaling[active_adapter]
    result += lora_B(lora_A(dropout(x))) * scaling
```
Conclusion: stored LoRA scaling is used only in the non-composition fallback branch, not in the active composition branch.

## 6. Training-Time Behavior

### 6.1 Modules included
- Task indexing is 0-based because the trainer loops with `enumerate(self.task_list)` in `training/trainer_continual_causal_llama_lora.py:399-400`.
- When training task `n`, the active adapter is set to the current task name by `self.model.set_adapter(adapter_name=task)` at `training/trainer_continual_causal_llama_lora.py:409-410`.
- The routing set and composition set both include tasks `0..n`, because `TaskKeyEncoder.forward` slices `[:task_id+1]`.

### 6.2 Role of old composed modules
- Old adapters remain in forward computation because `Linear.forward` iterates over all `self.lora_A.keys()` and uses the first `adapter_weights.shape[-1]` entries.
- Old adapters contribute weighted increments to every active LoRA target projection.
- Old adapters are frozen when inactive via `mpeft/tuners/tuners_utils.py:473-484`.

### 6.3 Role of the current module
- The current adapter is included in routing weight computation and in the same weighted sum.
- There is no separate branch like “composed old module + unweighted current module” in the active global path.

### 6.4 First-task special case
- For the first task (`task_id = 0`), `K_meta` and `K_task` each contain one row.
- Softmax over one visible task yields weight `1`.
- The first-task LoRA increment in the active composition path is therefore:
\[
W_0 x + B_0(A_0(D_0(x)))
\]
not the standard stored LoRA scaling form `W_0 x + \frac{\alpha}{r} B_0 A_0 x`.

### 6.5 Training formula

For task index `n`, the code-backed training forward is:

\[
w = softmax(8 \cdot cos(query, dynamic\_keys_{0:n}))
\]
\[
h' = BaseModel(h;\, W_0) + \sum_{k=0}^{n} \bar{w}_k \, \Delta h_k
\]
\[
\Delta h_k = B_k(A_k(D_k(h)))
\]
\[
loss = lm\_loss + key\_encoder\_loss + \lambda_1 \, orthogonal\_loss + \lambda_2 \, l2\_loss
\]

where:
- `key_encoder_loss` is returned as `match_loss` from `TaskKeyEncoder.forward`.
- `orthogonal_loss` is computed between the current adapter’s LoRA parameters and previous adapters’ LoRA parameters.
- `l2_loss` is applied only to the current trainable LoRA parameters in the trainer.

### 6.6 Training pseudocode

```python
query = query_encoder(query_input_ids, query_attention_mask)   # no_grad
query = avg_word_embed(query)                                  # trainer path
w, key_loss = key_encoder(query, current_task_name)

for each decoder layer:
    for each targeted q_proj / v_proj:
        out = base_linear(x)
        for k in visible_tasks_0_to_n:
            out += mean_batch(w[:, k]) * B_k(A_k(dropout_k(x)))

lm_loss = masked_token_ce(...)
base_loss = lm_loss + key_loss
orth_loss = orthogonal(current_adapter, previous_adapters)
l2_loss = sum(||param||_2 for current_trainable_lora_params)
total_loss = base_loss + lambda1 * orth_loss + lambda2 * l2_loss
```

### 6.7 Extra loss facts
- `match_loss` is added inside `model/causal_lm_llama.py:147-164`.
- `orthogonal_loss` and current-adapter `l2_loss` are added in `training/trainer_continual_causal_llama_lora.py:488-523`.
- The key encoder also contains optional key-orthogonality and key-L2 losses in `model/key_encoder.py:190-233`.
- Important inactive-flag finding:
  - The run script passes `--matching_loss_v2` (`run_continual_script/keshi_llama/scrip.sh:24-48`).
  - But `src/run_continual_causal_llama2.py:188-196` does not forward `matching_loss_v2` into `KeyEncoderConfig`.
  - Static analysis found no later assignment to `mpeft_config.matching_loss_v2`.
  - Therefore `TaskKeyEncoder.config.matching_loss_v2` remains its default `False` unless altered outside the searched code. This should be treated as a code fact, not an assumption.

## 7. Inference-Time Behavior

### 7.1 Loaded modules
- `utils/inference_utils.py:55-56` loads adapter order from `checkpoint_info.json`.
- `utils/inference_utils.py:110-112` reconstructs all adapters by repeatedly calling `get_peft_model(..., adapter_name=task_name)`.
- `utils/inference_utils.py:113-119` loads `state_dict.pt` into the rebuilt model.

### 7.2 Routing process
- Generation calls `model.generate(..., active_adapter=task_list[-1], query_embed=query_embed, train=False, final=False)` at `utils/inference_utils.py:166-175`.
- `PeftModel.generate` forwards `peft_config` when global composition is enabled at `mpeft/peft_model.py:1121-1129`.
- `LlamaContinualForCausalLM.forward` again calls `self.key_encoder(...)` to produce routing weights during generation when `query_embed` is provided.
- Because the key encoder slices `[:task_id+1]`, using `active_adapter=task_list[-1]` makes all learned tasks visible at inference.

### 7.3 Inference formula

For the standalone inference utility:

\[
query = last\_token\_hidden\_state(query\_encoder("Query: " + question))
\]
\[
w = softmax(8 \cdot cos(query, dynamic\_keys_{0:K-1}))
\]
\[
y = W_0 x + \sum_{k=0}^{K-1} \bar{w}_k \, B_k(A_k(D_k(x)))
\]

where `K` is the number of learned task adapters in the loaded checkpoint.

### 7.4 Inference pseudocode

```python
task_list = checkpoint_info["lora_adapters"]
rebuild_all_named_adapters(task_list)
load_state_dict(snapshot_state_dict)

query = query_encoder("Query: " + question)   # no_grad
query = last_hidden_state[:, -1, :]           # standalone utility path
w = key_encoder(query, adapter_name=task_list[-1], train=False, final=False)

generate with active_adapter=task_list[-1]
# each targeted q_proj / v_proj uses the same weighted mixture over all visible adapters
```

### 7.5 Differences from training
- The trainer path uses `avg_word_embed` query pooling in `training/trainer_continual_causal_llama_lora.py:557-564`.
- The standalone inference utility uses the last token hidden state in `utils/inference_utils.py:124-128`.
- This is a real training/inference implementation difference in the repository.
- Both paths keep the query encoder frozen and run it under `torch.no_grad()`.

## 8. Gradient-Flow Audit

The table below describes the code-backed state when training task `n > 0`. First-task behavior is the same except there are no old LoRA adapters yet.

| Parameter category | Participates in forward when training task `n` | `requires_grad` | Enters optimizer | Receives gradient | Updated |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base LLM parameters | Yes | No | No | No | No |
| Current task LoRA `P_n` | Yes | Yes | Yes | Yes | Yes |
| Old task LoRA `P_0..P_{n-1}` | Yes | No | No | No parameter grad | No |
| Current task vector row in `self.keys` | Yes | Yes | Yes | Yes | Yes |
| Old task vector rows in `self.keys` | Yes | Yes | Yes | Yes | Yes |
| Static metadata embeddings `all_meta_keys` | Yes | No | No | No | No |
| Metadata encoder module | No runtime module found | N/A | N/A | N/A | N/A |
| Query encoder | Yes | No | No | No | No |
| Router/matching parameters | `self.keys` only | Yes | Yes | Yes | Yes |
| Temperature/scale parameter | No trainable parameter | N/A | No | No | No |
| Dynamic fusion parameters | None found in `DynamicAttentionLayer` | N/A | No | No | No |

### Gradient evidence

File: `mpeft/tuners/lora/model.py`  
Function: `_mark_only_adapters_as_trainable`  
Lines: 157-160  
Snippet:
```python
for n, p in model.named_parameters():
    if self.prefix not in n:
        p.requires_grad = False
```
Conclusion: the base model is frozen after LoRA injection.

File: `mpeft/tuners/tuners_utils.py`  
Function: `set_adapter`  
Lines: 473-484  
Snippet:
```python
for key, layer in module_dict.items():
    if key in adapter_names:
        layer.requires_grad_(True)
    else:
        layer.requires_grad_(False)
```
Conclusion: only the active adapter’s LoRA parameters remain trainable; old adapters are frozen.

File: `mpeft/peft_model.py`  
Function: `PeftModel.__init__`  
Lines: 128-135  
Snippet:
```python
if adapter_name == first_task:
    model.key_encoder = key_encoder
else:
    for p in model.key_encoder.parameters():
        p.requires_grad_(True)
```
Conclusion: the global key encoder is present and explicitly kept trainable for later tasks; on the first task it is attached after LoRA freezing logic, so its parameters keep their default trainable state.

File: `model/key_encoder.py`  
Function: `_initialize_keys`  
Lines: 171-176  
Snippet:
```python
self.keys = nn.Parameter(self.all_meta_keys.clone().detach(), requires_grad=True)
self.keys_mask = [False for _ in range(self.num_tasks)]
```
Conclusion: task vectors are trainable; metadata embeddings are copied into a trainable parameter matrix but the original metadata tensor stays non-trainable.

File: `model/key_encoder.py`  
Function: `__init__`  
Lines: 78-84  
Snippet:
```python
loaded_meta_keys = torch.load(...)
self.register_buffer("all_meta_keys", loaded_meta_keys)
```
Conclusion: metadata embeddings are stored as a buffer, not as a parameter.

File: `training/trainer_continual_causal_llama_lora.py`  
Function: `_prepare_optimizer`  
Lines: 135-149  
Snippet:
```python
"params": [p for n, p in self.model.named_parameters() if p.requires_grad ...]
self.optimizer = AdamW(...)
```
Conclusion: only trainable model parameters enter the optimizer; the separate query encoder does not.

File: `training/trainer_continual_causal_llama_lora.py`  
Function: `_get_query_embed`  
Lines: 552-565  
Snippet:
```python
with torch.no_grad():
    hidden_states = self.query_encoder(...)
```
Conclusion: the query encoder is forward-only and frozen.

File: `model/key_encoder.py`  
Function: `TaskKeyEncoder.forward`  
Lines: 136-148  
Snippet:
```python
n_q = nn.functional.normalize(x_query.to(E_dynamic_k.dtype), dim=-1).detach().unsqueeze(1)
w = nn.functional.softmax(cos_sim*self.config.softmax_match_scale, dim=-1)
```
Conclusion: gradients from task loss do not flow back into the query embedding, but they do flow into `E_dynamic_k`, then into `self.keys`.

Additional findings:
- `detach_w_from_task_loss` is declared in `mpeft/tuners/lora/config.py:105-110` but no usage was found in searched forward/training code. No detach of `w` from task loss is implemented.
- No code path was found that freezes old rows of `self.keys`; therefore old task vectors remain jointly trainable.
- Old LoRA parameters are frozen, but their outputs remain in the forward graph as constant transforms multiplied by differentiable routing weights, so the routing weights and task vectors can still be updated based on loss contributions flowing through those fixed outputs.

## 9. Routing and Metadata Parameters
- Static or trainable:
  - `all_meta_keys`: static buffer.
  - `self.keys`: trainable.
  - `DynamicAttentionLayer`: no trainable parameters found.
  - `softmax_match_scale`: fixed config value `8`, not a parameter.
- Updated across tasks or frozen:
  - `self.keys` persists across tasks and remains trainable.
  - `process_task_count()` only increments `task_count` and optionally re-orthogonalizes keys if configured; it does not freeze old task vectors.
- Gradient sources:
  - LM loss through adapter weights into `self.keys`.
  - Optional key-orthogonality and key-L2 losses inside the key encoder.
- Relevant losses:
  - `matching_loss`, `matching_loss_v2`, `matching_loss_cls`, `matching_loss_cls_all` exist in code, but only those actually present in `mpeft_config` can activate.

## 10. Checkpoint Contents

| Artifact | Included Parameters | Key Examples | Shapes | Loading Code |
| --- | --- | --- | --- | --- |
| `state_dict.pt` | keys containing `key_encoder`, `lora`, or `lm_head` | `...q_proj.lora_A.neike.weight`, `...v_proj.lora_B.waike.weight`, `key_encoder.keys` | e.g. `(4,4096)`, `(4096,4)` | `training/trainer_continual_causal_llama_lora.py:345-390` saves; `utils/inference_utils.py:113-119` loads |
| `checkpoint_info.json` | task metadata, adapter order, key summary, saved key names | `lora_adapters`, `task_id`, `state_dict_keys` | JSON | `training/trainer_continual_causal_llama_lora.py:375-389` |

Checkpoint facts:
- Save filter is explicit in `training/trainer_continual_causal_llama_lora.py:345-349`:
```python
if 'key_encoder' in k or 'lora' in k or 'lm_head' in k:
    state_dict_to_save[k] = v.detach().cpu()
```
- Therefore the snapshots do not save the full quantized base model weights.
- `checkpoint_info.json` records `lora_adapters`, which the inference utility reuses as adapter reconstruction order.
- No optimizer state is saved in this snapshot path.
- The final snapshot `task_5_zhongliuke_train_end_20260626_123739` corresponds to the sixth task because trainer indexing is 0-based.

## 11. Architecture Consistency Across Task Modules

| Task Adapter | Rank | Target Layers | Parameter Shapes | Parameter Count | Identical |
| --- | --- | --- | --- | ---: | --- |
| `neike` | 4 | q_proj + v_proj across all injected decoder sites | `A:(4,4096)`, `B:(4096,4)` only | 2,097,152 | Yes |
| `waike` | 4 | q_proj + v_proj across all injected decoder sites | `A:(4,4096)`, `B:(4096,4)` only | 2,097,152 | Yes |
| `erke` | 4 | q_proj + v_proj across all injected decoder sites | `A:(4,4096)`, `B:(4096,4)` only | 2,097,152 | Yes |
| `fuchanke` | 4 | q_proj + v_proj across all injected decoder sites | `A:(4,4096)`, `B:(4096,4)` only | 2,097,152 | Yes |
| `nanke` | 4 | q_proj + v_proj across all injected decoder sites | `A:(4,4096)`, `B:(4096,4)` only | 2,097,152 | Yes |
| `zhongliuke` | 4 | q_proj + v_proj across all injected decoder sites | `A:(4,4096)`, `B:(4096,4)` only | 2,097,152 | Yes |

Final conclusion: All task-specific LoRA modules have identical architectures: Yes.

## 12. Comparison Between Equation (6) and Implementation
- Current manuscript meaning: `P'_n = \sum_k W_k P_k` is under-specified because it does not define whether `P_k` is a parameter object, a weight update, or an output increment.
- Actual implementation: the code computes a weighted sum of adapter output increments at each targeted projection layer.
- Identified ambiguity or mismatch:
  - `P_k` should not be described as a plain module tensor unless the paper explicitly defines it as the adapter-induced hidden-state increment.
  - The summation range must include the current task during training task `n`.
  - The active global path omits the stored LoRA `alpha/r` scaling factor.
  - The same routing weights are shared across layers and batch-averaged inside each target layer.
- Minimal correction required:
  - Replace parameter-level wording with adapter-output composition wording.
  - Make the sum range explicit.
  - State that the implemented coefficient is the routing weight alone in the active global-composition code path.

Recommended code-faithful formula:

\[
\Delta h^{(\ell,m)} = \sum_{k=0}^{n} \bar{w}_k \, B_k^{(\ell,m)} A_k^{(\ell,m)} D_k^{(\ell,m)}(h)
\]
\[
h' = W_0^{(\ell,m)} h + \Delta h^{(\ell,m)}
\]

where `(\ell,m)` denotes a targeted projection module such as `q_proj` or `v_proj` in layer `\ell`.

## 13. Manuscript-Ready Technical Facts

The reviewed code implements each task-specific module as a full LoRA adapter identified by task name, with task-specific `A` and `B` matrices injected into the `q_proj` and `v_proj` linear projections at all adapted decoder layers. During both continual training and final routing-based inference, the model does not weight or merge LoRA parameters directly; instead, it computes a routing distribution over visible task adapters and adds a weighted sum of their adapter output increments to the base projection output. All task adapters share the same architecture and configuration in the released checkpoints (`r=4`, `lora_alpha=16`, `lora_dropout=0.1`, bias disabled, 4-bit quantized backbone). When training task `n`, the routing set includes tasks `0..n`, so the current task adapter is included in the weighted mixture rather than being added through a separate unweighted branch. Old adapter parameters are frozen, but old task vectors in the routing module remain trainable. Metadata embeddings are static buffers, the query encoder is frozen and evaluated under `no_grad`, and the primary trainable routing parameters are the task-key vectors. A separate repository inference utility uses a different query-pooling rule from the trainer path and should be documented if used.

## 14. Need for Additional Experiments
- Verdict: No for clarifying the mechanism behind the reported continual-learning results.
- Reason: the main issue is that the manuscript formula and implementation description are incomplete or imprecise relative to code; the mechanism can be described accurately without retraining.
- Whether retraining is necessary: Not required to answer the reviewer’s clarification request, assuming the reported results come from the repository’s main training/evaluation path.
- Whether existing results remain valid: Yes as results from the implemented mechanism, but the manuscript should describe the mechanism as adapter-output composition and should note the exact training/inference paths used.

## 15. Unconfirmed Items
- `TBD`: whether any external, unsearched script mutates `mpeft_config.matching_loss_v2` after construction.
- `TBD`: whether the manuscript’s reported inference numbers used the trainer evaluation path or the standalone `utils/inference_utils.py` path.
- `TBD`: whether the omission of stored LoRA `alpha/r` scaling in the active global-composition branch was intentional; the audit reports only the code fact.

## 16. Evidence Appendix

### A. Adapter creation
File path: `mpeft/tuners/lora/layer.py`  
Line range: 77-97  
Minimal snippet:
```python
self.lora_A[adapter_name] = nn.Linear(self.in_features, r, bias=False)
self.lora_B[adapter_name] = nn.Linear(r, self.out_features, bias=False)
self.scaling[adapter_name] = lora_alpha / r
```
Interpretation: each adapter is a task-named LoRA `A/B` pair with stored scaling.

### B. Active composition branch
File path: `mpeft/tuners/lora/layer.py`  
Line range: 323-331  
Minimal snippet:
```python
for i, adapter in enumerate(self.lora_A.keys()):
    scaling = adapter_weights.mean(dim=0)[i]
    result += lora_B(lora_A(dropout(x))) * scaling
```
Interpretation: composition is over adapter outputs and uses batch-averaged routing weights.

### C. Non-composition fallback branch
File path: `mpeft/tuners/lora/layer.py`  
Line range: 333-342  
Minimal snippet:
```python
scaling = self.scaling[active_adapter]
result += lora_B(lora_A(dropout(x))) * scaling
```
Interpretation: stored LoRA scaling exists but is only used in the fallback path.

### D. Routing visible-task range
File path: `model/key_encoder.py`  
Line range: 129-149  
Minimal snippet:
```python
K_meta = self.all_meta_keys[:task_id+1]
K_task = self.keys[:task_id+1]
w = nn.functional.softmax(cos_sim*self.config.softmax_match_scale, dim=-1)
```
Interpretation: both old and current tasks are visible to routing.

### E. Query detach
File path: `model/key_encoder.py`  
Line range: 136-148  
Minimal snippet:
```python
n_q = nn.functional.normalize(x_query.to(E_dynamic_k.dtype), dim=-1).detach().unsqueeze(1)
```
Interpretation: routing does not backpropagate into the query embedding.

### F. Query encoder freeze
File path: `src/run_continual_causal_llama2.py`  
Line range: 167-179  
Minimal snippet:
```python
query_encoder = LlamaModel.from_pretrained(...)
for param in query_encoder.parameters():
    param.requires_grad = False
```
Interpretation: query encoder parameters are frozen from initialization.

### G. Trainer-side query embedding
File path: `training/trainer_continual_causal_llama_lora.py`  
Line range: 552-565  
Minimal snippet:
```python
with torch.no_grad():
    hidden_states = self.query_encoder(...)[0]
return masked_sum / (num_tokens + 1e-9)
```
Interpretation: trainer uses averaged word embeddings under `no_grad`.

### H. Standalone inference query embedding
File path: `utils/inference_utils.py`  
Line range: 124-128  
Minimal snippet:
```python
with torch.no_grad():
    query_outputs = query_encoder(**query_inputs)
return query_outputs.last_hidden_state[:, -1, :]
```
Interpretation: standalone inference uses last-token pooling, not trainer pooling.

### I. LoRA freezing and activation
File path: `mpeft/tuners/tuners_utils.py`  
Line range: 473-484  
Minimal snippet:
```python
if key in adapter_names:
    layer.requires_grad_(True)
else:
    layer.requires_grad_(False)
```
Interpretation: inactive adapters are frozen.

### J. Base model freezing
File path: `mpeft/tuners/lora/model.py`  
Line range: 157-160  
Minimal snippet:
```python
for n, p in model.named_parameters():
    if self.prefix not in n:
        p.requires_grad = False
```
Interpretation: non-LoRA backbone parameters are frozen.

### K. Key encoder attachment
File path: `mpeft/peft_model.py`  
Line range: 113-135  
Minimal snippet:
```python
if first_task and global_composition:
    key_encoder = TaskKeyEncoder(...)
...
if global_composition:
    if adapter_name == first_task:
        model.key_encoder = key_encoder
```
Interpretation: a single global key encoder is attached to the underlying LM for routing.

### L. Checkpoint save filter
File path: `training/trainer_continual_causal_llama_lora.py`  
Line range: 345-349  
Minimal snippet:
```python
if 'key_encoder' in k or 'lora' in k or 'lm_head' in k:
    state_dict_to_save[k] = v.detach().cpu()
```
Interpretation: snapshots save LoRA, key-encoder, and LM-head state, not the full base model.
