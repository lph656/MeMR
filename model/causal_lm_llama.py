from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import (
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)

from .modeling_llama import LlamaModel, LlamaForCausalLM, LLAMA_INPUTS_DOCSTRING, _CONFIG_FOR_DOC
from mpeft.tuners.lora.config import LoraConfig

class LlamaContinualForCausalLM(LlamaForCausalLM):
    # 指定模型中共享权重的键列表
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        # 创建LlamaModel实例并将其赋值给self.model
        self.model = LlamaModel(config)
        # 指定词汇表大小
        self.vocab_size = config.vocab_size
        # 创建语言模型头，将隐藏状态映射到词汇表大小的logits
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # 初始化query_encoder和key_encoder属性为None
        self.query_encoder = None
        self.key_encoder = None
        # 执行模型初始化的后处理步骤，例如，对未加载预训练权重的模块进行初始化
        self.post_init()
    # 装饰器添加输入文档字符串
    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    # 更新返回值的文档，指定输出类型为CausalLMOutputWithPast
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        # 输入token ID
        input_ids: torch.LongTensor = None,
        # 注意力掩码
        attention_mask: Optional[torch.Tensor] = None,
        # 位置索引
        position_ids: Optional[torch.LongTensor] = None,
        # 缓存的键值对
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        # 可选的嵌入向量
        inputs_embeds: Optional[torch.FloatTensor] = None,
        # 目标标签，用于计算损失
        labels: Optional[torch.LongTensor] = None,
        # 是否使用缓存加速生成
        use_cache: Optional[bool] = None,
        # 是否输出注意力权重
        output_attentions: Optional[bool] = None,
        # 是否输出隐藏状态
        output_hidden_states: Optional[bool] = None,
        # 是否返回字典格式的输出
        return_dict: Optional[bool] = None,
        # 损失掩码，控制哪些token参与损失计算
        loss_mask: Optional[torch.Tensor] = None,
        # LoRA配置，用于适配器微调
        peft_config: Optional[LoraConfig] = None,
        # 当前适配器名称
        active_adapter: Optional[str] = None,
        # 查询嵌入，用于生成适配器权重
        query_embed: Optional[torch.FloatTensor] = None,
        # 是否解耦模块
        disentangle_modules: Optional[bool] = False,
        # 缓存位置
        cache_position: Optional[bool] = None,
        # 是否训练模式
        train: Optional[bool] = True,
        # 是否最终阶段
        final: Optional[bool] = False,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        # 设置默认输出选项
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 初始化适配器权重和匹配损失
        adapter_weights = None
        match_loss = 0.
        # 检查PEFT配置并准备适配器权重
        if peft_config and not disentangle_modules:
            assert active_adapter
            assert query_embed is not None
            
            # 使用 key_encoder 生成适配器权重和匹配损失
            adapter_weights, match_loss = self.key_encoder(x_query=query_embed, adapter_name=active_adapter, train=train, final=final)

        # 调用LlamaModel前向传播，生成隐藏状态。
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            adapter_weights=adapter_weights,
        )

        # 提取隐藏状态
        hidden_states = outputs[0]
        # false
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        
        # true
        if labels is not None:
            # 对logits进行时间步移位，移除最后一个时间步，并确保张量内存连续
            shift_logits = logits[..., :-1, :].contiguous()
            # 对labels进行时间步移位，移除第一个时间步，并确保张量内存连续
            shift_labels = labels[..., 1:].contiguous()
            # 展平shift_logits，准备输入损失函数
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            # 展平shift_labels，与shift_logits对齐
            shift_labels = shift_labels.view(-1)
            # 确保shift_labels与shift_logits在同一设备上
            shift_labels = shift_labels.to(shift_logits.device)
            # 计算逐token的交叉熵损失
            loss = F.cross_entropy(shift_logits, shift_labels, reduction='none')
            if loss_mask != None:
                loss = loss * loss_mask[..., :-1].contiguous().view(-1)
            # 计算掩码加权后的平均损失
            loss = loss.sum()/loss_mask.sum()
            loss = loss + match_loss
        
        # false    
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        # 封装loss,logits,past_key_values,hidden_states,attentions并返回。
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )