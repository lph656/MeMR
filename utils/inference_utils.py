import json
import os
from typing import List, Optional

import torch
from transformers import BitsAndBytesConfig, GenerationConfig, LlamaModel, LlamaTokenizer

from model.causal_lm_llama import LlamaContinualForCausalLM
from mpeft import KeyEncoderConfig, get_peft_model
from mpeft.tuners.lora import LoraConfig
from mpeft.utils import TaskType


DEFAULT_PROMPT_TEMPLATE = "你是一位专业的AI医生，请根据患者的提问给出建议。\n\n患者提问：{question}\n\nAI医生回答："


def load_checkpoint_task_list(checkpoint_dir: str) -> List[str]:
    info_path = os.path.join(checkpoint_dir, "checkpoint_info.json")
    if not os.path.isfile(info_path):
        raise FileNotFoundError(f"checkpoint_info.json not found in {checkpoint_dir}")

    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)

    task_list = info.get("lora_adapters")
    if not task_list:
        raise ValueError(f"'lora_adapters' is missing or empty in {info_path}")
    return task_list


def find_key_encoder(module):
    if hasattr(module, "key_encoder") and module.key_encoder is not None:
        return module.key_encoder
    if hasattr(module, "base_model"):
        found = find_key_encoder(module.base_model)
        if found is not None:
            return found
    if hasattr(module, "model"):
        return find_key_encoder(module.model)
    return None


def load_memr_model(
    base_model_path: str,
    checkpoint_dir: str,
    meta_embeddings_path: str,
    inference_log_dir: str = "/tmp/memr_inference_logs",
):
    state_dict_path = os.path.join(checkpoint_dir, "state_dict.pt")
    if not os.path.isfile(state_dict_path):
        raise FileNotFoundError(f"state_dict.pt not found in {checkpoint_dir}")
    if not os.path.isfile(meta_embeddings_path):
        raise FileNotFoundError(f"metadata embeddings not found: {meta_embeddings_path}")

    task_list = load_checkpoint_task_list(checkpoint_dir)
    print(f"Loaded task adapters from checkpoint: {task_list}")

    tokenizer = LlamaTokenizer.from_pretrained(
        base_model_path,
        add_prefix_space=True,
        padding_side="left",
        trust_remote_code=True,
    )
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2
    tokenizer.pad_token_id = 1
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )

    model = LlamaContinualForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )
    query_encoder = LlamaModel.from_pretrained(
        base_model_path,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=compute_dtype,
        trust_remote_code=True,
    )
    for param in query_encoder.parameters():
        param.requires_grad = False
    query_encoder.eval()

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=4,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        target_modules=["q_proj", "v_proj"],
        mpeft_enabled=True,
    )
    peft_config.mpeft_config = KeyEncoderConfig(
        task_list=task_list,
        meta_embeddings_path=meta_embeddings_path,
        key_dim=query_encoder.config.hidden_size,
        output_dir=inference_log_dir,
    )

    for task_name in task_list:
        model = get_peft_model(model, peft_config, adapter_name=task_name)

    custom_state_dict = torch.load(state_dict_path, map_location="cpu")
    source_state_dict = {
        k: v.float() if torch.is_tensor(v) else torch.tensor(v, dtype=torch.float32)
        for k, v in custom_state_dict.items()
    }
    model.load_state_dict(source_state_dict, strict=False)
    model.eval()

    return model, tokenizer, query_encoder, task_list


def build_query_embedding(tokenizer, query_encoder, question: str):
    query_inputs = tokenizer(f"Query: {question}", return_tensors="pt").to(query_encoder.device)
    with torch.no_grad():
        query_outputs = query_encoder(**query_inputs)
    return query_outputs.last_hidden_state[:, -1, :]


def generate_memr_response(
    model,
    tokenizer,
    query_encoder,
    task_list: List[str],
    question: str,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
    repetition_penalty: float = 1.1,
):
    query_embed = build_query_embedding(tokenizer, query_encoder, question)

    key_encoder = find_key_encoder(model)
    if key_encoder is None:
        raise RuntimeError("TaskKeyEncoder not found in model.")

    key_encoder_device = next(key_encoder.parameters()).device
    if query_embed.device != key_encoder_device:
        query_embed = query_embed.to(key_encoder_device)

    prompt = prompt_template.format(question=question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
        repetition_penalty=repetition_penalty,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            generation_config=generation_config,
            active_adapter=task_list[-1],
            query_embed=query_embed,
            train=False,
            final=False,
        )

    input_length = inputs.input_ids.shape[1]
    return tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()
