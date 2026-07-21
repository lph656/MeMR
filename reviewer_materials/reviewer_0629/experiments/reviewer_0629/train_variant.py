from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass, field

import torch
from transformers import BitsAndBytesConfig, HfArgumentParser, LlamaModel, LlamaTokenizer, TrainingArguments, set_seed

sys.path.append(".")
sys.path.append("../")
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_script_dir))
sys.path.insert(0, project_root)

from arguments import DataTrainingArguments, ModelArguments, OSLArguments
from model.causal_lm_llama import LlamaContinualForCausalLM
from mpeft import KeyEncoderConfig, LoraConfig, TaskType
from tasks.mtl5.dataloader_mtl_causal_llama import DataLoaderMTL  # noqa: F401
from training.trainer_continual_causal_llama_lora import ContinualTrainerMTL
from utils.set_logger import set_logger

from .data_loader import DataLoaderMTLRoot


@dataclass
class ReviewerDatasetArguments:
    dataset_root: str = field(
        default="datasets/medical_consult",
        metadata={"help": "Dataset root used only by reviewer-side training wrappers."},
    )


def main():
    parser = HfArgumentParser(
        (TrainingArguments, DataTrainingArguments, ModelArguments, OSLArguments, ReviewerDatasetArguments)
    )
    training_args, data_args, model_args, osl_args, reviewer_args = parser.parse_args_into_dataclasses()

    seed = training_args.seed
    set_seed(seed)
    if training_args.overwrite_output_dir and os.path.isdir(training_args.output_dir):
        shutil.rmtree(training_args.output_dir)
    os.makedirs(training_args.output_dir, exist_ok=True)
    lora_output_dir = os.path.join(training_args.output_dir, "lora")
    os.makedirs(lora_output_dir, exist_ok=True)

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=4,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        output_dir=lora_output_dir,
        target_modules=["q_proj", "v_proj"],
        mpeft_enabled=model_args.mpeft_enabled,
    )

    logfile = os.path.join(training_args.output_dir, "log.txt")
    logger = set_logger(logfile)
    config_path = os.path.join(training_args.output_dir, "configs.json")
    with open(config_path, "w", newline="\n", encoding="utf-8") as handle:
        handle.write(f"\n(m)peft_args:\n {peft_config}\n")
        handle.write(f"\ntraining_args:\n {training_args}\n")
        handle.write(f"\nosl_args:\n {osl_args}\n")
        handle.write(f"\nreviewer_dataset_args:\n {reviewer_args}\n")

    task_list = data_args.task_list.split("_")
    tokenizer = LlamaTokenizer.from_pretrained(
        model_args.model_name_or_path,
        add_prefix_space=True,
        padding_side="left",
        trust_remote_code=True,
    )
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2
    tokenizer.pad_token_id = 1
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if data_args.max_seq_length_list is not None:
        max_seq_length = [int(item) for item in data_args.max_seq_length_list.split("_")]
    else:
        max_seq_length = data_args.max_seq_length

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )

    dataloaders = DataLoaderMTLRoot(
        data_args=data_args,
        training_args=training_args,
        task_list=task_list,
        tokenizer=tokenizer,
        dataset_root=reviewer_args.dataset_root,
        max_seq_length=max_seq_length,
        overwrite_cache=data_args.overwrite_cache,
    )
    train_dataloaders = {task: dataloaders[task]["train"] for task in task_list}
    dev_dataloaders = {task: dataloaders[task]["dev"] for task in task_list}
    test_dataloaders = {task: dataloaders[task]["dev"] for task in task_list}

    model = LlamaContinualForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_args.model_name_or_path,
        quantization_config=quantization_config,
        device_map="auto",
        offload_folder="offload",
        trust_remote_code=True,
    )
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    for arg in dir(model_args):
        if not arg.startswith("__") and not callable(getattr(model_args, arg)):
            setattr(model.config, arg, getattr(model_args, arg))

    query_encoder = None
    if model_args.mpeft_enabled:
        query_encoder = LlamaModel.from_pretrained(
            model_args.model_name_or_path,
            quantization_config=quantization_config,
            device_map="auto",
            offload_folder="offload_query",
            torch_dtype=compute_dtype,
            trust_remote_code=True,
        )
        for param in query_encoder.parameters():
            param.requires_grad = False

    if model_args.mpeft_enabled:
        mpeft_config = KeyEncoderConfig(
            seed=seed,
            query_encoder_type=model_args.query_encoder_type,
            task_list=task_list,
            meta_embeddings_path=model_args.meta_embeddings_path,
            key_dim=model.config.hidden_size,
            matching_loss_v2=model_args.matching_loss_v2,
            matching_loss_coeff=model_args.matching_loss_coeff,
            ortho_loss_key=osl_args.atr_enable_key_ortho,
            ortho_loss_coeff=osl_args.atr_key_ortho_coeff,
            key_ortho_mode=osl_args.atr_key_ortho_mode,
            key_ortho_threshold=osl_args.atr_key_ortho_threshold,
            lamda_keys_L2=osl_args.atr_key_l2_lambda,
        )
        peft_config.mpeft_config = mpeft_config
        with open(config_path, "a", newline="\n", encoding="utf-8") as handle:
            handle.write(f"\n(m)peft_args_after_model_init:\n {peft_config}\n")

    trainer = ContinualTrainerMTL(
        args=training_args,
        model=model,
        query_encoder=query_encoder,
        logger=logger,
        task_list=task_list,
        label_list=None,
        peft_config=peft_config,
        lora_save_dir=os.path.join(training_args.output_dir, "checkpoint_loras"),
        early_stopping_patience=data_args.early_stopping_patience if data_args.early_stop else -1,
        tokenizer=tokenizer,
        max_target_length=data_args.max_target_length,
        learning_rate_list=data_args.learning_rate_list,
        max_train_batches_per_epoch=data_args.max_train_batches_per_epoch,
        max_eval_batches=data_args.max_eval_batches,
        max_final_test_batches=data_args.max_final_test_batches,
        lamda_1=osl_args.lamda_1,
        lamda_2=osl_args.lamda_2,
        orthogonal_threshold=osl_args.orthogonal_threshold,
        lm_loss_weight=osl_args.lm_loss_weight,
    )
    trainer.train(train_dataloaders, dev_dataloaders, test_dataloaders)


if __name__ == "__main__":
    main()

