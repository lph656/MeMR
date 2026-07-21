from __future__ import annotations

import os

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

PROMPT_TEMPLATE = "你是一位专业的AI医生，请根据患者的提问给出建议。\n\n患者提问：{instruction}\n\nAI医生回答："


def custom_data_collator(features, tokenizer):
    targets_list = [feature.pop("targets", None) for feature in features]
    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    batch = data_collator(features)
    batch["targets"] = targets_list
    if "loss_mask" in features[0] and isinstance(features[0]["loss_mask"], list):
        batch["loss_mask"] = torch.tensor([feature["loss_mask"] for feature in features], dtype=torch.long)
    return batch


def preprocess_function(examples, tokenizer, max_seq_length, is_eval=False):
    instructions = examples["instruction"]
    outputs = examples["output"]
    sources = []
    targets = []
    for instruction, output in zip(instructions, outputs):
        sources.append(PROMPT_TEMPLATE.format(instruction=instruction))
        targets.append(f" {output}{tokenizer.eos_token}")

    tokenized_sources = tokenizer(sources, add_special_tokens=False)
    tokenized_targets = tokenizer(targets, add_special_tokens=False)

    all_input_ids = []
    all_loss_masks = []
    all_labels_for_eval = []
    all_padded_labels = []
    all_query_input_ids = []
    all_query_attention_masks = []

    for source_ids, target_ids in zip(tokenized_sources["input_ids"], tokenized_targets["input_ids"]):
        if is_eval:
            input_ids = [tokenizer.bos_token_id] + source_ids
            labels = target_ids
        else:
            input_ids = [tokenizer.bos_token_id] + source_ids + target_ids
            labels = input_ids

        source_len = len(source_ids) + 1
        loss_mask = [0] * source_len + [1] * len(target_ids) if not is_eval else [0] * len(input_ids)

        if len(input_ids) > max_seq_length:
            input_ids = input_ids[:max_seq_length]
            loss_mask = loss_mask[:max_seq_length]
        else:
            pad_length = max_seq_length - len(input_ids)
            input_ids = [tokenizer.pad_token_id] * pad_length + input_ids
            loss_mask = [0] * pad_length + loss_mask

        if len(labels) > max_seq_length:
            labels = labels[:max_seq_length]
        else:
            pad_length = max_seq_length - len(labels)
            labels = [tokenizer.pad_token_id] * pad_length + labels

        query_ids = [tokenizer.bos_token_id] + source_ids
        if len(query_ids) > max_seq_length:
            query_ids = query_ids[:max_seq_length]
        else:
            pad_length = max_seq_length - len(query_ids)
            query_ids = [tokenizer.pad_token_id] * pad_length + query_ids
        query_attention_mask = [0 if token_id == tokenizer.pad_token_id else 1 for token_id in query_ids]

        all_input_ids.append(input_ids)
        all_loss_masks.append(loss_mask)
        all_labels_for_eval.append(target_ids)
        all_padded_labels.append(labels)
        all_query_input_ids.append(query_ids)
        all_query_attention_masks.append(query_attention_mask)

    return {
        "input_ids": torch.tensor(all_input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(
            [
                [0] * sum(1 for token_id in ids if token_id == tokenizer.pad_token_id)
                + [1] * (len(ids) - sum(1 for token_id in ids if token_id == tokenizer.pad_token_id))
                for ids in all_input_ids
            ],
            dtype=torch.long,
        ),
        "labels": torch.tensor(all_padded_labels, dtype=torch.long),
        "loss_mask": torch.tensor(all_loss_masks, dtype=torch.long),
        "targets": all_labels_for_eval,
        "query_input_ids": torch.tensor(all_query_input_ids, dtype=torch.long),
        "query_attention_mask": torch.tensor(all_query_attention_masks, dtype=torch.long),
    }


def DataLoaderMTLRoot(
    data_args,
    training_args,
    task_list,
    tokenizer,
    dataset_root,
    max_seq_length=None,
    overwrite_cache=False,
):
    validation_split_percentage = (
        data_args.validation_split_percentage if hasattr(data_args, "validation_split_percentage") else 0.1
    )
    dataloaders = {}
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    custom_collator_fn = lambda features: custom_data_collator(features, tokenizer)

    for task in task_list:
        train_file_path = os.path.join(dataset_root, task, "train.json")
        if not os.path.exists(train_file_path):
            raise FileNotFoundError(f"Data file not found for task '{task}' at path: {train_file_path}")
        raw_dataset = load_dataset("json", data_files=train_file_path, split="train")
        split_dataset = raw_dataset.train_test_split(
            test_size=validation_split_percentage,
            seed=training_args.seed,
        )
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]

        train_dataset = train_dataset.map(
            lambda examples: preprocess_function(examples, tokenizer, max_seq_length, is_eval=False),
            batched=True,
            remove_columns=raw_dataset.column_names,
            load_from_cache_file=not overwrite_cache,
            desc=f"Running tokenizer on {task} train dataset",
        )
        eval_dataset = eval_dataset.map(
            lambda examples: preprocess_function(examples, tokenizer, max_seq_length, is_eval=True),
            batched=True,
            remove_columns=raw_dataset.column_names,
            load_from_cache_file=not overwrite_cache,
            desc=f"Running tokenizer on {task} eval dataset",
        )

        dataloaders[task] = {
            "train": DataLoader(
                train_dataset,
                batch_size=training_args.per_device_train_batch_size,
                sampler=torch.utils.data.RandomSampler(train_dataset),
                collate_fn=custom_collator_fn,
                drop_last=training_args.dataloader_drop_last,
                num_workers=training_args.dataloader_num_workers,
                pin_memory=training_args.dataloader_pin_memory,
            ),
            "dev": DataLoader(
                eval_dataset,
                batch_size=training_args.per_device_eval_batch_size,
                sampler=torch.utils.data.SequentialSampler(eval_dataset),
                collate_fn=custom_collator_fn,
                drop_last=False,
                num_workers=training_args.dataloader_num_workers,
                pin_memory=training_args.dataloader_pin_memory,
            ),
        }
    return dataloaders

