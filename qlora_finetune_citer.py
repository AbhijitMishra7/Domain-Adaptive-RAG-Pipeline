#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 31 14:35:00 2025

@author: Abhijit Mishra
"""

import os
import json
import argparse
from typing import Dict

import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

PROMPT_TEMPLATE = (
    """### Instruction\nYou are a research assistant. Answer the question using ONLY the given context.\n"
    "Cite any facts using the format [Paper {paper_id}]. If the context is insufficient, say so.\n"
    "\n### Context\n{context}\n\n### Question\n{question}\n\n### Answer\n"""
)

def format_example(example: Dict[str, str]) -> Dict[str, str]:
    prompt = PROMPT_TEMPLATE.format(
        context=example["context"], question=example["question"], paper_id=example["paper_id"]
    )
    return {"prompt": prompt, "completion": example["answer"]}


def prepare_dataset(jsonl_path: str) -> Dataset:
    raw = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            raw.append(format_example(obj))
    return Dataset.from_list(raw)


def main():
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for faithful citer LLM.")
    parser.add_argument("--dataset_jsonl", required=True, help="JSONL dataset from citation_dataset_generator.py")
    parser.add_argument("--model_name", default="microsoft/Phi-3-mini-4k-instruct", help="Base model name or path")
    parser.add_argument("--output_dir", required=True, help="Directory to save fine-tuned adapter")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    args = parser.parse_args()

    ds = prepare_dataset(args.dataset_jsonl)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(ex):
        full = ex["prompt"] + ex["completion"] + tokenizer.eos_token
        tokenized = tokenizer(full, truncation=True, padding="max_length", max_length=4096)
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized_ds = ds.map(tokenize_function, batched=False, remove_columns=list(ds.column_names))

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        learning_rate=args.lr,
        bf16=torch.cuda.is_available(),
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=tokenized_ds,
        tokenizer=tokenizer,
        args=training_args,
        packing=False,
    )

    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()

