#!/usr/bin/env python
# coding: utf-8

from huggingface_hub import interpreter_login
interpreter_login()

import os
import random
import json
import torch
import torch.nn as nn
from functools import partial
from collections import namedtuple
from dataclasses import dataclass, field, asdict
from huggingface_hub import HfApi

import evaluate
import numpy as np
from datasets import load_dataset
from transformers import Trainer
from transformers import DataCollatorWithPadding
from transformers import AutoTokenizer, TrainingArguments

import warnings
warnings.filterwarnings("ignore")


# # Setup config

MODEL_NAME = "VietAI/vit5-base-vietnamese"  # Sửa để sử dụng ViT5
MAX_LENGTH = 512


# # Download dataset

dataset = load_dataset("bmd1905/vi-error-correction-v2")

# Giảm dữ liệu xuống 30% của tổng số mẫu
dataset['train'] = dataset['train'].select(range(int(len(dataset['train']) * 0.4)))
dataset['test'] = dataset['test'].select(range(int(len(dataset['test']) * 0.05)))


# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Example dữ liệu
input = dataset["train"][-1]['input']
output = dataset["train"][-1]['output']

inputs = tokenizer(
    input,
    text_target=output,
    max_length=MAX_LENGTH,
    truncation=True
)

# Tokenize dataset
def preprocess_function(examples):
    return tokenizer(
        examples["input"],
        text_target=examples["output"],
        max_length=MAX_LENGTH,
        truncation=True,
    )

tokenized_datasets = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["train"].column_names,
)

# Model
from transformers import AutoModelForSeq2SeqLM
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# Data collator
from transformers import DataCollatorForSeq2Seq
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Metric
metric = evaluate.load("sacrebleu")

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {"sacrebleu": result["score"]}


# Training Arguments
from transformers import Seq2SeqTrainingArguments
args = Seq2SeqTrainingArguments(
    do_train=True,
    do_eval=True,
    output_dir=f"kimnhuvu276/vit5-vietnamese-correction",
    num_train_epochs=50,
    learning_rate=1e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,
    evaluation_strategy="steps",
    eval_steps=2000,
    save_strategy="steps",
    save_steps=2000,
    logging_steps=2000,
    save_total_limit=5,
    predict_with_generate=True,
    fp16=True,
    push_to_hub=False,
)

# Trainer
from transformers import Seq2SeqTrainer
trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Training
trainer.train()

# Evaluation
trainer.evaluate()

# Push model to Hugging Face Hub nếu cần
# trainer.push_to_hub("Training complete")

# Inference
from transformers import pipeline

corrector = pipeline("text2text-generation", model="kimnhuvu276/vit5-vietnamese-correction")

texts = [
    "côn viec kin doanh thì rất kho khan nên toi quyết dinh chuyển sang nghề khac",
    "toi dang là sinh diên nam hai ở truong đạ hoc khoa jọc tự nhiên",
]

predictions = corrector(texts, max_length=MAX_LENGTH)

for text, pred in zip(texts, predictions):
    print("- " + pred['generated_text'])
