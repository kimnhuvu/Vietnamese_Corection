#!/usr/bin/env python
# coding: utf-8

# In[5]:


get_ipython().system('pip install datasets evaluate accelerate')
get_ipython().system('pip install sacrebleu jiwer')
get_ipython().system('pip install sentencepiece')


# In[6]:


from huggingface_hub import interpreter_login
interpreter_login()


# In[7]:


get_ipython().system('pip install transformers')


# In[8]:


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

# In[9]:


MODEL_NAME = "vinai/bartpho-syllable"
MAX_LENGTH = 512


# # Download dataset

# In[10]:


# Tải bộ dataset
dataset = load_dataset("bmd1905/vi-error-correction-v2")


# In[11]:


dataset


# In[7]:


# reduce dataset for testing
# dataset['train'] = dataset['train'].select(range(1_000))
# dataset['test'] = dataset['test'].select(range(1_000))


# In[12]:


dataset["train"][-1]


# In[13]:


# Example
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

input = dataset["train"][-1]['input']
output = dataset["train"][-1]['output']

# Sử dụng tokenizer để mã hóa dữ liệu đầu vào
inputs = tokenizer(
    input,
    text_target=output,
    max_length=MAX_LENGTH,
    truncation=True
)

inputs, len(inputs['input_ids'])


# # Tokenize dataset

# In[14]:


def preprocess_function(examples):
    # Tokenize the text and apply truncation
    return tokenizer(examples["input"],
                     text_target=examples["output"],
                     max_length=MAX_LENGTH,
                     truncation=True,
                     )

# Apply tokenization in a batched manner for efficiency
tokenized_datasets = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["train"].column_names,
)


# # Model

# In[15]:


from transformers import AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)


# # Metric

# In[16]:


from transformers import DataCollatorForSeq2Seq

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


# In[17]:


import evaluate

metric = evaluate.load("sacrebleu")


# In[18]:


predictions = [
    "Nếu làm được như vậy thì chắc chắn sẽ không còn trường nào tùy tiện thu tiền cao, gây sự lo lắng của phụ huynh và ai không có tiền thì không cần đóng."
]
references = [
    [
        "Nếu làm được như vậy thì chắc chắn sẽ không còn trường nào tùy tiện thu tiền cao, gây sự lo lắng của phụ huynh và ai không có tiền thì không cần đóng."
    ]
]
metric.compute(predictions=predictions, references=references)


# In[19]:


predictions = [
    "Nếu làm được như vậy thì chắc chắn sẽ khôTng còn trường nà tùy tiện tu tiềncaogây sự lo hắng của phụ huynh và ai khÔng có tiền thì kông cần dong"
]
references = [
    [
        "Nếu làm được như vậy thì chắc chắn sẽ không còn trường nào tùy tiện thu tiền cao, gây sự lo lắng của phụ huynh và ai không có tiền thì không cần đóng."
    ]
]
metric.compute(predictions=predictions, references=references)


# In[20]:


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # In case the model returns more than the prediction logits
    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100s in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {"sacrebleu": result["score"]}


# # Fine-tuning the model

# In[23]:



from transformers import Seq2SeqTrainingArguments

args = Seq2SeqTrainingArguments(
    do_train=True,
    do_eval=True,
    output_dir=f"kimnhuvu276/vietnamese-correction-v2",
    num_train_epochs=50,
    learning_rate=1e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,
    evaluation_strategy="steps",
    eval_steps=20_000,
    save_strategy="steps",
    logging_steps=20_000,
    save_total_limit=5,
    predict_with_generate=True,
    fp16=False,
    push_to_hub=True,
)


# In[ ]:


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

trainer.evaluate()


# In[ ]:


trainer.train()


# In[ ]:


trainer.evaluate()


# In[ ]:


trainer.push_to_hub(tags="text2text-generation", commit_message="Training complete")


# # Inference

# In[ ]:


from transformers import pipeline

corrector = pipeline("text2text-generation", model="kimnhuvu276/vietnamese-correction-v2")


# In[ ]:


MAX_LENGTH = 512

# Define the text samples
texts = [
    "côn viec kin doanh thì rất kho khan nên toi quyết dinh chuyển sang nghề khac  ",
    "toi dang là sinh diên nam hai ở truong đạ hoc khoa jọc tự nhiên , trogn năm ke tiep toi sẽ chọn chuyen nganh về trí tue nhana tạo",
    "Tôi  đang học AI ở trun tam AI viet nam  ",
    "Nhưng sức huỷ divt của cơn bão mitch vẫn chưa thấm vào đâu lsovớithảm hoạ tại Bangladesh ăm 1970 ",
    "Lần này anh Phươngqyết xếp hàng mua bằng được 1 chiếc",
    "một số chuyen gia tài chính ngâSn hànG của Việt Nam cũng chung quan điểmnày",
    "Cac so liệu cho thay ngươi dân viet nam đang sống trong 1 cuôc sóng không duojc nhu mong đọi",
    "Nefn kinh té thé giới đang đúng trươc nguyen co của mọt cuoc suy thoai",
    "Khong phai tất ca nhưng gi chung ta thấy dideu là sụ that",
    "chinh phủ luôn cố găng het suc để naggna cao chat luong nền giáo duc =cua nuoc nhà",
    "nèn kinh te thé giới đang đứng trươc nguy co của mọt cuoc suy thoai",
    "kinh tế viet nam dang dứng truoc 1 thoi ky đổi mơi chưa tung có tienf lệ trong lịch sử"
]

# Batch prediction
predictions = corrector(texts, max_length=MAX_LENGTH)

# Print predictions
for text, pred in zip(texts, predictions):
    print("- " + pred['generated_text'])


# In[ ]:




