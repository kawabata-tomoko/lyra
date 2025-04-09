import os

from datasets import Dataset, load_dataset
from transformers import (AutoConfig, AutoModelForSequenceClassification,
                          AutoTokenizer, DataCollatorForLanguageModeling,
                          DefaultDataCollator, Trainer, TrainingArguments)

import wandb
from models import LyraDNAForCausalLM,LyraDNAForSequenceClassification

os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3,4,5,6,7'


# 初始化模型和数据集
model_path="./HyenaBase"
tokenizer = AutoTokenizer.from_pretrained(
    model_path, trust_remote_code=True
)
config = AutoConfig.from_pretrained(
    model_path,
    trust_remote_code=True,
    num_labels=12,
)

model = LyraDNAForSequenceClassification.from_pretrained(
    "/pf9550-bdp-A800/zhengyulong/lyradna/NCBIVirus0.1_Pretrain/checkpoint-6620",
    trust_remote_code=True,
    config=config,
    ignore_mismatched_sizes=True,
)

dataset_path="/pf9550-bdp-A800/zhengyulong/hyenadna/dataset/substrain_hcov_cls12_appendix"
from datasets import load_from_disk

trainset=load_from_disk(f"{dataset_path}/trainset")
evalset=load_from_disk( f"{dataset_path}/evalset")
# testset=load_from_disk( f"{dataset_path}/testset")
testset=load_from_disk( "/pf9550-bdp-A800/zhengyulong/hyenadna/dataset/refseq/test")

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from transformers import DefaultDataCollator, Trainer, TrainingArguments


def compute_metrics(p):
    logits,labels= p
    pred=np.argmax(logits, axis=-1)
    precision, recall, fscore, support = precision_recall_fscore_support(labels, pred, average="weighted")
    return {"precision":precision,"recall":recall,"fscore":fscore}
def p_count(m):
    ttp=0
    tp=0
    for p in m.parameters():
        c=p.numel()
        if p.requires_grad == True:
            ttp+=c
        tp+=c
    print(f"Total trainable parameters: {ttp}")
    print(f"Total parameters: {tp}")

p_count(model)
datacollator = DefaultDataCollator()
     
training_args=TrainingArguments(
    output_dir="/pf9550-bdp-A800/zhengyulong/hyenadna/loadingsample",
    evaluation_strategy="steps",
    gradient_checkpointing=True,
    eval_steps=10,
    save_steps=10,
    save_total_limit=10,
    learning_rate=5e-5,
    lr_scheduler_type= "cosine",
    warmup_ratio = 0.1,
    weight_decay=0.1,
    num_train_epochs=50,
    gradient_accumulation_steps=1,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    neftune_noise_alpha=10.0,
    max_grad_norm=5,
    bf16=False,
    logging_steps =1,
    report_to="wandb",
    optim="adamw_apex_fused",
    save_safetensors=False
)

trainer=Trainer(
    model=model,
    args=training_args,
    train_dataset= trainset,
    eval_dataset= evalset,
    data_collator=datacollator,
    compute_metrics=None,#compute_metrics
)
print(len(testset[0]["input_ids"]))
# trainer.train()
ans=trainer.predict(testset.remove_columns("label"))
torch.save(ans,"12lyra_extra.pth")
