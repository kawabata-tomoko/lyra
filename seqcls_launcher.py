from models import LyraDNAForSequenceClassification

from datasets import Dataset, load_dataset

# 使用DataCollatorForLanguageModeling处理因果语言建模
from transformers import (AutoConfig, AutoTokenizer,
                          DataCollatorForLanguageModeling, DefaultDataCollator,
                          Trainer, TrainingArguments)
import wandb
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
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
    ttp = 0
    tp = 0
    for p in m.parameters():
        c = p.numel()
        if p.requires_grad == True:
            ttp += c
        tp += c
    print(f"Total trainable parameters: {ttp}")
    print(f"Total parameters: {tp}")

# 初始化模型和数据集
model_path="./HyenaBase"
tokenizer = AutoTokenizer.from_pretrained(
    model_path, trust_remote_code=True
)


# def pack(
#     _tokenizer,
#     max_length,
#     padding="max_length",
#     pad_to_multiple_of=None,
#     return_tensors="pt",
# ):
#     def padseq(line):
#         inputs = tokenizer(
#             line["sequence"], max_length=max_length, truncation=True, padding=padding
#         )
#         return inputs
#     return padseq

# func = pack(tokenizer, 2048, padding="max_length")
# def initial_dataset(data_files,settype,output_path = "./data",perfix = "nvbi_virus_0.1"):
#     dataset_temp = load_dataset("csv", data_files=data_files)
#     dataset_temp = dataset_temp.map(func, batched=True, num_proc=128)["train"].remove_columns(["ID","sequence","Length","genome"])
#     dataset_temp.save_to_disk(f"{output_path}/{perfix}/{settype}", num_proc=128)
#     return dataset_temp

dataset_path="/pf9550-bdp-A800/zhengyulong/hyenadna/dataset/substrain_hcov_cls12_appendix"
from datasets import load_from_disk
trainset=load_from_disk(f"{dataset_path}/trainset")
evalset=load_from_disk( f"{dataset_path}/evalset")
testset=load_from_disk( f"{dataset_path}/testset")

datacollator = DefaultDataCollator()
     
config = AutoConfig.from_pretrained(
    # model_path,
    "/pf9550-bdp-A800/zhengyulong/lyradna/BVBRC_Pretrain_1e-3/checkpoint-4500",
    trust_remote_code=True,
    num_labels=12,
    # classfier_depth=2,
    # depths=8
)
model =  LyraDNAForSequenceClassification.from_pretrained(
    # "/pf9550-bdp-A800/zhengyulong/lyradna/NCBIVirus0.1_Pretrain/checkpoint-6620",
    "/pf9550-bdp-A800/zhengyulong/lyradna/BVBRC_Pretrain_1e-3/checkpoint-4500",
    trust_remote_code=True,
    config=config,
    ignore_mismatched_sizes=True
    )
p_count(model)


training_args = TrainingArguments(
    output_dir="/pf9550-bdp-A800/zhengyulong/lyradna/seqcls_hcov12_1e-4e25_BVBRC",
    evaluation_strategy="steps",
    gradient_checkpointing=False,
    eval_steps=50,
    save_steps=50,
    save_total_limit=10,
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    weight_decay=0.1,
    num_train_epochs=25,
    # eval_accumulation_steps=8,
    gradient_accumulation_steps=4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    neftune_noise_alpha=5.0,
    max_grad_norm=5,
    bf16=False,
    logging_steps=1,
    report_to="wandb",
    optim="adamw_apex_fused",
    save_safetensors=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=trainset,
    eval_dataset=evalset,
    data_collator=datacollator,
    compute_metrics=compute_metrics,
)
trainer.train()
