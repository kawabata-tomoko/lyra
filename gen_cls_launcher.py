from models import LyraDNAForCausalLM

from datasets import Dataset, load_dataset

# 使用DataCollatorForLanguageModeling处理因果语言建模
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DefaultDataCollator,
    Trainer,
    TrainingArguments,
)
import wandb
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6,7"


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
model_path = "./HyenaBase"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

from itertools import product
import numpy as np

code = []
for i in product(
    ["A", "T", "C", "G"],
    ["A", "T", "C", "G"],
    ["A", "T", "C", "G"],
    ["A", "T", "C", "G"],
    ["A", "T", "C", "G"],
    ["A", "T", "C", "G"],
):
    code.append("".join(i))
np.random.seed(42)
np.random.shuffle(code)

def translabelcode(labels,code_list,MAX_ELEMENT=5):
    result=[]
    for start in range(labels):
        # 切片操作：start::5 表示从start开始，每隔5个元素取一次
        sublist = code_list[start::labels]
        np.random.seed(42)
        np.random.shuffle(sublist)
        result.append("".join(sublist[:MAX_ELEMENT]))
    return result
label_1 = translabelcode(5,code,MAX_ELEMENT=5)
label_2 = translabelcode(12,code,MAX_ELEMENT=5)

def pack(
    _tokenizer,
    max_length,
    padding="max_length",
    pad_to_multiple_of=None,
    return_tensors="pt",
):
    def padseq(line):
        line["sequence"]+"|"+_tokenizer.cls_token+"|"+...+"|"+_tokenizer.cls_token+"|"
        inputs = _tokenizer(
            line["sequence"], max_length=max_length, truncation=True, padding=padding
        )
        return inputs

    return padseq


func = pack(tokenizer, 2048, padding="max_length")


def initial_dataset(data_files, settype, output_path="./data", perfix="nvbi_virus_0.1"):
    dataset_temp = load_dataset("csv", data_files=data_files)
    dataset_temp = dataset_temp.map(func, batched=True, num_proc=128)[
        "train"
    ].remove_columns(["ID", "sequence", "Length", "genome"])
    dataset_temp.save_to_disk(f"{output_path}/{perfix}/{settype}", num_proc=128)
    return dataset_temp


trainset = initial_dataset(
    "/home/zhengyulong/models/HyenaModel/data/trainset_0.1.csv", "trainset"
)
evalset = initial_dataset(
    "/home/zhengyulong/models/HyenaModel/data/evalset_0.1.csv", "evalset"
)

datacollator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # 使用因果语言建模
    pad_to_multiple_of=tokenizer.pad_token_id,  # 可选填充对齐
)
config = AutoConfig.from_pretrained(
    model_path, trust_remote_code=True, num_labels=12, classfier_depth=2
)
model = LyraDNAForCausalLM(config)
p_count(model)


training_args = TrainingArguments(
    output_dir="/pf9550-bdp-A800/zhengyulong/lyradna/NCBIVirus0.1_Pretrain",
    evaluation_strategy="steps",
    gradient_checkpointing=False,
    eval_steps=500,
    save_steps=500,
    save_total_limit=10,
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    weight_decay=0.1,
    num_train_epochs=10,
    gradient_accumulation_steps=1,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
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
    # compute_metrics=compute_metrics,
)
trainer.train()
