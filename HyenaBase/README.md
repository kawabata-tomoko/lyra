---
license: bsd-3-clause
tags:
- dna
- biology
- genomics
- hyena
---

# HyenaDNA

Welcome! HyenaDNA is a long-range genomic foundation model pretrained on context lengths of up to **1 million tokens** at **single nucleotide resolution**. 

See below for an [overview](#model) of the model and training. Better yet, check out these resources.

**Resources:**  

- [arxiv](https://arxiv.org/abs/2306.15794)  
- [blog](https://hazyresearch.stanford.edu/blog/2023-06-29-hyena-dna)  
- [colab](https://colab.research.google.com/drive/1wyVEQd4R3HYLTUOXEEQmp_I8aNC_aLhL?usp=sharing)
- [github](https://github.com/HazyResearch/hyena-dna)


**Links to all HuggingFace models:**  

We've uploaded a [collection](https://huggingface.co/collections/LongSafari/hyenadna-models-654d0cbbe113b04ba5a0f638) of all the pretrained HyenaDNA checkpoints.

You'll see models of different sizes and sequence lengths. There are also original weights-only versions of each model in the [LongSafari organization](https://huggingface.co/LongSafari), which are designed to be loaded with the original [github](https://github.com/HazyResearch/hyena-dna) repo. These models have identical outputs to the models in the collection above, just different interfaces.

See [GPU requirements](#hardware) for each model.

### Using HyenaDNA


In this brief code sample we demonstrate fine-tuning HyenaDNA on a sequence classification task. This sample uses the `medium` checkpoint, with a maximum sequence length of 160k nucleotides. Note that training will fail if you use a sequence length longer than the maximum supported length for your chosen checkpoint.

In testing, we have been able to train at a sequence length up to about 250k nucleotides on a Colab T4 GPU (16GB VRAM). For longer sequence lengths, more memory will be required.


```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import TrainingArguments, Trainer, logging
import torch

# instantiate pretrained model
checkpoint = 'LongSafari/hyenadna-medium-160k-seqlen-hf'
max_length = 160_000

# bfloat16 for better speed and reduced memory usage
tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)

# Generate some random sequence and labels
# If you're copying this code, replace the sequences and labels
# here with your own data!
sequence = 'ACTG' * int(max_length/4)
sequence = [sequence] * 8  # Create 8 identical samples
tokenized = tokenizer(sequence)["input_ids"]
labels = [0, 1] * 4

# Create a dataset for training
ds = Dataset.from_dict({"input_ids": tokenized, "labels": labels})
ds.set_format("pt")

# Initialize Trainer
# Note that we're using extremely small batch sizes to maximize
# our ability to fit long sequences in memory!
args = {
    "output_dir": "tmp",
    "num_train_epochs": 1,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "gradient_checkpointing": True,
    "learning_rate": 2e-5,
}
training_args = TrainingArguments(**args)

trainer = Trainer(model=model, args=training_args, train_dataset=ds)
result = trainer.train()

print(result)

# Now we can save_pretrained() or push_to_hub() to share the trained model!
```

You may also find these [notebooks](https://huggingface.co/docs/transformers/notebooks) useful. Although they're not specific to HyenaDNA, they contain additional examples of training DNA and sequence classification models.

- [How to fine-tune a Nucleotide Transformer model](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/nucleotide_transformer_dna_sequence_modelling.ipynb)
- [How to fine-tune a model on text classification](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification.ipynb)

### GPU requirements (suggested)
<a name="hardware"></a>

Here are suggestions on the hardware (preferred minimum) we think you can use for each model.

GPU during: Pretrain, fine-tune, inference

- [tiny-1k](https://huggingface.co/LongSafari/hyenadna-tiny-1k-seqlen/tree/main): (T4, T4, T4)
- [small-32k](https://huggingface.co/LongSafari/hyenadna-small-32k-seqlen/tree/main): (A100-40GB, T4, T4)
- [medium-160k](https://huggingface.co/LongSafari/hyenadna-medium-160k-seqlen/tree/main): (A100-40GB, T4, T4)
- [medium-450k](https://huggingface.co/LongSafari/hyenadna-medium-450k-seqlen/tree/main): (A100-40GB, A100-40GB, T4)
- [large-1m](https://huggingface.co/LongSafari/hyenadna-large-1m-seqlen/tree/main): (A100-80GB, A100-80GB, A100-40GB)


## Model & Training Overview
<a name="model"></a>

HyenaDNA uses a simple stack of [Hyena](https://arxiv.org/abs/2302.10866) operators, which are a subquadratic drop-in replacement for attention in Transformers. The Hyena operator is able to match quality in language modeling by using modified input projections, implicit convolutions and gating, all subquadratic operations.

This enables HyenaDNA to reach context lengths of up to 500x longer than previous genomic Transformer models using dense attention, and train 160x faster at sequence length 1M (compared to Flash Attention).

We use a single character tokenizer with a primary vocab of 4 nucleotides (plus special tokens), enabling the single nucleotide resolution, a first in genomic foundation models. In addition, the implicit long convolution enables a **global receptive field** at each layer.

We pretrain using next token (nucleotide) prediction on the human reference genome (HG38).

HyenaDNA sets new SotA on 23 downstream tasks including predicting regulatory elements, chromatin profiles, and species classification. We also explore what new capabilities open up with long context in genomics, including the first use of in-context learning with soft prompt tuneable tokens and instruction fine-tuning.

Check out our [blog](https://hazyresearch.stanford.edu/blog/2023-06-29-hyena-dna) for more details on HyenaDNA!

### Authors

Eric Nguyen*, Michael Poli*, Marjan Faizi*, Armin Thomas, Callum Birch-Sykes, Michael Wornow, Aman Patel, Clayton Rabideau, Stefano Massaroli, Yoshua Bengio, Stefano Ermon, Stephen Baccus, Chris Re.

**Contact**

Eric Nguyen, etnguyen@stanford.edu  
Michael Poli, poli@stanford.edu  
Marjan Faizi, Marjan_Faizi@hms.harvard.edu  


## Citation


Feel free to cite us :)

```
@article{nguyen2023hyenadna,
      title={HyenaDNA: Long-Range Genomic Sequence Modeling at Single Nucleotide Resolution}, 
      author={Eric Nguyen and Michael Poli and Marjan Faizi and Armin Thomas and Callum Birch-Sykes and Michael Wornow and Aman Patel and Clayton Rabideau and Stefano Massaroli and Yoshua Bengio and Stefano Ermon and Stephen A. Baccus and Chris Ré},
      year={2023},
      eprint={2306.15794},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}

```