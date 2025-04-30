import math
import os
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_from_disk
from einops import rearrange, repeat
from HyenaBase.modeling_hyena import HyenaDNAPreTrainedModel, HyenaEmbeddings
from sklearn.metrics import precision_recall_fscore_support
from transformers import (AutoConfig, AutoTokenizer,
                          DataCollatorForLanguageModeling, DefaultDataCollator,
                          EvalPrediction, Trainer, TrainingArguments)
from transformers.modeling_outputs import (BaseModelOutputWithNoAttention,
                                           CausalLMOutput,
                                           SequenceClassifierOutput)
from unet_lyra import UNetForSequenceClassification

# # RMSNorm implementation
# class RMSNorm(nn.Module):
#     def __init__(self, dim: int, eps: float = 1e-6):
#         super().__init__()
#         self.eps = eps
#         self.weight = nn.Parameter(torch.ones(dim))

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # x shape [..., dim]
#         if x.size(-1) != self.weight.numel():
#             raise ValueError(f"RMSNorm expected last dimension {self.weight.numel()}, got {x.size(-1)}")
#         norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
#         return x * norm * self.weight

# # PGC block
# class PGC_Optimized(nn.Module):
#     def __init__(self, d_model: int, expansion_factor: float = 0.5, dropout: float = 0.0):
#         super().__init__()
#         self.d_model = d_model
#         expanded_dim = int(d_model * expansion_factor)

#         # fused projection
#         self.proj = nn.Linear(d_model, 2 * expanded_dim)
#         # convs
#         self.depthwise = nn.Conv1d(expanded_dim, expanded_dim, kernel_size=3, padding=1, groups=expanded_dim)
#         self.pointwise = nn.Conv1d(expanded_dim, expanded_dim, kernel_size=1)

#         self.in_norm = RMSNorm(d_model)
#         self.norm = RMSNorm(expanded_dim)
#         self.out_proj = nn.Linear(expanded_dim, d_model)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, u: torch.Tensor) -> torch.Tensor:
#         # u: [B, C, L]
#         u = u.permute(0, 2, 1)               # -> [B, L, C]
#         u_norm = self.in_norm(u)
#         uv = self.proj(u_norm)              # [B, L, 2*E]
#         x, v = uv.chunk(2, dim=-1)          # each [B, L, E]

#         # conv path
#         x = x.permute(0, 2, 1)              # [B, E, L]
#         x = self.depthwise(x)
#         x = self.pointwise(x)
#         x = x.permute(0, 2, 1)              # [B, L, E]

#         gate = self.norm(x * v)
#         out = self.out_proj(gate)
#         out = self.dropout(out)
#         return out.permute(0, 2, 1)         # [B, C, L]

# # Feature/dropout across dimensions
# class DropoutNd(nn.Module):
#     def __init__(self, p: float = 0.5, tie: bool = True, transposed: bool = True):
#         super().__init__()
#         self.p = p
#         self.tie = tie
#         self.transposed = transposed

#     def forward(self, X: torch.Tensor) -> torch.Tensor:
#         if self.training and self.p > 0:
#             if not self.transposed:
#                 X = rearrange(X, 'b ... d -> b d ...')
#             shape = X.shape
#             if self.tie:
#                 mask_shape = (shape[0], shape[1]) + (1,) * (X.ndim - 2)
#             else:
#                 mask_shape = shape
#             mask = (torch.rand(mask_shape, device=X.device) < (1 - self.p)).to(X.dtype)
#             X = X * mask * (1.0 / (1 - self.p))
#             if not self.transposed:
#                 X = rearrange(X, 'b d ... -> b ... d')
#         return X

# # S4D core kernel
# class S4DKernel(nn.Module):
#     def __init__(self, d_model: int, N: int = 64, dt_min: float = 0.001, dt_max: float = 0.1):
#         super().__init__()
#         self.log_dt = nn.Parameter(torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        
#         # self.C = nn.Parameter(torch.randn(d_model, N//2, dtype=torch.cfloat))
#         self.C_real = nn.Parameter(torch.randn(d_model, N//2))
#         self.C_imag = nn.Parameter(torch.randn(d_model, N//2))
#         A_imag = math.pi * torch.arange(N//2).unsqueeze(0).repeat(d_model, 1)
#         self.register_buffer("A_imag", A_imag)
#         self.log_A_real = nn.Parameter(torch.log(0.5 * torch.ones(d_model, N//2)))

#     def forward(self, L: int) -> torch.Tensor:
#         dt = torch.exp(self.log_dt)                    # [C]
#         A = -torch.exp(self.log_A_real) + 1j * self.A_imag  # [C, N/2]

#         # avoid near-zero
#         threshold = 1e-6
#         safe_real = torch.where(A.abs() < threshold,
#                                 threshold * torch.ones_like(A.real),
#                                 A.real)
#         A_safe = safe_real + 1j * self.A_imag

#         dtA = A * dt.unsqueeze(-1)
#         C = torch.complex(self.C_real, self.C_imag)
#         C = C * (torch.exp(dtA) - 1.0) / A_safe

#         l = torch.arange(L, device=A.device)
#         exp_term = torch.exp(dtA.unsqueeze(-1) * l)
#         K = 2 * torch.einsum('cn, cnl -> cl', C, exp_term).real
#         return K

# # S4D block
# class S4D(nn.Module):
#     def __init__(self, d_model: int, d_state: int = 64, dropout: float = 0.0):
#         super().__init__()
#         self.kernel = S4DKernel(d_model, N=d_state)
#         self.D = nn.Parameter(torch.randn(d_model))
#         self.output_linear = nn.Linear(d_model, d_model)
#         self.dropout = nn.Dropout(dropout)
#         # self.register_buffer("cached_k_f", None)
#         self.cached_L = None

#     def forward(self, u: torch.Tensor) -> torch.Tensor:
#         # u: [B, C, L]
#         B, C, L = u.size()
#         # if self.cached_k_f is None or self.cached_L != L:
#         #     k = self.kernel(L)
#         #     k[:, L//2+1:] = 0
#         #     self.cached_k_f = torch.fft.rfft(k, n=2*L)
#         #     self.cached_L = L
#         k = self.kernel(L)

#         u_f = torch.fft.rfft(u.float(), n=2*L)
#         k_f = torch.fft.rfft(k.float(), n=2*L)
#         # y = torch.fft.irfft(u_f * self.cached_k_f, n=2*L)[..., :L]
#         y = torch.fft.irfft(u_f * k_f, n=2*L)[..., :L]
#         y = y.to(u.dtype)

#         y = y + u * self.D.unsqueeze(-1)
#         out = self.output_linear(y.permute(0, 2, 1))
#         out = self.dropout(out)
#         return out.permute(0, 2, 1)

# # UNet building blocks
# class UNetEncoderBlock(nn.Module):
#     def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
#                  padding: int = 1, expansion_factor: float = 0.5, dropout: float = 0.0):
#         super().__init__()
#         self.pgc = PGC_Optimized(in_channels, expansion_factor, dropout)
#         self.conv = nn.Sequential(
#             nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
#             nn.BatchNorm1d(out_channels),
#             nn.ReLU()
#         )
#         self.pool = nn.MaxPool1d(2)
#         self.s4d = nn.Sequential(
#             S4D(out_channels, d_state=out_channels, dropout=dropout),
#             S4D(out_channels, d_state=out_channels, dropout=dropout)
#         )

#     def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         x = self.conv(self.pgc(x))
#         skip = x
#         x = self.pool(x)
#         x = self.s4d(x)
#         return x, skip

# class UNetDecoderBlock(nn.Module):
#     def __init__(self, in_channels: int, out_channels: int, skip_channels: int):
#         super().__init__()
#         self.up = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2)
#         self.skip = nn.Linear(skip_channels, out_channels)
#         self.conv = nn.Sequential(
#             nn.Conv1d(2*out_channels, out_channels, kernel_size=3, padding=1),
#             nn.GroupNorm(8, out_channels),
#             nn.GELU()
#         )
#         self.s4d = S4D(out_channels)

#     def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
#         x = self.up(x)
#         if x.size(-1) != skip.size(-1):
#             diff = skip.size(-1) - x.size(-1)
#             x = F.pad(x, (diff // 2, diff - diff // 2))
#         skip = self.skip(skip.transpose(-1,-2)).transpose(-1,-2)
#         x = torch.cat([x, skip], dim=1)
#         x = self.conv(x)
#         return self.s4d(x)
# class UnetBackbone(nn.Module):
#     def __init__(self,config):
#         self.embeddings = HyenaEmbeddings(config)
#         super().__init__()
#         self.embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.padding_idx)
#         self.enc1 = UNetEncoderBlock(config.d_model, config.d_model)
#         self.enc2 = UNetEncoderBlock(config.d_model, config.d_model)
#         self.dec2 = UNetDecoderBlock(config.d_model, config.d_model, skip_channels=config.d_model)
#         self.dec1 = UNetDecoderBlock(config.d_model, config.d_model, skip_channels=config.d_model)
#         self.hidden = nn.Linear(config.d_model, config.d_model)
#         self.ln_hidden = nn.LayerNorm(config.d_model)
#     def forward(self,input_ids, inputs_embeds=None, output_hidden_states=False):
#         all_hidden_states = []
        
#         if inputs_embeds is None:
#             x = self.embedding(input_ids).transpose(1, 2)
#         else:
#             x = inputs_embeds.transpose(1, 2)
#         x, skip1 = self.enc1(x)
#         x, skip2 = self.enc2(x)
#         x = self.dec2(x, skip2)
#         x = self.dec1(x, skip1)
#         x = self.gap(x).squeeze(-1)
#         x = F.gelu(self.hidden(x))
#         x = self.ln_hidden(x)
#         if output_hidden_states:
#             all_hidden_states.append(x)
#         return x,all_hidden_states


# class UnetLyraDNAModel(HyenaDNAPreTrainedModel):
#     def __init__(self, config, **kwargs) -> None:
#         super().__init__(config, **kwargs)

#         self.backbone = UnetBackbone(config)
#         self.config = config

#         # Initialize weights and apply final processing
#         self.post_init()

#     def forward(
#         self, input_ids, inputs_embeds=None, output_hidden_states=None, return_dict=None
#     ):
#         output_hidden_states = (
#             output_hidden_states
#             if output_hidden_states is not None
#             else self.config.output_hidden_states
#         )
#         return_dict = (
#             return_dict if return_dict is not None else self.config.use_return_dict
#         )

#         hidden_states, all_hidden_states = self.backbone(
#             input_ids,
#             inputs_embeds=inputs_embeds,
#             output_hidden_states=output_hidden_states,
#         )
#         if return_dict:
#             return BaseModelOutputWithNoAttention(
#                 last_hidden_state=hidden_states,
#                 hidden_states=all_hidden_states if output_hidden_states else None,
#             )
#         elif output_hidden_states:
#             return hidden_states, all_hidden_states
#         else:
#             return hidden_states


# class HyenaDNAForCausalLM(HyenaDNAPreTrainedModel):

#     def __init__(self, config, **kwargs):
#         super().__init__(config, **kwargs)
#         self.lyra = UnetLyraDNAModel(config)
#         vocab_size = config.vocab_size
#         if vocab_size % config.pad_vocab_size_multiple != 0:
#             vocab_size += config.pad_vocab_size_multiple - (
#                 vocab_size % config.pad_vocab_size_multiple
#             )
#         self.vocab_size = vocab_size
#         self.lm_head = nn.Linear(config.d_model, vocab_size, bias=False)

#         # Initialize weights and apply final processing
#         self.post_init()

#     def get_input_embeddings(self):
#         return self.lyra.backbone.embeddings.word_embeddings

#     def set_input_embeddings(self, value):
#         self.lyra.backbone.embeddings.word_embeddings = value

#     def get_output_embeddings(self):
#         return self.lm_head

#     def set_output_embeddings(self, new_embeddings):
#         self.lm_head = new_embeddings

#     def set_decoder(self, decoder):
#         self.lyra = decoder

#     def get_decoder(self):
#         return self.lyra

#     def forward(
#         self,
#         input_ids: torch.LongTensor = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         labels: Optional[torch.LongTensor] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#     ) -> Union[Tuple, CausalLMOutput]:

#         output_hidden_states = (
#             output_hidden_states
#             if output_hidden_states is not None
#             else self.config.output_hidden_states
#         )
#         return_dict = (
#             return_dict if return_dict is not None else self.config.use_return_dict
#         )

#         # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
#         outputs = self.lyra(
#             input_ids=input_ids,
#             inputs_embeds=inputs_embeds,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )

#         hidden_states = outputs[0]
#         logits = self.lm_head(hidden_states)
#         logits = logits.float()

#         loss = None
#         if labels is not None:
#             # Shift so that tokens < n predict n
#             shift_logits = logits[..., :-1, :].contiguous()
#             shift_labels = labels[..., 1:].contiguous()
#             # Flatten the tokens
#             loss_fct = nn.CrossEntropyLoss()
#             shift_logits = shift_logits.view(-1, self.vocab_size)
#             shift_labels = shift_labels.view(-1)
#             # Enable model parallelism
#             shift_labels = shift_labels.to(shift_logits.device)
#             loss = loss_fct(shift_logits, shift_labels)

#         if not return_dict:
#             output = (logits,) + outputs[1:]
#             return (loss,) + output if loss is not None else output

#         return CausalLMOutput(
#             loss=loss,
#             logits=logits,
#             hidden_states=outputs.hidden_states,
#         )
# # UNet model
# class UNetForSequenceClassification(HyenaDNAPreTrainedModel):
#     supports_gradient_checkpointing = True

#     def __init__(self, config, **kwargs):
#         super().__init__(config, **kwargs)
#         self.config=config
#         self.lyra=UnetLyraDNAModel(config)
#         self.gap = nn.AdaptiveAvgPool1d(1)
#         self.hidden = nn.Linear(config.d_model, config.d_model*2)
#         self.ln_hidden = nn.LayerNorm(config.d_model*2)
#         self.classifier = nn.Linear(config.d_model*2, config.num_classes)
#         self.post_init()

#     def get_input_embeddings(self) -> nn.Embedding:
#         return self.backbone.embedding

#     def set_input_embeddings(self, value: nn.Embedding) -> None:
#         self.backbone.embedding = value

#     def forward(
#         self,
#         input_ids: Optional[torch.LongTensor] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         labels: Optional[torch.LongTensor] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: bool = True
#     ) -> Union[SequenceClassifierOutput, Tuple[torch.Tensor, torch.Tensor, None]]:
#         if inputs_embeds is None:
#             x = self.embedding(input_ids).transpose(1, 2)
#         else:
#             x = inputs_embeds.transpose(1, 2)
#         outputs=self.lyra(
#             input_ids=x,
#             inputs_embeds=inputs_embeds,
#             output_hidden_states=output_hidden_states
#         )
#         x = self.gap(outputs[0]).squeeze(-1)
#         x = F.gelu(self.hidden(x))
#         x = self.ln_hidden(x)
#         logits = self.classifier(x)
#         loss = F.cross_entropy(logits, labels) if labels is not None else None
#         if return_dict:
#             return SequenceClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states,)
#         return (loss, logits, None)

# Set GPU
# os.environ['CUDA_VISIBLE_DEVICES'] = '7'

# Load tokenizer and model
model_path="./HyenaBase"
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True
)
config = AutoConfig.from_pretrained(
    model_path,
    trust_remote_code=True,
    num_labels=12,
    classfier_depth=2,
)
model = UNetForSequenceClassification.from_pretrained(
    "/pf9550-bdp-A800/zhengyulong/unetlyradna/BVBRC_Pretrain_1e-3/checkpoint-6000",
    config=config
)

# Load datasets
dataset_path = "/pf9550-bdp-A800/zhengyulong/hyenadna/dataset/substrain_hcov_cls12_appendix"
trainset = load_from_disk(f"{dataset_path}/trainset")
evalset  = load_from_disk(f"{dataset_path}/evalset")
testset  = load_from_disk(f"{dataset_path}/testset")

def compute_metrics(p: EvalPrediction) -> dict:
    logits = p.predictions
    labels = p.label_ids
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    return {"precision": precision, "recall": recall, "fscore": f1}

def p_count(m: nn.Module) -> None:
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in m.parameters())
    print(f"Total trainable parameters: {trainable}")
    print(f"Total parameters: {total}")

p_count(model)

datacollator = DefaultDataCollator()

training_args = TrainingArguments(
    output_dir="/pf9550-bdp-A800/zhengyulong/unetlyradna/unetsample",
    evaluation_strategy="steps",
    eval_steps=50,
    save_steps=10,
    save_total_limit=10,
    learning_rate=5e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    weight_decay=0.1,
    num_train_epochs=20,
    gradient_accumulation_steps=4,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    bf16=False,
    logging_steps=1,
    report_to="wandb",
    optim="adamw_torch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=trainset,
    eval_dataset=evalset,
    data_collator=datacollator,
    compute_metrics=compute_metrics
)

trainer.train()