import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from einops import rearrange, repeat
from HyenaBase.modeling_hyena import HyenaDNAPreTrainedModel, HyenaEmbeddings
from transformers.modeling_outputs import (BaseModelOutputWithNoAttention,
                                           CausalLMOutput,
                                           SequenceClassifierOutput
                                           )

# 新增RMSNorm实现
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import (BaseModelOutputWithNoAttention,
                                           CausalLMOutput,
                                           SequenceClassifierOutput)

class UNetEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
        self.pool = nn.MaxPool1d(2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        skip = x  # 保留特征用于跳跃连接
        x = self.pool(x)
        return x, skip

class UNetDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_skip = nn.Conv1d(skip_channels, out_channels, kernel_size=1)  # 调整跳跃连接的通道数
        self.conv = nn.Sequential(
            nn.Conv1d(2 * out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )

    def forward(self, x, skip):
        x = self.up(x)
        skip = self.conv_skip(skip)  # 调整通道数
        x = torch.cat([x, skip], dim=1)  # 按通道拼接
        return self.conv(x)

class UNetForSequenceClassification(nn.Module):
    supports_gradient_checkpointing = True
    def __init__(self, input_dim, num_classes, vocab_size, padding_idx):
        super().__init__()
        self.num_classes=num_classes
        self.embedding = nn.Embedding(vocab_size, input_dim, padding_idx=padding_idx)
        # 编码器
        self.enc1 = UNetEncoderBlock(input_dim, input_dim * 2)
        self.enc2 = UNetEncoderBlock(input_dim * 2, input_dim * 3)
        self.enc3 = UNetEncoderBlock(input_dim * 3, input_dim * 4)
        
        # 解码器
        self.dec3 = UNetDecoderBlock(input_dim * 4, input_dim * 3, skip_channels=input_dim * 4)
        self.dec2 = UNetDecoderBlock(input_dim * 3, input_dim * 2, skip_channels=input_dim * 3)
        self.dec1 = UNetDecoderBlock(input_dim * 2, input_dim , skip_channels=input_dim * 2)
        # 分类头
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.hidden = torch.nn.Linear(input_dim,input_dim*2)#.to(torch.bfloat16)
        self.classifier = torch.nn.Linear(input_dim*2,num_classes)#.to(torch.bfloat16)#load as bf16
        self.ln_hidden = torch.nn.LayerNorm(input_dim*2)
     
    def get_input_embeddings(self):
        return self.embedding

    def set_input_embeddings(self, value):
        self.embedding = value   
    def forward(
        self,
        input_ids = None,
        inputs_embeds = None,
        labels = None,
        output_hidden_states= None,
        return_dict = None,
    ):
       
        x = self.embedding(input_ids).transpose(-1, -2) if inputs_embeds is None else inputs_embeds # [B, C, L]
        # 编码
        x, skip1 = self.enc1(x)
        x, skip2 = self.enc2(x)
        x, skip3 = self.enc3(x)  # 保存所有跳跃连接
        # 解码
        x = self.dec3(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec1(x, skip1)
        
        # 分类
        x = self.gap(x).squeeze(-1)
        logits = self.classifier(self.ln_hidden(F.gelu(self.hidden(x))))
        
        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                logits.view(-1, self.num_classes), labels.view(-1)
            )

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=x,
        )

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        # 合并计算，减少中间变量
        norm_x = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm_x * self.weight

class PGC(nn.Module):
    def __init__(self, d_model, expansion_factor=1.0, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.expansion_factor = expansion_factor
        self.dropout = dropout
        expaned_dim=int(d_model * expansion_factor)
        self.conv = nn.Conv1d(
            expaned_dim, expaned_dim, kernel_size=3, 
            padding=1, groups=expaned_dim
        )
        self.in_proj = nn.Linear(d_model, expaned_dim*2)
        self.norm = RMSNorm(expaned_dim)
        self.in_norm = RMSNorm(d_model)
        self.out_proj = nn.Linear(expaned_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, u):
        xv = self.in_proj(self.in_norm(u))
        x, v = xv.chunk(2, dim=-1)
        x_conv = self.conv(x.transpose(-1, -2)).transpose(-1, -2)
        gate = v * x_conv
        x = self.norm(gate)
        return self.dropout(self.out_proj(x))

class DropoutNd(nn.Module):
    def __init__(self, p: float = 0.5, tie=True, transposed=True):
        super().__init__()
        self.p = p
        self.tie = tie
        self.transposed = transposed

    def forward(self, X):
        if self.training and self.p > 0:
            shape = X.shape
            if not self.transposed: 
                X = rearrange(X, 'b ... d -> b d ...')
            
            mask_shape = (shape[0], shape[1]) + (1,)*(X.ndim-2) if self.tie else X.shape
            mask = torch.rand(*mask_shape, device=X.device) < (1 - self.p)
            
            X = X * mask * (1.0 / (1 - self.p))
            
            if not self.transposed: 
                X = rearrange(X, 'b d ... -> b ... d')
        return X

class S4DKernel(nn.Module):
    def __init__(self, d_model, N=64, dt_min=0.001, dt_max=0.1, lr=None):
        super().__init__()
        H = d_model
        log_dt = torch.rand(H) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        
        C = torch.randn(H, N//2, dtype=torch.cfloat)
        self.C = nn.Parameter(torch.view_as_real(C))
        
        self.register("log_dt", log_dt, lr)
        self.register("log_A_real", torch.log(0.5 * torch.ones(H, N//2)), lr)
        self.register("A_imag", math.pi * repeat(torch.arange(N//2), 'n -> h n', h=H), lr)

    def forward(self, L):
        dt = torch.exp(self.log_dt)
        C = torch.view_as_complex(self.C)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag
        
        dtA = A * dt.unsqueeze(-1)
        K = dtA.unsqueeze(-1) * torch.arange(L, device=A.device)
        C = C * (torch.exp(dtA) - 1.) / A
        
        K = 2 * torch.einsum('hn, hnl -> hl', C, torch.exp(K)).real
        return K

    def register(self, name, tensor, lr=None):
        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))
            optim = {"weight_decay": 0.0}
            if lr is not None: optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)

class S4D(nn.Module):
    def __init__(self, d_model, d_state=64, dropout=0.0, transposed=True, **kernel_args):
        super().__init__()
        self.h = d_model
        self.n = d_state
        self.d_output = self.h
        self.transposed = transposed
        
        self.D = nn.Parameter(torch.randn(self.h))
        self.kernel = S4DKernel(self.h, N=self.n, **kernel_args)
        self.activation = nn.GELU()
        self.dropout = DropoutNd(dropout) if dropout > 0 else nn.Identity()
        
        self.output_linear = nn.Sequential(
            nn.Conv1d(self.h, 2*self.h, kernel_size=1),
            nn.GLU(dim=-2),
        )

    def forward(self, u, **kwargs):
        if not self.transposed:
            u = u.transpose(-1, -2)
        L = u.size(-1)
        
        # Compute SSM Kernel
        k = self.kernel(L=L)
        
        # FFT Convolution
        k_f = torch.fft.rfft(k, n=2*L)
        u_f = torch.fft.rfft(u, n=2*L)
        y = torch.fft.irfft(u_f * k_f, n=2*L)[..., :L]
        
        # Add skip connection
        y = y + u * self.D.unsqueeze(-1)
        
        y = self.dropout(self.activation(y))
        y = self.output_linear(y)
        
        if not self.transposed:
            y = y.transpose(-1, -2)
        return y

class Lyra(nn.Module):
    def __init__(
        self,
        model_dimension,
        pgc_configs,
        num_s4,
        d_input,
        d_output=10,
        dropout=0.2,
        prenorm=True,
        final_dropout=0.2
    ):
        super().__init__()
        self.encoder = nn.Linear(d_input, model_dimension)
        
        # 修正PGC层初始化
        self.pgc_layers = nn.ModuleList()
        for pgc_hidden, num_layers in pgc_configs:
            expansion_factor = pgc_hidden / (2 * model_dimension)
            for _ in range(num_layers):
                self.pgc_layers.append(
                    PGC(model_dimension, expansion_factor, dropout)
                )
        
        self.prenorm = prenorm
        
        # S4层堆叠
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(num_s4):
            self.s4_layers.append(
                S4D(model_dimension, dropout=dropout, transposed=True, lr=0.001)
            )
            self.norms.append(RMSNorm(model_dimension))
            self.dropouts.append(DropoutNd(dropout, transposed=True))
        
        self.decoder = nn.Linear(model_dimension, d_output)
        # self.final_dropout = nn.Dropout(final_dropout)

    def forward(self, x, return_embeddings=False):

        x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)

        for pgc_layer in self.pgc_layers:
            x = pgc_layer(x)+x
            
        x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
        
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            z = x
            if self.prenorm:
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)
            z = layer(z)
            z = dropout(z)
            x = z + x
            if not self.prenorm:
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)
        
        x = x.transpose(-1, -2)  # (B, d_model, L) -> (B, L, d_model)
        embeddings = x
        return embeddings,[]
        # x = x.mean(dim=1)  # (B, d_model)
        # x = self.final_dropout(x)
        # x = self.decoder(x)
        
        # return (x, embeddings) if return_embeddings else x

class LyraDNAModel(HyenaDNAPreTrainedModel):
    def __init__(self, config, **kwargs) -> None:
        super().__init__(config, **kwargs)
        self.embeddings = HyenaEmbeddings(config)
        self.backbone = Lyra(
                model_dimension=config.d_model,
                pgc_configs=[(256, config.n_layer//4),(128, config.n_layer//4),(64, config.n_layer//4),(32, config.n_layer//4)],  # (hidden_dim, num_layers)
                num_s4=config.depths,
                d_input=config.d_model,
                d_output=config.vocab_size,
                dropout=config.embed_dropout
            )
        self.config = config

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self, input_ids, inputs_embeds=None, output_hidden_states=None, return_dict=None
    ):
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        hidden_states = self.embeddings(input_ids) if inputs_embeds is None else inputs_embeds
        hidden_states, all_hidden_states = self.backbone(
            hidden_states,
        )
        if return_dict:
            return BaseModelOutputWithNoAttention(
                last_hidden_state=hidden_states,
                hidden_states=all_hidden_states if output_hidden_states else None,
            )
        elif output_hidden_states:
            return hidden_states, all_hidden_states
        else:
            return hidden_states

class LyraDNAForCausalLM(HyenaDNAPreTrainedModel):

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.lyra = LyraDNAModel(config)
        vocab_size = config.vocab_size
        if vocab_size % config.pad_vocab_size_multiple != 0:
            vocab_size += config.pad_vocab_size_multiple - (
                vocab_size % config.pad_vocab_size_multiple
            )
        self.vocab_size = vocab_size
        self.lm_head = nn.Linear(config.d_model, vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.lyra.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.lyra.embeddings.word_embeddings = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.lyra = decoder

    def get_decoder(self):
        return self.lyra

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutput]:

        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.lyra(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )
        


class LyraDNAForSequenceClassification(HyenaDNAPreTrainedModel):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.num_labels = kwargs.get("num_labels", config.num_labels)
        self.lyra = LyraDNAModel(config)
        self.score = nn.Linear(config.d_model, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.lyra.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.lyra.embeddings.word_embeddings = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        transformer_outputs = self.lyra(
            input_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError(
                "Cannot handle batch sizes > 1 if no padding token is defined."
            )
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (
                    torch.eq(input_ids, self.config.pad_token_id).long().argmax(-1) - 1
                ).to(logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[
            torch.arange(batch_size, device=logits.device), sequence_lengths
        ]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    pooled_logits.view(-1, self.num_labels), labels.view(-1)
                )
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=pooled_logits,
            hidden_states=None#hidden_states,
        )