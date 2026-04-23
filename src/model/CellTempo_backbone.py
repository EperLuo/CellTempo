import math
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple
from transformers import PreTrainedModel, PretrainedConfig
from transformers.utils import logging
from transformers.modeling_outputs import CausalLMOutputWithPast
from dataclasses import dataclass, field
from typing import Optional, Tuple, Union
import json

from utils.tokenizer import mixMulanTokenizer
from .CellTempo_VQVAE.model import VQModel

logger = logging.get_logger(__name__)

def new_gelu(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))

class RMSNorm(nn.Module):
    def __init__(self, ndim, bias, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        # rms(x) = sqrt(mean(x^2)) = ||x||_2 / sqrt(d)
        norm = x.norm(2, dim=-1, keepdim=True)            # L2 norm along last dim: shape(B, T, 1)
        rms = norm * (1.0 / math.sqrt(x.size(-1)))        # convert L2 norm to RMS
        x = x / (rms + self.eps)                          # RMS normalize
        x = x * self.weight                               # scale
        if self.bias is not None:
            x = x + self.bias
        return x

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and config.use_flash

        # self.flash = False
        if not self.flash:
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, C = x.size()

        # Compute Q, K, V
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Concatenate past key-value pairs if provided
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)

        # Save key-value pairs if caching is enabled
        if use_cache:
            next_past = (k, v)
        else:
            next_past = None

        # Attention computation
        if self.flash:
            if attention_mask is None:
                # 没有attention_mask，直接依赖is_causal=True实现因果掩码
                y = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=None,
                    dropout_p=self.dropout if self.training else 0,
                    is_causal=True
                )
                
            else:
                causal_mask = torch.tril(torch.ones((T, T), dtype=torch.bool, device=x.device))
                
                # 将attention_mask扩展到(B,T,T)，使其既包含padding屏蔽，又能与因果mask相"与"
                extended_attn_mask = attention_mask[:, None, :].expand(B, T, T)  # (B,T,T)
                final_mask = extended_attn_mask & causal_mask.unsqueeze(0)  # (B,T,T)

                # 再扩展到(B, h, T, T), 这里h=1因为mask对每个头相同
                final_mask = final_mask.unsqueeze(1)  # (B,1,T,T)

                # 使用combined mask，并且is_causal=False，因为我们手动实现了因果性
                y = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=final_mask,
                    dropout_p=self.dropout if self.training else 0,
                    is_causal=False
                )

        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            if attention_mask is not None:
                causal_mask = (self.bias[:,:,:T,:T] == 1)
                combined_mask = attention_mask[:, None, None, :] & causal_mask
                att = att.masked_fill(~combined_mask, float('-inf'))
            else:
                att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y, next_past


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = new_gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Attention with KV cache
        attn_output, new_past = self.attn(
            self.ln_1(x),
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            use_cache=use_cache
        )
        x = x + attn_output

        # MLP
        x = x + self.mlp(self.ln_2(x))

        return x, new_past



@dataclass
class CellTempoConfig(PretrainedConfig):
    model_type: str = "mulan-decoder"
    block_size: int = 4000
    vocab_size: int = 36604
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 1024
    dropout: float = 0.0
    bias: bool = False
    train_mode: str = 'pretrain'
    cell_pos_num: int = 256

    pruned_heads: dict = field(default_factory=dict)

    use_flash: bool = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__dict__.update(kwargs)


class CE_logits_loss(nn.Module):
    def __init__(self, vocab_size, block_size=1000, train_mode='pretrain'):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.ce_loss = nn.CrossEntropyLoss()
        self.train_mode = train_mode

    def forward(self, logits_cls, targets, xlen, c1_len, c2_start):
        B, T, C = logits_cls.shape
        seq_lens = xlen.unsqueeze(-1)
        c2_starts = (c2_start - 1).unsqueeze(-1)
        prefix_lens = c1_len.unsqueeze(-1)
        range_tensor = torch.arange(T, device=targets.device).unsqueeze(0).expand(B, T)

        batchLogi = logits_cls.view(-1, logits_cls.shape[-1])
        batchTar = targets.view(-1).long()
        loss_cls = self.ce_loss(batchLogi, batchTar) if batchLogi.size(0) > 0 else torch.tensor(0.0, device=batchLogi.device)

        return loss_cls


@dataclass
class CausalLMOutputWithValues(CausalLMOutputWithPast):
    logits: Optional[torch.FloatTensor] = None

class CellTempo_backbone(PreTrainedModel):
    config_class = CellTempoConfig
    base_model_prefix = "transformer"

    def __init__(self, config: CellTempoConfig):
        super().__init__(config)
        self.config = config

        self.vq_model = VQModel.from_pretrained(config.vq_vae_path,cvq_distance = 'cos',cvq_anchor='probrandom')
        # freeze vq model
        for param in self.vq_model.parameters():
            param.requires_grad = False

        with open(os.path.join(config.data_folders[0], config.meta_info_name), 'r') as f:
            self.meta_info = json.load(f)
        self.__chars = self.meta_info['token_set']
        self.vocab_size = len(self.__chars)
        self.tokenizer = mixMulanTokenizer(self.__chars)
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.cell_pos_num, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = RMSNorm(config.n_embd, bias=config.bias),
        ))

        self.criterion = CE_logits_loss(
            vocab_size=config.vocab_size,
            block_size=config.block_size,
            train_mode=config.train_mode,
        )

        self.post_init()

         # report number of parameters
        if 'LOCAL_RANK' not in os.environ or os.environ['LOCAL_RANK'] == '0':
            logger.info("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.zeros_(module.bias)

    def get_num_params(self, non_embedding=True): ## we don't have wpe
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        # if non_embedding:
        #     n_params -= self.transformer.wpe.weight.numel()
        return n_params
    

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        xlen=None,
        c1_len=None,
        c2_start=None,
        cell_pos=None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        use_cache: bool = False,
        return_dict=True
    ):
        # input_ids: (B,T)
        b, t = input_ids.size()
        
        tok_emb = self.transformer.wte(input_ids)
        pos_emb = self.transformer.wpe(cell_pos) if cell_pos is not None else 0
        x = self.transformer.drop(tok_emb + pos_emb)


        new_past_key_values = [] if use_cache else None
        for i, block in enumerate(self.transformer.h):
            past = past_key_values[i] if (past_key_values is not None) else None
            x, present = block(
                x,
                past_key_value=past,
                attention_mask=attention_mask,
                use_cache=use_cache
            )
            if use_cache:
                new_past_key_values.append(present)

        x = self.transformer.ln_f(x)
        logits_cls = torch.matmul(x, self.transformer.wte.weight.transpose(0,1))

        loss = None
        if labels is not None:
            loss = self.criterion(
                logits_cls=logits_cls,
                targets=labels,
                xlen=xlen,
                c1_len=c1_len,
                c2_start=c2_start,
            )

        if not return_dict:
            return (logits_cls, loss, new_past_key_values)

        return CausalLMOutputWithValues(
            loss=loss,
            logits=logits_cls,      
            past_key_values=tuple(new_past_key_values) if use_cache else None,
            hidden_states=None,
            attentions=None,
        )


    @torch.no_grad()
    def generate_debug(
        self,
        input_ids,
        cell_pos,
        max_new_tokens=20,
        ignore_Idx=None,
        top_k=None,
        use_cache=True,
        debug=False,
        **generate_kwargs
    ):
        batch_size = input_ids.size(0)
        past_key_values = None

        if debug:
            print("=== Generation Debug Mode ON ===")
            print(f"Initial input_ids shape: {input_ids.shape}, values: {input_ids}")
            print(f"Initial cell_pos shape: {cell_pos.shape}, values: {cell_pos}")

        sentence_cls = []
        sentence_entropy = []
        for step in range(max_new_tokens):

            # if already have past_key_values, take the last token, otherwise take all input
            if past_key_values is not None:
                cur_input_ids = input_ids[:, -1:].contiguous()
                cur_cell_pos = cell_pos[:, -1:].contiguous()
            else:
                cur_input_ids = input_ids
                cur_cell_pos = cell_pos

            if debug:
                print(f"\n[Step {step}] use_cache={use_cache}")
                print(f"cur_input_ids shape: {cur_input_ids.shape}, values: {cur_input_ids}")
                print(f"cur_cell_pos shape: {cur_cell_pos.shape}, values: {cur_cell_pos}")

            # one step forward
            outputs = self(
                input_ids=cur_input_ids,
                cell_pos=cur_cell_pos,
                past_key_values=past_key_values,
                use_cache=use_cache,
                return_dict=True,
                **generate_kwargs
            )

            logits_cls = outputs.logits[:, -1, :]       # (B, vocab_size)
            past_key_values = outputs.past_key_values

            # ignore specified tokens
            if ignore_Idx is not None:
                logits_cls[:, ignore_Idx] = float('-inf')

            sentence_cls.append(logits_cls.unsqueeze(1).clone())

            # top_k truncation
            if top_k is not None:
                v, topid = torch.topk(logits_cls, min(top_k, logits_cls.size(-1)))
                logits_cls[logits_cls < v[:, [-1]]] = float('-inf')

            # sample next entity token
            probs = F.softmax(logits_cls, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B,1)

            # append to current sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # create new cell_pos for the next token
            # obtain the last token's position value, ensure not exceeding max value
            new_cell_pos_token = torch.full((batch_size, 1), cell_pos.max(), dtype=cell_pos.dtype, device=cell_pos.device)

            cell_pos = torch.cat([cell_pos, new_cell_pos_token], dim=1)

            if debug:
                print(f"next_token: {next_token}, appended input_ids shape: {input_ids.shape}")
                print(f"new_cell_pos_token: {new_cell_pos_token}, appended cell_pos shape: {cell_pos.shape}")

            if (step+1) % self.vq_model.num_code == 0:
                inter_token = ['<E>', f'traj_{str(cell_pos.max().item())}', '<S>']
                inter_token = self.tokenizer.encode(inter_token)
                inter_pos = [cell_pos.max(), 1, cell_pos.max()+1]

                input_ids = torch.cat([input_ids, torch.tensor(inter_token, device=input_ids.device).repeat(batch_size,1)], dim=1)
                cell_pos = torch.cat([cell_pos, torch.tensor(inter_pos, device=input_ids.device).repeat(batch_size,1)], dim=1)

                sentence_cls = torch.concatenate(sentence_cls,axis=1) # [bs, tokens, vocab_size]
                probs = F.softmax(sentence_cls, dim=-1)[:,:,:512]

                # token-level entropy
                entropy = -(probs * probs.log()).sum(dim=-1)  # (batch, seq_len)
                # normalized_certainty = 1 - entropy / torch.log(torch.tensor(probs.shape[-1]))

                # sentence-level certainty
                sentence_certainty = torch.exp(-entropy.mean(dim=-1))  # (batch,)
                sentence_entropy.append(sentence_certainty.unsqueeze(-1))

                sentence_cls = []

        if debug:
            print("=== Generation Finished ===")
            print(f"Final generated input_ids: {input_ids}")
            print(f"Final generated cell_pos: {cell_pos}")
        sentence_entropy = torch.concat(sentence_entropy, axis=-1)

        return input_ids, sentence_entropy, cell_pos
    
    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu
