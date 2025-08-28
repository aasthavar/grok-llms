import math
import einops
from fancy_einsum import einsum
from dataclasses import dataclass

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn

from transformer_lens.utils import gelu_new
# NOTE: 'position' here means 'sequence length' of input

@dataclass
class Config:
    d_vocab: int = 50257         # size of vocab (number of unique tokens)
    d_model: int = 768           # size of residual stream
    d_head: int = 64             # dimension of attention head
    d_mlp: int = 3072            # dimension of mlp layer
    
    n_layers: int = 12           # number of transformer layers
    n_heads: int = 12            # number of attention heads in MHA
    n_ctx: int = 1024            # max context length (number of tokens model can attend to)

    layer_norm_eps: float = 1e-5 # epsilon added to LayerNorm to prevent division by zero 
    init_range: float = 0.02     # std deviation for weight intialization
    debug: bool = True           # flag to enable extra logging
        

class Embed(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.W_E = nn.Parameter(torch.empty((config.d_vocab, config.d_model)))
        nn.init.normal(self.W_E, std=config.init_range)

    def forward(self, tokens):
        # tokens: [batch, position]
        if self.config.debug: print(f"tokens: {tokens.shape}") # tensor of token ids
        # embed: [batch, position, d_model]
        embed = self.W_E[tokens, :] # lookup in embedding matrix for each token id
        if self.config.debug: print(f"embed: {embed.shape}")
        return embed


class PosEmbed(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.W_pos = nn.Parameter(torch.empty((config.n_ctx, config.d_model)))
        nn.init.normal(self.W_pos, std=config.init_range)

    def forward(self, tokens):
        # tokens: [batch, position]
        if self.config.debug: print(f"tokens: {tokens.shape}")
        # pos_embed = [position, d_model]
        pos_embed = self.W_pos[:tokens.size(1), :]
        # pos_embed = [batch, position, d_model]
        pos_embed = einops.repeat(
            pos_embed,
            "position d_model -> batch position d_model",
            batch=tokens.size(0)
        )
        if self.config.debug: print(f"pos_embed: {pos_embed.shape}")
        return pos_embed


class LayerNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.w = nn.Parameter(torch.ones(config.d_model))
        self.b = nn.Parameter(torch.zeros(config.d_model))
        
    def forward(self, residual):
        # residual: [batch, position, d_model]
        if self.config.debug: print(f"residual: {residual.shape}")
        residual = residual - einops.reduce(
            residual,
            "batch position d_model -> batch position 1",
            "mean"
        )
        scale = (einops.reduce(
            residual.pow(2),
            "batch position d_model -> batch position 1",
            "mean"
        ) + self.config.layer_norm_eps).sqrt()
        normalized = residual / scale
        normalized = normalized * self.w + self.b
        
        if self.config.debug: print(f"normalized: {normalized.shape}")
        return normalized        


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.W_Q = nn.Parameter(torch.empty((config.n_heads, config.d_model, config.d_head)))
        nn.init.normal(self.W_Q, std=config.init_range)
        self.b_Q = nn.Parameter(torch.zeros((config.n_heads, config.d_head)))
        
        self.W_K = nn.Parameter(torch.empty((config.n_heads, config.d_model, config.d_head)))
        nn.init.normal(self.W_K, std=config.init_range)
        self.b_K = nn.Parameter(torch.zeros((config.n_heads, config.d_head)))
        
        self.W_V = nn.Parameter(torch.empty((config.n_heads, config.d_model, config.d_head)))
        nn.init.normal(self.W_V, std=config.init_range)
        self.b_V = nn.Parameter(torch.zeros((config.n_heads, config.d_head)))

        self.W_O = nn.Parameter(torch.empty((config.n_heads, config.d_head, config.d_model)))
        nn.init.normal(self.W_O, std=config.init_range)
        self.b_O = nn.Parameter(torch.zeros((config.d_model)))
        
        self.register_buffer(
            "EPSILON",
            torch.tensor(-1e5, dtype=torch.float32, device="cuda")
        )

    def forward(self, normalized_resid_pre):
        # normalized_resid_pre: [batch, position, d_model]
        if self.config.debug: print(f"normalized_resid_pre: {normalized_resid_pre.shape}")
        q = einsum(
            "batch query_pos d_model, n_heads d_model d_head -> batch query_pos n_heads d_head",
            normalized_resid_pre,
            self.W_Q
        ) + self.b_Q

        k = einsum(
            "batch key_pos d_model, n_heads d_model d_head -> batch key_pos n_heads d_head",
            normalized_resid_pre,
            self.W_K
        ) + self.b_K
        
        v = einsum(
            "batch key_pos d_model, n_heads d_model d_head -> batch key_pos n_heads d_head",
            normalized_resid_pre,
            self.W_V
        ) + self.b_V

        attn_scores = einsum(
            "batch query_pos n_heads d_model, batch key_pos n_heads d_model -> batch n_heads query_pos key_pos",
            q, 
            k
        )
        attn_scores = attn_scores / math.sqrt(self.config.d_head)
        attn_scores = self.apply_causal_mask(attn_scores)
        
        pattern = attn_scores.softmax(dim=-1)
        
        z = einsum(
            "batch n_heads query_pos key_pos, batch key_pos n_heads d_head -> batch query_pos n_heads d_head",
            pattern,
            v
        )
        
        attn_out = einsum(
            "batch query_pos n_heads d_head, n_heads d_head d_model -> batch query_pos d_model",
            z,
            self.W_O
        ) + self.b_O
        
        if self.config.debug: print(f"attn_out: {attn_out.shape}")
        return attn_out
        
    def apply_causal_mask(self, attn_scores):
        # attn_scores: [batch, n_heads, query_pos, key_pos] 
        mask = torch.triu(torch.ones(attn_scores.size(-2), attn_scores.size(-1), device=attn_scores.device), diagonal=1).bool()
        attn_scores.masked_fill(mask, self.EPSILON)
        return attn_scores
        
    
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.W_in = nn.Parameter(torch.empty((config.d_model, config.d_mlp)))
        nn.init.normal_(self.W_in, std=config.init_range)
        self.b_in = nn.Parameter(torch.zeros((config.d_mlp)))
        
        self.W_out = nn.Parameter(torch.empty((config.d_mlp, config.d_model)))
        nn.init.normal_(self.W_out, std=config.init_range)
        self.b_out = nn.Parameter(torch.zeros((config.d_model)))

    def forward(self, normalized_resid_mid):
        pre = einsum(
            "batch position d_model, d_model d_mlp -> batch position d_mlp",
            normalized_resid_mid,
            self.W_in
        ) + self.b_in
        
        post = gelu_new(pre)
        

    
# class TransformerBlock(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
        

#     def forward(self, ):

# class Unembed(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
        

#     def forward(self, ):
    
# class DemoTransformer(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
        

#     def forward(self, ):
    
    
if __name__ == "__main__":
    config = Config()
    print(config)
    