# import re
# from typing import Optional
# from dataclasses import dataclass

# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# @dataclass
# class ModelConfig:
#     vocab_size: int                     # vocab_size
#     d_model: int = 576                  # hidden_szie
#     d_head: int = 64                    # head_dim
#     d_mlp_proj: int = 1536              # intermediate_size 
#     n_kv_heads: int = 3                 # num_key_value_heads
#     n_attn_heads: int = 9               # num_attention_heads
#     n_layers: int = 30                  # num_hidden_layers
#     rms_norm_eps: float = 1e-5          # rms_norm_eps
#     rope_theta: float = 100000.0        # rope_theta
#     initializer_range: float = 0.02     # initializer_range = 0.041666666666666664
#     padding_idx: Optional[int] = None
#     tie_word_embeddings: bool = False   # tie_word_embeddings = True
#     debug: bool = True
    

# # class Rotary(nn.Module):
# #     def __init__(self, config):
        
# #     def forward(self, x, seq_dim=1):
        
    
# class GroupedQueryAttention(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.use_flash = hasattr(F, "scaled_dot_product_attention")
#         self.config = config
#         self.attn_scale = self.config.d_head ** -0.5
#         # self.q_proj = nn.Linear(self.config.d_model, self.config.n_attn_heads * self.config.d_head, bias=False)
#         # self.k_proj = nn.Linear(self.config.d_model, self.config.n_kv_heads * self.config.d_head, bias=False)
#         # self.v_proj = nn.Linear(self.config.d_model, self.config.n_kv_heads * self.config.d_head, bias=False)
        
#         self.q_proj = nn.Parameter(torch.empty((config.d_model, config.n_attn_heads, config.d_head)))
#         nn.init.normal_(self.q_proj, std=config.initializer_range)
        
#         self.k_proj = nn.Parameter(torch.empty((config.d_model, config.n_kv_heads, config.d_head)))
#         nn.init.normal_(self.k_proj, std=config.initializer_range)
        
#         self.v_proj = nn.Parameter(torch.empty((config.d_model, config.n_kv_heads, config.d_head)))
#         nn.init.normal_(self.v_proj, std=config.initializer_range)
        
#         self.o_proj = nn.Parameter()
        
#     @staticmethod
#     def _rotate_half(x):
#         """
#         given input, x = [a, b, c, d]
#         1. split the last dimension in half, eg: x1 = [a, b], x2 = [c, d]
#         2. rotate the halves
#             - negate the second half 
#             - concatentate -x2 and x1 along the last dimension)
#         3. return the rotated tensor, eg: [-c, -d, a, b]
#         """
#         half = x.shape[-1] // 2
#         x1, x2 = x[..., :half], x[..., half:]
#         return torch.cat([-x2, x1, dim=-1])
    
#     @staticmethod   
#     def _apply_rotary_pos_emb(q, k, cos, sin):
#         """
#         RoPE(x) = [x1⋅cosθ - x2⋅sinθ, x1⋅sinθ + x2⋅cosθ]
#         which is equivalent to:
#                 x⋅cosθ + rotate_half(x)⋅sinθ
#         """
#         return (
#             q*cos + self._rotate_half(q)*sin,  # rotated queries
#             k*cos + self._rotate_half(k)*sin   # rotated keys
#         )
        
#     def forward(self, x, cos, sin):
#         # x.shape: [batch_size, seq_len, d_model]
#         batch_size, seq_len, _ = x.shape
        
#         q = self.q_proj(x) # shape: [batch_size, seq_len, n_attn_heads*d_head]
#         k = self.k_proj(x) # shape: [batch_size, seq_len, n_kv_heads*d_head]
#         v = self.v_proj(v) # shape: [batch_size, seq_len, n_kv_heads*d_head]
        
#         q = q.view()
        
     
   
# # class GatedMlp(nn.Module):
# #     def __init__(self, config):
        
# #     def forward(self, x):
        
        
# # class DecoderLayer(nn.Module):
# #     def __init__(self, config):
        
# #     def forward(self, x, cos, sin):

    
# # class LlamaModel(nn.Module):
# #     def __init__(self, config):        
        
# #     def forward(self, x, y=None):
        
    
# def main():
#     config = ModelConfig(
#         vocab_size=49152,
#         d_model=576,
#         d_head=64,
#         d_mlp_proj=1536,
#         n_layers=30,
#         n_kv_heads=3,
#         n_attn_heads=9,
#         rms_norm_eps=1e-5,
#         initializer_range=0.041666666666666664,
#         rope_theta=100000.0
#     )
    
#     model = LlamaModel(config)
#     x = torch.randint(0, config.vocab_size, (4, 1024))
#     out, _ = model(x)
#     print(out.shape)
    
# if __name__ == "__main__":
#     main()