import torch
from torch import nn
from torch.nn import functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, num_heads: int, d_embed: int, in_proj_bias = True, out_proj_bias = True):
        super().__init__()
        assert d_embed % num_heads == 0, "d_embed must be divisible by num_heads"

        self.num_heads = num_heads
        self.d_head = d_embed // num_heads
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)

    def forward(self, x: torch.Tensor, causal_mask: bool = False) -> torch.Tensor:
        input_shape = x.shape
        batch_size, seq_len, d_embed = input_shape

        interm_shape = (batch_size, seq_len, self.num_heads, self.d_head)

        # Following can be replaced with
        # def to_heads(t):  # (b,s,d) -> (b,h,s,dh)
        #     return t.view(b, s, h, dh).transpose(1, 2)

        # q, k, v = map(to_heads, (q, k, v))

        q, k, v = self.in_proj(x).chunk(3, dim = -1)

        # (batch_size, seq_len, d_embed) -> (batch_size, seq_len, num_heads, d_head) -> (batch_size, num_heads, seq_len, d_head)
        q = q.view(interm_shape).transpose(1, 2)
        k = k.view(interm_shape).transpose(1, 2)
        v = v.view(interm_shape).transpose(1, 2)

        # PyTorch SDPA handles scaling, softmax, and matmul efficiently
        # TODO: replace with F.scaled_dot_product_attention

        weights = q @ k.transpose(-1, -2)

        if causal_mask:
            # Mask where the upper triangle (above the principal diagonal) is 1
            mask = torch.ones_like(weights, dtype=torch.bool).triu(1) 
            # Fill the upper triangle with -inf
            weights.masked_fill_(mask, -torch.finfo(weights.dtype).max) 
        
        weights /= math.sqrt(self.d_head)
        weights = F.softmax(weights, dim = -1)

        output = weights @ v
        output = output.transpose(1, 2)
        output = output.reshape(input_shape)
        output = self.out_proj(output)
        return output


class CrossAttention(nn.Module):
    def __init__(self, num_heads: int, d_embed: int, d_cross: int, in_proj_bias = True, out_proj_bias = True):
        super().__init__()
        self.q_proj = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.v_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.num_heads = num_heads
        self.d_head = d_embed // num_heads

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x: batch_size, seq_len_q, dim_q (latent)
        # y: batch_size, seq_len_kv, dim_kv (context)

        input_shape = x.shape
        batch_size, seq_len_q, d_embed = input_shape

        interm_shape = (batch_size, -1, self.num_heads, self.d_head)

        q = self.q_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)

        q = q.view(interm_shape).transpose(1, 2)
        k = k.view(interm_shape).transpose(1, 2)
        v = v.view(interm_shape).transpose(1, 2)
        
        weight = q @ k.tranpose(-1, -2)
        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight, dim = -1)

        output = weight @ v
        output = output.transpose(1, 2).contiguous()
        output = output.reshape(input_shape)
        output = self.out_proj(output)
        return output
