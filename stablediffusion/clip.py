import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention


class CLIP_Embedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, max_length: int):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Parameter(torch.zeros(max_length, d_model))

        
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(tokens)
        x += self.position_embedding
        return x


class CLIP_Layer(nn.Module):
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        self.layernorm_1 = nn.LayerNorm(d_model)
        self.attention = SelfAttention(num_heads, d_model)
        self.layernorm_2 = nn.LayerNorm(d_model)
        self.linear_1 = nn.Linear(d_model, d_model * 4)
        self.linear_2 = nn.Linear(d_model * 4, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residue = x

        x = self.layernorm_1(x)
        x = self.attention(x, causal_mask = True)
        x = x + residue

        residue = x
        x = self.layernorm_2(x)
        x = self.linear_1(x)
        x = x * torch.sigmoid(1.702 * x)  # Quick Gelu activation
        x = self.linear_2(x)
        x += residue

        return x



class CLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = CLIP_Embedding(49408, 768, 77) # vocab_size, d_model, max_length
        self.layers = nn.ModuleList(
            [CLIP_Layer(12, 768) for i in range(12)]
        )

        self.layernorm = nn.LayerNorm(768)
    
    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        tokens = tokens.type(torch.long)
        state = self.embedding(tokens)

        for layer in self.layers:
            state = layer(state)
        
        output = self.layernorm(state)

        return output

