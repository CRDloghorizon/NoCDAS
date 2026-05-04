import torch.nn.functional as F
import torch.nn as nn
import torch
import math

def apply_rotary_emb(x):
    """
    Apply RoPE (Rotary Positional Embedding)
    x shape: [Batch, num_heads, Seq_Len, head_dim]
    """
    B, n_heads, T, head_dim = x.shape
    half_dim = head_dim // 2
    
    # Calculate frequencies (base 10000.0) as in MAC.cpp
    feat_idx = torch.arange(half_dim, dtype=torch.float32, device=x.device)
    freq = 1.0 / (10000.0 ** ((feat_idx * 2.0) / head_dim))
    
    # theta = pos * freq
    pos = torch.arange(T, dtype=torch.float32, device=x.device)
    theta = torch.outer(pos, freq)  # Shape: [T, half_dim]
    
    # broadcasting [1, 1, T, half_dim]
    theta = theta.unsqueeze(0).unsqueeze(0)
    cos_val = torch.cos(theta)
    sin_val = torch.sin(theta)
    
    # splitting the tensor in val_curr and val_pair
    x1 = x[..., :half_dim]
    x2 = x[..., half_dim:]
    
    # ratation
    out1 = x1 * cos_val - x2 * sin_val
    out2 = x2 * cos_val + x1 * sin_val
    
    return torch.cat([out1, out2], dim=-1)

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, nhead, num_kv_heads):
        super().__init__()
        self.num_heads = nhead
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // nhead
        
        assert nhead % num_kv_heads == 0, "nhead must be divisible by num_kv_heads"
        
        self.q_proj = nn.Linear(d_model, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.num_kv_heads * self.head_dim, bias=False)
        
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, C = x.size()
        
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rotary_emb(q)
        k = apply_rotary_emb(k)

        num_queries_per_kv = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(num_queries_per_kv, dim=1)
        v = v.repeat_interleave(num_queries_per_kv, dim=1)

        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.out_proj(attn_out)

class ModernTransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, num_kv_heads, dim_feedforward):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = GroupedQueryAttention(d_model, nhead, num_kv_heads)
        self.norm2 = RMSNorm(d_model)
        
        self.gate_up_proj = nn.Linear(d_model, dim_feedforward * 2, bias=False)
        self.down_proj = nn.Linear(dim_feedforward, d_model, bias=False)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        
        res = x
        x = self.norm2(x)
        
        x = self.gate_up_proj(x)
        
        gate, up = x.chunk(2, dim=-1)
        x = F.silu(gate) * up
        
        x = self.down_proj(x)
        
        return res + x

# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_len = 5000):
#         super().__init__()
#         pe          = torch.zeros(max_len, d_model)
#         position    = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term    = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         self.register_buffer('pe', pe.unsqueeze(0))

#     def forward(self, x):
#         return x + self.pe[:, :x.size(1), :]

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_kv_heads=2, num_layers=2, dim_feedforward=256, max_seq_len=100):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        # self.pos_encoder = PositionalEncoding(d_model, max_seq_len)
        
        self.layers = nn.ModuleList([
            ModernTransformerBlock(d_model, nhead, num_kv_heads, dim_feedforward)
            for _ in range(num_layers)
        ])
        
        self.final_norm = RMSNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        x = self.embedding(x) * math.sqrt(self.d_model)
        # x = self.pos_encoder(x)
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.final_norm(x)
        out = self.fc_out(x)
        return out