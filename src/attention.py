"""
Originally forked from Andrej Karpathy's minGPT.
"""

import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)


def precompute_rotary_emb(dim, max_positions):
    """
    RoPE uses the following sinusoidal functions to encode positions:

    cos(t theta_i) and sin(t theta_i)
        where t is the position and
              theta_i = 1/10000^(-2(i-1)/dim) for i in [1, dim/2]

    Since the maximum length of sequences is known, we can precompute
    these values to speed up training.

    Implement the precompute_rotary_emb function that returns a tensor of
    shape (max_positions, dim/2, 2) where the last dimension contains
    the cos and sin values for each position and each dimension of
    the embedding.
    """
    
    # We will create a position and dimension tensors here
    positions = torch.arange(max_positions).float()
    # Then we will create the dimension constants for each position 
    dim_tensor = torch.arange(0, dim, 2).float()
    
    # Calculate frequencies for each dimension
    freqs = 1.0 / (10000 ** (dim_tensor / dim))
    
    # This will be the outer product of positions and frequencies
    # and it will give us t*theta_i for all positions and dimensions passed
    t_theta = torch.outer(positions, freqs)
    
    # Calculating the  cos and sin values here
    cos_values = torch.cos(t_theta)
    sin_values = torch.sin(t_theta)
    
    # I will stack them together in the last dimension
    rope_cache = torch.stack([cos_values, sin_values], dim=-1)
    
    #Finally return the results
    return rope_cache


def apply_rotary_emb(x, rope_cache):
    """Apply the RoPE to the input tensor x.
    
    This implementation treats each pair of dimensions as a complex number:
    - Even indices (0, 2, 4...) are treated as the real part
    - Odd indices (1, 3, 5...) are treated as the imaginary part
    
    The rotation is applied using the complex multiplication formula:
    (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
    
    Where (a + bi) is the embedding and (c + di) is cos(θ) + i*sin(θ)
    """
    # Getting the dimensions
    B, T, hs = x.shape  # Batch, Sequence length, Head size
    
    # Ensure head size is even (required for complex number interpretation)
    assert hs % 2 == 0, "Head size must be even for RoPE"
    
    # Truncate the cache if necessary
    rope_cache_truncated = rope_cache[:T]
    
    # Reshape for proper application of rotary embeddings
    # Treat each consecutive pair of dimensions as a complex number
    x_reshape = x.reshape(B, T, hs // 2, 2)
    
    # Extracting the cos and sin from the cache
    cos = rope_cache_truncated[..., 0]  # Shape: [T, hs//2]
    sin = rope_cache_truncated[..., 1]  # Shape: [T, hs//2]
    
    # Preparing the cos and sin for broadcasting
    cos = cos.unsqueeze(0)  # [1, T, hs//2]
    sin = sin.unsqueeze(0)  # [1, T, hs//2]
    
    # Extracting the real and imaginary parts
    x_real = x_reshape[..., 0]  # [B, T, hs//2]
    x_imag = x_reshape[..., 1]  # [B, T, hs//2]
    
    # Applying rotation using complex multiplication formula:
    # (a + bi) * (cos(θ) + i*sin(θ)) = (a*cos(θ) - b*sin(θ)) + i(a*sin(θ) + b*cos(θ))
    out_real = x_real * cos - x_imag * sin  # Real part
    out_imag = x_real * sin + x_imag * cos  # Imaginary part
    
    # Combining and reshaping back
    out = torch.stack([out_real, out_imag], dim=-1)  # [B, T, hs//2, 2]
    out = out.reshape(B, T, hs)
    
    return out

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    I believe I could have just used torch.nn.MultiheadAttention but their documentation
    is all but absent and code ugly so I don't trust it, rolling my own here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)

        self.rope = config.rope
        if self.rope:
            assert (config.n_embd % config.n_head) % 2 == 0

            # Compute the head dimension
            head_dim = config.n_embd // config.n_head
            # Precompute the rotary embeddings
            rope_cache = precompute_rotary_emb(head_dim, config.block_size)

            self.register_buffer("rope_cache", rope_cache)

        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head

    def forward(self, x):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        if self.rope:
            # Apply rotary embeddings to query and key for each head separately
            for h in range(self.n_head):
                q[:, h] = apply_rotary_emb(q[:, h], self.rope_cache)
                k[:, h] = apply_rotary_emb(k[:, h], self.rope_cache)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, -1e10)
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class CausalCrossAttention(nn.Module):
    """
    Modifications over the self-attention layer to handle two inputs and perform
    cross-attention between them.
    This follows the implementation of the self attention module with
    auto-regressive masking on (key).
    Manipulation of batch-size to allow for different batch size between the
    two inputs, with broadcasting over to the higher batch size value.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head

    def forward(self, x_kv, x_q):
        Bk, Tk, Ck = x_kv.size()
        Bq, Tq, Cq = x_q.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim

        # keys of x1
        k = self.key(x_kv).view(Bk, Tk, self.n_head, Ck // self.n_head).transpose(1, 2) # (B, nh, Tk, hs)

        # query with x2
        q = self.query(x_q).view(Bq, Tq, self.n_head, Cq // self.n_head).transpose(1, 2) # (B, nh, Tq, hs)

        # values from x1
        v = self.value(x_kv).view(Bk, Tk, self.n_head, Ck // self.n_head).transpose(1, 2) # (B, nh, Tk, hs)

        # causal self-attention;  (B, nh, Tk, hs) x (B, nh, hs, Tq) -> (B, nh, Tq, Tk)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        B = max(Bk, Bq)

        att = att.masked_fill(self.mask[:,:,:Tq,:Tk] == 0, -1e10)
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, Tq, Tk) x (B, nh, Tk, hs) -> (B, nh, Tq, hs)
        y = y.transpose(1, 2).contiguous().view(B, Tq, Cq) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y
