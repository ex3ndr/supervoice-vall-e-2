import torch
from torch import nn
import math
import torch.nn.functional as F
from einops import rearrange, repeat, reduce, pack, unpack
from torch.cuda.amp import autocast
from .tensors import RMSNorm
# from flash_attn import flash_attn_func
from torch.profiler import record_function

class Transformer(nn.Module):
    def __init__(self, 
        n_heads,
        n_layers,
        n_dim,
        n_dim_head,
        n_dim_ffn,
        att_dropout, 
        ffn_dropout,
        enable_skip_connections = False
    ):
        super(Transformer, self).__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.enable_skip_connections = enable_skip_connections

        # Attention blocks
        self.layers = torch.nn.ModuleList([])
        for i in range(n_layers):
            self.layers.append(AttentionBlock(
                n_heads = n_heads, 
                n_dim = n_dim, 
                n_dim_head = n_dim_head, 
                n_dim_ffn = n_dim_ffn,
                att_dropout = att_dropout,
                ffn_dropout = ffn_dropout
            ))
        
        # Skip connections
        self.skip_combiners = torch.nn.ModuleList([])
        if enable_skip_connections:
            for i in range(n_layers//2):
                self.skip_combiners.append(torch.nn.Linear(n_dim * 2, n_dim))

        # Output normalization
        self.output_norm = RMSNorm(n_dim)

    def forward(self, x, mask = None, casual = False):
        batch, seq_len, *_ = x.shape

        # Run through attention blocks
        connections = []
        for i in range(self.n_layers):

            # Skip connection
            if self.n_layers - (self.n_layers // 2) < i and self.enable_skip_connections:
                s = connections.pop()
                x = torch.cat([x, s], dim = -1)
                x = self.skip_combiners[i - (self.n_layers // 2)](x)

            # Attention
            with record_function("attention"):
                x = self.layers[i](x, mask = mask, casual = casual)

            # Skip connection
            if i <= self.n_layers // 2:
                connections.append(x)

        # Output normalization
        x = self.output_norm(x)

        # Result
        return x


class AttentionBlock(torch.nn.Module):
    def __init__(self, n_heads, n_dim, n_dim_head, n_dim_ffn, att_dropout, ffn_dropout):
        super(AttentionBlock, self).__init__()

        self.n_heads = n_heads
        self.n_dim_head = n_dim_head
        self.att_dropout = att_dropout

        # Attention input layer norm
        self.attention_ln = RMSNorm(n_dim)

        # Input -> Query/Key/Value for each head in single tensor for speedup
        self.attention = nn.Linear(n_dim, 3 * n_dim_head * n_heads, bias=False)
        torch.nn.init.normal_(self.attention.weight, mean=0.0, std=0.02)

        # Attention dropout
        # self.attention_dropout = nn.Dropout(att_dropout)

        # Output flatten multiple heads into single tensor
        # self.attention_output = nn.Linear(n_dim_head * n_heads, n_dim, bias=False)
        self.attention_output = nn.Linear(n_dim_head * n_heads, n_dim)
        torch.nn.init.normal_(self.attention_output.weight, mean=0.0, std=0.02)
        torch.nn.init.zeros_(self.attention_output.bias)

        # MLP part
        self.mlp_ln = RMSNorm(n_dim)
        
        self.mlp_input = nn.Linear(n_dim, n_dim_ffn)
        torch.nn.init.normal_(self.mlp_input.weight, mean=0.0, std=0.02)
        torch.nn.init.zeros_(self.mlp_input.bias)

        self.mlp_output = nn.Linear(n_dim_ffn, n_dim)
        torch.nn.init.normal_(self.mlp_output.weight, mean=0.0, std=0.02)
        torch.nn.init.zeros_(self.mlp_output.bias)

        self.mlp_output_dropout = nn.Dropout(ffn_dropout)

    def forward(self, x, mask = None, casual = False):

        with record_function("attention:pre"):
            B, T, C = x.size() # batch size, sequence length, context width

            # Residual
            residual = x

            # Input normalization
            y = self.attention_ln(x)

            # Calculation Q/K/V for each head
            q, k, v = self.attention(y).chunk(3, dim = -1)

            # Reshape for head-first attention
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.n_heads), (q, k, v))

            # Prepare mask
            # mask = None
            # if padding_mask is not None:
            #     mask = rearrange(mask, "b j -> b 1 1 j")
            #     mask = mask.expand(-1, self.d_heads, q_len, -1)
            
            # Run through attention
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=self.att_dropout if self.training else 0.0, attn_mask = mask, is_causal = casual)

            # Reshape back
            y = rearrange(y, 'b h n d -> b n (h d)')

            # Output
            y = self.attention_output(y)

            # Residual
            y = residual + y
            residual = y

        with record_function("attention:post-post"):
            # MLP
            y = self.mlp_ln(y)
            y = self.mlp_input(y)
            y = F.gelu(y)
            y = self.mlp_output_dropout(y)
            y = self.mlp_output(y)
            y = residual + y

        return y