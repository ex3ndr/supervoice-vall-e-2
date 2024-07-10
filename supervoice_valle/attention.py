import torch
from torch import nn
import math
from einops import rearrange
import xformers.ops as xops
from flash_attn import flash_attn_func, flash_attn_varlen_func

class Attend(torch.nn.Module):
    def __init__(self, *, heads, engine = "direct"):
        super().__init__()
        self.heads = heads
        self.engine = engine

    def forward(self, q, k, v, lenghts = None):

        # Check argument shapes
        assert q.dim() == 4
        assert k.dim() == 4
        assert v.dim() == 4
        assert q.size(0) == k.size(0) == v.size(0), "Batch size mismatch"
        assert q.size(1) == k.size(1) == v.size(1), "Sequence length mismatch"
        assert q.size(2) == k.size(2) == v.size(2) == self.heads, "Heads length mismatch"
        assert q.size(3) == k.size(3) == v.size(3), "Embeddings dimensions mismatch"
        if lenghts is not None:
            assert sum(lenghts) == q.size(1)

        # Check padding mask
        if self.engine == "direct":
            return self.direct_attention(q, k, v, lenghts)
        elif self.engine == "torch":
            return self.pytorch_attention(q, k, v, lenghts)
        elif self.engine == "xformers":
            return self.xformers_attention(q, k, v, lenghts)
        elif self.engine == "flash":
            return self.flash_attention(q, k, v, lenghts)
        else:
            raise ValueError("Invalid engine")

    def flash_attention(self, q, k, v, lengths):
        (B, L, H, E) = q.size()

        # With lengths
        if lengths is not None:

            # Max lengths
            max_len = torch.tensor(max(lengths), dtype = q.dtype, device = q.device)

            # Seq lens
            seqlens = [0]
            last = 0
            for l in lengths:
                last += l
                seqlens.append(last)
            seqlens = torch.tensor(seqlens, dtype = torch.int32, device = q.device)

            return flash_attn_varlen_func(q.squeeze(0), k.squeeze(0), v.squeeze(0), seqlens, seqlens, max_len, max_len).unsqueeze(0)

        # Non length
        return flash_attn_func(q, k, v)


    def xformers_attention(self, q, k, v, lenghts):
        (B, L, H, E) = q.size()

        # Attention bias
        attn_bias = None
        if lenghts is not None:
            attn_bias = xops.fmha.BlockDiagonalMask.from_seqlens(lenghts)

        # Calcualte output
        output = xops.memory_efficient_attention(q, k, v, attn_bias = attn_bias)

        return output

    def pytorch_attention(self, q, k, v, lenghts):
        (B, L, H, E) = q.size()

        # Transpose
        q = rearrange(q, 'B L H E -> B H L E')
        k = rearrange(k, 'B L H E -> B H L E')
        v = rearrange(v, 'B L H E -> B H L E')

        # Attention bias
        attn_bias = None
        if lenghts is not None:
            attn_bias = create_block_mask(lenghts, q.device)
            attn_bias = torch.where(attn_bias, 0, torch.tensor(-10000.0, dtype = q.dtype))

        # Calcualte output
        output = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask = attn_bias)
        output = output.transpose(1, 2)

        return output

    def direct_attention(self, q, k, v, lenghts):
        (B, L, H, E) = q.size()

        # Transpose
        q = rearrange(q, 'B L H E -> B H L E')
        k = rearrange(k, 'B L H E -> B H L E')
        v = rearrange(v, 'B L H E -> B H L E')

        # Similarity
        scale = 1 / math.sqrt(E)
        attn_weight = q @ k.transpose(-2, -1)
        attn_weight = attn_weight * scale

        # Attention bias
        if lenghts is not None:
            attn_bias = create_block_mask(lenghts, q.device)
            attn_bias = torch.where(attn_bias, 0, torch.tensor(-10000.0, dtype = q.dtype))
            attn_weight += attn_bias

        # Softmax
        attn_weight = torch.softmax(attn_weight, dim=-1)

        # Caluclate output
        output = attn_weight @ v

        return output.transpose(1, 2)


def create_block_mask(lengths, device):
    L = sum(lengths)
    mask = torch.zeros(L, L, dtype = torch.bool, device = device)
    for i in range(len(lengths)):
        mask[sum(lengths[:i]):sum(lengths[:i + 1]), sum(lengths[:i]):sum(lengths[:i + 1])] = 1
    return mask
