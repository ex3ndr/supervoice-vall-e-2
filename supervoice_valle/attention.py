import torch
from torch import nn
import math

class Attend(nn.Module):
    def __init__(self, *, heads, dropout = 0., engine = "direct"):
        super().__init__()
        self.heads = heads
        self.dropout = nn.Dropout(dropout)
        self.dropout_p = dropout
        self.engine = engine

    def forward(self, q, k, v, padding_mask = None):

        # Check argument shapes
        assert q.dim() == 4
        assert k.dim() == 4
        assert v.dim() == 4
        assert q.size(0) == k.size(0) == v.size(0), "Batch size mismatch"
        assert q.size(1) == k.size(1) == v.size(1) == self.heads, "Heads length mismatch"
        assert q.size(2) == k.size(2) == v.size(2), "Sequence length mismatch"
        assert q.size(3) == k.size(3) == v.size(3), "Embeddings dimensions mismatch"
        if padding_mask is not None:
            assert padding_mask.dim() == 2
            assert padding_mask.size(0) == q.size(0)
            assert padding_mask.size(2) == q.size(2)

        # Check padding mask
        if self.engine == "direct":
            return self.direct_attention(q, k, v, padding_mask)
        elif self.engine == "torch":
            return self.pytorch_attention(q, k, v, padding_mask)
        else:
            raise ValueError("Invalid engine")

    def pytorch_attention(self, q, k, v, padding_mask):
        (B, H, L, E) = q.size()

        # Calcualte output
        output = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout_p if self.training else 0.0)

        return output

    def direct_attention(self, q, k, v, padding_mask):
        (B, H, L, E) = q.size()

        # Attention bias
        # attn_bias = torch.zeros(L, S, dtype=q.dtype, device = q.device)
        
        # Similarity
        attn_weight = q @ k.transpose(-2, -1)
        attn_weight = attn_weight / math.sqrt(E)
        print(attn_weight.shape)

        # Softmax
        # attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)

        # Dropout
        attn_weight = torch.dropout(attn_weight, self.dropout_p, train=True)

        # Caluclate output
        output = attn_weight @ v

        return output