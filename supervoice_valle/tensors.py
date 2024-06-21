import torch
import torch.nn.functional as F
from einops import rearrange
import random

class RMSNorm(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = torch.nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.scale * self.gamma
    
class AdaptiveRMSNorm(torch.nn.Module):
    def __init__(
        self,
        dim
    ):
        super().__init__()
        self.scale = dim ** 0.5
        self.to_gamma = torch.nn.Linear(dim, dim)
        self.to_beta = torch.nn.Linear(dim, dim)

        # Identity initialization
        torch.nn.init.zeros_(self.to_gamma.weight)
        torch.nn.init.ones_(self.to_gamma.bias)
        torch.nn.init.zeros_(self.to_beta.weight)
        torch.nn.init.zeros_(self.to_beta.bias)

    def forward(self, x, *, cond):
        normed = F.normalize(x, dim = -1) * self.scale
        gamma, beta = self.to_gamma(cond), self.to_beta(cond)
        gamma, beta = map(lambda t: rearrange(t, 'b d -> b 1 d'), (gamma, beta))

        return normed * gamma + beta

def probability_binary_mask(shape, true_prob, device):
    return torch.zeros(shape, device = device).float().uniform_(0, 1) < true_prob


def debug_if_invalid(x):
    if torch.isnan(x).any() or torch.isinf(x).any():
        print('Invalid tensor')

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def drop_using_mask(source, replacement, mask):
    while mask.dim() < source.dim():
        mask = mask.unsqueeze(-1)
    return torch.where(mask, torch.full(source.shape, replacement, dtype = source.dtype, device = source.device), source)

def merge_mask(source, replacement, mask):
    while mask.dim() < source.dim():
        mask = mask.unsqueeze(-1)
    return torch.where(mask, replacement, source)

def random_interval_masking(batch_size, length, *, min_size, min_count, max_count, device):
    tensor = torch.full((batch_size, length), False, device=device, dtype=torch.bool)
    for i in range(batch_size):

        # Expected sum of all intervals
        expected_length = random.randint(min_count, max_count)

        # Number of intervals
        num_intervals = random.randint(1, expected_length // min_size)

        # Generate interval lengths
        lengths = [min_size] * num_intervals
        for _ in range(expected_length - num_intervals * min_size):
            lengths[random.randint(0, num_intervals - 1)] += 1

        # Generate start points
        placements = []
        offset = 0
        remaining = expected_length
        for l in lengths:
            start_position = random.uniform(offset, remaining - l)
            placements.append(start_position)
            offset = start_position + l
            remaining -= l

        # Write to tensor
        for l, p in zip(lengths, placements):
            tensor[i, int(p):int(p + l)] = True

    return tensor
