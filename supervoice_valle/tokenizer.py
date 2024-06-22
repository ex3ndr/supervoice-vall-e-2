import torch
import sentencepiece as spm

class Tokenizer:
    def __init__(self, path):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(path)

    def encode(self, text):
        return torch.tensor(self.sp.encode(text), dtype=torch.long).squeeze(0).squeeze(0)

    def encode_sample(self, text):
        return torch.tensor(self.sp.encode(text, enable_sampling=True, alpha=0.1, nbest_size=-1), dtype=torch.long).squeeze(0).squeeze(0)