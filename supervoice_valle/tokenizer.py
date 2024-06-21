import torch
import sentencepiece as spm

class Tokenizer:
    def __init__(self, path):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(path)

    def encode(self, text):
        return torch.tensor(self.sp.encode(text), dtype=torch.long)