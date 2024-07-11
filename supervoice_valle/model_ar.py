import torch
from torch.nn import functional as F
from .transformer import Transformer
from .tensors import sinusoids, list_to_tensors
from torch.nn.utils.rnn import pad_sequence
from torch.profiler import record_function

class SupervoceARModel(torch.nn.Module):
    def __init__(self):
        super(SupervoceARModel, self).__init__()
        self.n_dim = 1024
        self.max_seq_len = 8 * 1024

        # Main transformer
        self.transformer = Transformer(
            n_heads = 16,
            n_layers = 12,
            n_dim = self.n_dim,
            n_dim_head = 16, # n_dim // n_heads
            n_dim_ffn = 4096,
            att_dropout = 0,
            ffn_dropout = 0.1
        )

        # Positional embeddings
        self.positional_embedding_text = torch.nn.Embedding(self.max_seq_len, self.n_dim)
        torch.nn.init.normal_(self.positional_embedding_text.weight, mean=0.0, std=0.02)
        self.positional_embedding_audio = torch.nn.Embedding(self.max_seq_len, self.n_dim)
        torch.nn.init.normal_(self.positional_embedding_audio.weight, mean=0.0, std=0.02)

        # Text embeddings
        self.text_embedding = torch.nn.Embedding(8 * 1024, self.n_dim)
        torch.nn.init.normal_(self.text_embedding.weight, mean=0.0, std=0.02)

        # Audio embeddings
        self.audio_embedding = torch.nn.Embedding(1024, self.n_dim)
        torch.nn.init.normal_(self.audio_embedding.weight, mean=0.0, std=0.02)

        # EOS embedding
        self.eos_embedding = torch.nn.Embedding(1, self.n_dim)
        torch.nn.init.normal_(self.eos_embedding.weight, mean=0.0, std=0.02)

        # BOS embedding
        self.bos_embedding = torch.nn.Embedding(1, self.n_dim)
        torch.nn.init.normal_(self.bos_embedding.weight, mean=0.0, std=0.02)

        # Output prediction
        self.prediction = torch.nn.Linear(self.n_dim, 1025)
        torch.nn.init.normal_(self.prediction.weight, mean=0.0, std=0.02)
        torch.nn.init.zeros_(self.prediction.bias)

    def forward(self, *, text, audio, loss = False):
        #
        # Check shapes
        #

        device = text[0].device
        B = len(text)
        assert len(text) == B
        assert len(audio) == B
        for b in range(B):
            assert text[b].dim() == 1, f"Unexpected shape: {text[b].shape}"
            assert audio[b].dim() == 1, f"Unexpected shape: {audio[b].shape}"

        #
        # Prepare EOS/BOS
        #

        eos = self.eos_embedding(torch.tensor([0]).to(device, non_blocking=True))
        bos = self.bos_embedding(torch.tensor([0]).to(device, non_blocking=True))

        #
        # Text embedding
        #

        l_t = []
        x_t = []
        for b in range(B):

            # Text
            t = torch.cat([self.text_embedding(text[b]), eos])

            # Positional embedding
            t = t + self.positional_embedding_text(torch.arange(t.shape[0]).to(t.device, non_blocking=True))

            # Append
            x_t.append(t)
            l_t.append(t.shape[0])

        #
        # Audio embedding
        #

        x_a = []
        l_a = []
        for b in range(B):

            # Audio
            t = torch.cat([bos, self.audio_embedding(audio[b])])

            # Positional embedding
            t = t + self.positional_embedding_audio(torch.arange(t.shape[0]).to(t.device, non_blocking=True))

            # Append
            x_a.append(t)
            l_a.append(t.shape[0])


        #
        # Concatenate
        #

        x = []
        for b in range(B):
            x.append(torch.cat([x_t[b], x_a[b]]))
        x, m = list_to_tensors(x)


        #
        # Transform
        #
        
        x = self.transformer(x, casual = True)

        #
        # Predict
        #

        x = self.prediction(x)

        #
        # Extract predictions
        #

        predicted = []
        for i in range(B):
            p_s = l_t[i]
            p_e = p_s + l_a[i]
            predicted.append(x[i, p_s:p_e])

        #
        # Loss
        #

        if loss:
            source = pad_sequence(predicted, batch_first=True, padding_value=-1)
            targets = pad_sequence([torch.cat([audio[i], torch.tensor([1024], dtype = torch.int).to(t.device, non_blocking=True)]) for i in range(B)], batch_first=True, padding_value=-1)
            loss = F.cross_entropy(source.view(-1, source.size(-1)), targets.view(-1), ignore_index = -1)
            return predicted, loss
        else:
            return predicted