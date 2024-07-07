import torch
from torch.nn import functional as F
from .transformer import Transformer
from .tensors import sinusoids, list_to_tensors
from torch.nn.utils.rnn import pad_sequence
from torch.profiler import record_function

class SupervoceNARModel(torch.nn.Module):
    def __init__(self):
        super(SupervoceNARModel, self).__init__()

        self.n_dim = 1024
        self.max_seq_len = 8 * 1024

        self.transformer = Transformer(
            n_heads = 16,
            n_layers = 12,
            n_dim = self.n_dim,
            n_dim_head = 16, # n_dim // n_heads
            n_dim_ffn = 4096,
            att_dropout = 0,
            ffn_dropout = 0.1
        )

        # Sinusoidal positional embedding
        # self.register_buffer("positional_embedding", sinusoids(self.max_seq_len, self.n_dim))
        self.positional_embedding_text = torch.nn.Embedding(self.max_seq_len, self.n_dim)
        torch.nn.init.normal_(self.positional_embedding_text.weight, mean=0.0, std=0.02)
        self.positional_embedding_audio = torch.nn.Embedding(self.max_seq_len, self.n_dim)
        torch.nn.init.normal_(self.positional_embedding_audio.weight, mean=0.0, std=0.02)

        # Text Condition
        self.text_embedding = torch.nn.Embedding(8 * 1024, self.n_dim)
        torch.nn.init.normal_(self.text_embedding.weight, mean=0.0, std=0.02)

        # Audio embedding
        self.audio_embedding = torch.nn.Embedding(1024, self.n_dim)
        torch.nn.init.normal_(self.audio_embedding.weight, mean=0.0, std=0.02)

        # EOS embedding
        self.eos_embedding = torch.nn.Embedding(1, self.n_dim)
        torch.nn.init.normal_(self.eos_embedding.weight, mean=0.0, std=0.02)

        # Codec index embedding
        self.codec_index_embedding = torch.nn.Embedding(7, self.n_dim)
        torch.nn.init.normal_(self.codec_index_embedding.weight, mean=0.0, std=0.02)

        # Output prediction
        # self.prediction = torch.nn.Linear(self.n_dim, 1024, bias=False)
        self.prediction = torch.nn.Linear(self.n_dim, 1024)
        torch.nn.init.normal_(self.prediction.weight, mean=0.0, std=0.02)
        torch.nn.init.zeros_(self.prediction.bias)
        
    
    def forward(self, *, condition_text, condition_audio, audio, codec, loss = False):

        # Prepare
        with record_function("prepare"):
            # Check inputs
            device = condition_text[0].device
            B = len(condition_text)
            assert len(condition_audio) == B
            assert len(audio) == B
            assert len(codec) == B

            # Check shapes
            for b in range(B):

                # Check condition shape
                assert condition_text[b].dim() == 1, f"Unexpected shape: {condition_text[b].shape}"
                assert condition_audio[b].dim() == 2, f"Unexpected shape: {condition_audio[b].shape}"
                assert condition_audio[b].shape[0] == 8, f"Unexpected shape: {condition_audio[b].shape}"

                # Check codec value
                assert codec[b] >= 1 and codec[b] <= 7, f"Unexpected codec value: {codec[b]}"

                # Check audio shape
                assert audio[b].dim() == 2, f"Unexpected shape: {audio[b].shape}"
                assert audio[b].shape[0] >= codec[b], f"Unexpected shape: {audio[b].shape}"

            #
            # Prepare EOS
            #
            with record_function("prepare:eos"):
                eos = self.eos_embedding(torch.tensor([0]).to(device, non_blocking=True))
        
            #
            # Text embedding
            #
            with record_function("prepare:text"):
                l_t = []
                x_t = []
                for b in range(B):
                    t = torch.cat([self.text_embedding(condition_text[b]), eos])
                    # t = t + self.positional_embedding[:t.shape[0]]
                    t = t + self.positional_embedding_text(torch.arange(t.shape[0]).to(t.device, non_blocking=True))
                    x_t.append(t)
                    l_t.append(t.shape[0])

            #
            # Audio embedding
            #

            x_a = []
            l_c = []
            l_a = []
            for b in range(B):

                # Condition embedding
                t_c = self.audio_embedding(condition_audio[b][0])
                for i in range(1, condition_audio[b].shape[0]):
                    t_c = t_c + self.audio_embedding(condition_audio[b][i])

                # Audio embedding
                t_a = self.audio_embedding(audio[b][0])
                for i in range(1, codec[b]):
                    t_a = t_a + self.audio_embedding(audio[b][i])
            
                # Concatenate all
                t = torch.cat([t_c, t_a, eos])

                # Positional embedding
                # t = t + self.positional_embedding[:t.shape[0]]
                t = t + self.positional_embedding_audio(torch.arange(t.shape[0]).to(t.device, non_blocking=True))

                # Append
                x_a.append(t)
                l_c.append(t_c.shape[0])
                l_a.append(t_a.shape[0])

            #
            # Codec embedding
            #

            x_ci = []
            for b in range(B):
                t_ci = self.codec_index_embedding(torch.tensor([codec[b] - 1]).long().to(t.device, non_blocking=True))
                x_ci.append(t_ci)
        
            #
            # Concatenate all
            #

            x = []
            for b in range(B):
                x.append(torch.cat([x_t[b], x_a[b], x_ci[b]]))
            x, m = list_to_tensors(x)
            m = m.unsqueeze(-1).unsqueeze(-1)
            m = m.contiguous()

        #
        # Transform
        #

        x = self.transformer(x, mask = m)

        #
        # Predict
        #

        x = self.prediction(x)

        #
        # Extract predictions
        #

        predicted = []
        for i in range(B):
            p_s = l_t[i] + l_c[i]
            p_e = p_s + l_a[i]
            predicted.append(x[i, p_s:p_e])

        #
        # Loss
        #

        if loss:
            source = pad_sequence(predicted, batch_first=True, padding_value=-1)
            targets = pad_sequence([audio[i][codec[i]] for i in range(B)], batch_first=True, padding_value=-1)
            loss = F.cross_entropy(source.view(-1, source.size(-1)), targets.view(-1), ignore_index = -1)
            return predicted, loss
        else:
            return predicted
    