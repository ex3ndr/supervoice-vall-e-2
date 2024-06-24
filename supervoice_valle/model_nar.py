import torch
from torch.nn import functional as F
from .transformer import Transformer
from .tensors import sinusoids, list_to_tensors

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
        self.register_buffer("positional_embedding", sinusoids(self.max_seq_len, self.n_dim))

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
        self.prediction = torch.nn.Linear(self.n_dim, 1024, bias=False)
    
    def forward(self, *, condition_text, condition_audio, audio, codec, loss = False):
        device = condition_text[0].device

        # print(condition_text.shape, condition_audio.shape, audio.shape, codec)

        # Check shapes
        # assert codec >= 1 and codec <= 7
        # # [N]
        # assert len(condition_text.shape) == 1
        #  # [8, N]
        # assert len(condition_audio.shape) == 2
        # assert condition_audio.shape[0] == 8
        #  # [<codec>, N]
        # assert len(audio.shape) == 2
        # assert audio.shape[0] >= codec

        # Prepare EOS
        eos = self.eos_embedding(torch.tensor([0], device = device))
        
        #
        # Text embedding
        #
        l_t = []
        x_t = []
        for i in range(len(condition_text)):
            t = torch.cat([self.text_embedding(condition_text[i]), eos])
            t = t + self.positional_embedding[:t.shape[0]]
            x_t.append(t)
            l_t.append(t.shape[0])

        #
        # Audio embedding
        #

        x_a = []
        l_c = []
        l_a = []
        for b in range(len(condition_audio)):

            # Condition embedding
            t_c = self.audio_embedding(condition_audio[b][0])
            for i in range(1, condition_audio[b].shape[0]):
                t_c = t_c + self.audio_embedding(condition_audio[b][i])

            # Audio embedding
            t_a = self.audio_embedding(audio[b][0])
            for j in range(1, codec[b] - 1):
                t_a = t_a + self.audio_embedding(audio[b][i])
            
            # Codec embedding
            t_ci = self.codec_index_embedding(torch.tensor([codec[b] - 1], device = device).long())

            # Concatenate all
            t = torch.cat([t_c, t_a, eos, t_ci])

            # Positional embedding
            t[:t.shape[0] - 1] = t[:t.shape[0] - 1] + self.positional_embedding[:t.shape[0] - 1]

            # Append
            x_a.append(t)
            l_c.append(t_c.shape[0])
            l_a.append(t_a.shape[0])
        
        #
        # Concatenate all
        #

        x = []
        x_l = []
        for i in range(len(x_t)):
            x.append(torch.cat([x_t[i], x_a[i]]))
            x_l.append(x[i].shape[0])

            
        x, m = list_to_tensors(x)
        m = m.unsqueeze(-1).unsqueeze(-1)
        # s = ""
        # for i in range(len(m)):
        #     for j in range(len(m[i])):
        #         if m[i][j][0][0]:
        #             s += "1"
        #         else:
        #             s += "0"
        #     s += "\n"
        # print(s)
        # print("1", torch.isnan(x).any(), torch.isinf(x).any())

        # Transform
        x = self.transformer(x, mask = m)
        # print("2", torch.isnan(x).any(), torch.isinf(x).any())

        # Predict
        x = self.prediction(x)
        # print("3", torch.isnan(x).any(), torch.isinf(x).any())

        # Load predictions
        predicted = []
        total = 0
        for i in range(len(x_t)):
            p_s = l_t[i] + l_c[i]
            p_e = p_s + l_a[i]
            predicted.append(x[i, p_s:p_e])
            total += p_e - p_s

        # Loss
        if loss:
            losses = []
            for i in range(len(x_t)):
                target = audio[i][codec[i]]
                loss = F.cross_entropy(predicted[i], target, reduction = "sum")
                losses.append(loss)
            loss = torch.sum(torch.stack(losses)) / total
            return predicted, loss
        else:
            return predicted
    