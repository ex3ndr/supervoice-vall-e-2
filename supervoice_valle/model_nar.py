import torch
from torch.nn import functional as F
from .transformer import Transformer
from .tensors import sinusoids

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
            ffn_dropout = 0.1,
            position_embedding = None,
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

        # Tie weights
        self.audio_embedding.weight = self.prediction.weight
    
    def forward(self, *, condition_text, condition_audio, audio, codec, loss = False):

        # print(condition_text.shape, condition_audio.shape, audio.shape, codec)

        # Check shapes
        assert codec >= 1 and codec <= 7
        # [N]
        assert len(condition_text.shape) == 1
         # [8, N]
        assert len(condition_audio.shape) == 2
        assert condition_audio.shape[0] == 8
         # [<codec>, N]
        assert len(audio.shape) == 2
        assert audio.shape[0] >= codec

        # Prepare EOS
        eos = self.eos_embedding(torch.tensor([0], device = condition_text.device))
        
        # Prepare text condition
        x_t = torch.cat([self.text_embedding(condition_text), eos])
        x_pt = self.positional_embedding[:x_t.shape[0]]
        x_t = x_t + x_pt

        # Prepare audio condition
        x_ea = self.audio_embedding(condition_audio[0])
        for i in range(1, condition_audio.shape[0]):
            x_ea = x_ea + self.audio_embedding(condition_audio[i])
        
        # Prepare audio coarse
        x_ec = self.audio_embedding(audio[0])
        for i in range(1, codec - 1):
            x_ec = x_ec + self.audio_embedding(audio[i])

        # Prepare audio
        x_a = torch.cat([x_ea, x_ec, eos])
        x_ap = self.positional_embedding[:x_a.shape[0]]
        x_a = x_a + x_ap

        # Prepare codec index
        x_ci = self.codec_index_embedding(torch.tensor([codec - 1], device = x_t.device).long())

        # Concatenate all
        x = torch.cat([x_t, x_a, x_ci], dim = 0)

        # Transform
        x = self.transformer(x.unsqueeze(0)).squeeze(0)

        # Predict
        x = self.prediction(x)

        # Offsets
        p_s = x_t.shape[0] + condition_audio.shape[1]
        p_e = p_s + audio.shape[1]
        predicted = x[p_s:p_e]

        # Loss
        if loss:
            target = audio[codec]
            loss = F.cross_entropy(predicted, target)
            return predicted, loss
        else:
            return predicted
    