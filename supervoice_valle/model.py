import torch
import torch.nn as nn
import torchaudio
from encodec import EncodecModel
from encodec.utils import convert_audio
from .voices_gen import available_voices
import os

class Supervoice(nn.Module):
    def __init__(self, model_ar, model_nar, model_encodec, vocoder, tokenizer):
        super(Supervoice, self).__init__()
        self.model_ar = model_ar
        self.model_nar = model_nar
        self.model_encodec = model_encodec
        self.tokenizer = tokenizer
        self.vocoder = vocoder

    @torch.inference_mode()
    def create_voice(self, audio, text):
        device = self._device()

        # Load audio
        if type(audio) is str:
            audio, sr = torchaudio.load(audio)
            if sr != 16000:
                audio = torchaudio.transforms.Resample(sr, 16000, dtype=audio.dtype)(audio)
            audio = audio.squeeze(0)
        else:
            assert audio.dim() == 2 or audio.dim() == 1, "Audio must be 1D or 2D tensor"
            if audio.dim() == 2:
                assert audio.size(0) == 1, "Audio must have a single channel"
                audio = audio.squeeze(0)

        # Preprocess audio
        audio = convert_audio(audio.unsqueeze(0), 16000, self.model_encodec.sample_rate, self.model_encodec.channels)

        # Encode audio
        wav = audio.unsqueeze(0)
        encoded_frames = self.model_encodec.encode(wav.to(device))
        audio_tokens = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1).squeeze().cpu()
        
        # Prepare text
        text = self._normalize_text(text)
        
        # Return
        return {
            "audio_tokens": audio_tokens,
            "text": text,
        }

    @torch.inference_mode()
    def synthesize(self, voice, text, top_k = None, top_p = 0.2):
        device = self._device()

        # Prepare voice
        if type(voice) is str:

            # Check if voice is available
            if voice not in available_voices:
                raise ValueError(f"Voice {voice} is not available")

            # Get the current file directory
            current_dir = os.path.dirname(os.path.abspath(__file__))

            # Load voice
            voice_file = os.path.join(current_dir, "..", "voices", voice + ".pt")
            voice = torch.load(voice_file, map_location = "cpu")

        # Tokenize text
        text_tokens = self.tokenizer.encode(self._normalize_text(voice["text"]) + " " + self._normalize_text(text)).to(device)

        # Audio tokens
        audio_tokens = voice["audio_tokens"].to(device)        

        # AR inference
        coarse_tokens = self.inference_ar(text_tokens, audio_tokens[0], top_k = top_k, top_p = top_p)

        # NAR inference
        tokens = self.inference_nar(text_tokens, audio_tokens, coarse_tokens)

        # Vocoder
        features = self.vocoder.codes_to_features(tokens.to(device))
        bandwidth_id = torch.tensor([2]).to(device)  # 6 kbps
        return self.vocoder.decode(features, bandwidth_id=bandwidth_id)  

    @torch.inference_mode()
    def inference_ar(self, text_tokens, audio_tokens, top_k = None, top_p = None):
        device = self._device()

        # Run inference
        text_tokens = text_tokens.to(device)
        output = audio_tokens.to(device)
        prev = None
        while True:

            # Inference
            p = self.model_ar(
                text = [text_tokens],
                audio = [output]
            )
            p = p[0][-1]

            # Sample code
            code, prev = self._sample_ar(p, top_k = top_k, top_p = top_p, prev = prev)

            # Append code
            if (code > 1023) or output.shape[0] > 2000:
                break
            output = torch.cat([output, torch.tensor([code], device = output.device)])
            
        # Cut the audio tokens
        output = output[audio_tokens.shape[0]:]
        
        return output

    @torch.inference_mode()
    def inference_nar(self, text_tokens, audio_tokens, coarse_tokens):
        device = self._device()

        # Run inference
        condition_text = text_tokens.to(device)
        condition_audio = audio_tokens.to(device)
        predicted = [coarse_tokens.to(device)]
        for i in range(1, 8):

            # Inference
            p = self.model_nar(
                condition_text = [condition_text], 
                condition_audio = [condition_audio],
                audio = [torch.stack(predicted)],
                codec = [i]
            )

            # Argmax sampling
            p = p[0]
            p = torch.nn.functional.softmax(p, dim=-1)
            p = torch.argmax(p, dim=-1, keepdim=True)
            p = p.squeeze(-1)

            # Append
            predicted.append(p)

        # Result
        return torch.stack(predicted)

    def _device(self):
        return next(self.parameters()).device

    def _normalize_text(self, text):
        # This method follows the same normalization of the libriheavy dataset
        table = str.maketrans("’‘，。；？！（）：-《》、“”【】", "'',.;?!(): <>/\"\"[]")
        text = text.translate(table)
        return text.strip()
        
    def _sample_ar(self, logits, top_k = None, top_p = None, prev = None):

        # Top-k
        if top_k is not None:

            # Find all indices which value is less than k-th one
            indices_to_remove = logits < torch.topk(logits, top_k, dim=-1)[0][..., -1, None]

            # Assign minus infinity for such values
            logits[indices_to_remove] = float('-inf')

        # Top-p
        if top_p is not None:

            # Sort logits
            sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)

            # Calculate cummulative probabilities
            cum_sum_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove all indices with cummulative probability more than top_p
            sorted_indices_to_remove = cum_sum_probs < top_p

            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # Assign minus infinity for such values
            sorted_logits[sorted_indices_to_remove] = float('-inf')
        
            # Then reverse the sorting process by mapping back sorted_logits to their original position
            logits = torch.gather(sorted_logits, 0, sorted_indices.argsort(-1))

        # Softmax
        probs = torch.nn.functional.softmax(logits, dim=-1)

        # Sample
        return torch.multinomial(probs, num_samples=1).item(), None