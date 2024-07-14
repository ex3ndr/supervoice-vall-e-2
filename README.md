# ✨ Supervoice VALL-E 2
An independent VALL-E 2 reproduction for voice synthesis with voice cloning.

## Features

* ⚡️ Narural sounding and voice cloning on human level
* 🎤 High quality - 24khz audio
* 🤹‍♂️ Versatile - synthesiszed voice has high variability
* 📕 Currently only English language is supported, but nothing stops us from adding more languages.

## Tips and tricks

* Network can follow voices, but they better to be in-domain and from librilight, libritts and from others similar sources

## Architecture

Repdorduction tries to follow papers as close as possible, but some minor changes include
* Linear annielation replaced with cosine one
* Not implemented codec grouping
* No padding masking used during training, since it would train 5 times slower using flash attention

![valle-2 arcitecture](/docs/arch.png)

## How to use

```python
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = torch.hub.load(repo_or_dir='ex3ndr/supervoice-vall-e-2', model='supervoice')
model = model.to(device)

# Synthesize
in_voice_1 = model.synthesize("voice_1", "What time is it, Steve?", top_p = 0.2).cpu()
in_voice_2 = model.synthesize("voice_2", "What time is it, Steve?", top_p = 0.2).cpu()

# Experimental voices
in_emo_1 = model.synthesize("emo_1", "What time is it, Steve?", top_p = 0.2).cpu()
in_emo_2 = model.synthesize("emo_2", "What time is it, Steve?", top_p = 0.2).cpu()

```

## License

MIT
