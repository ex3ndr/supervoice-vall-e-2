# ✨ Supervoice VALL-E 2
An independent VALL-E 2 reproduction for voice synthesis with voice cloning.

## Features

* ⚡️ Narural sounding and voice cloning on human level
* 🎤 High quality - 24khz audio
* 🤹‍♂️ Versatile - synthesiszed voice has high variability
* 📕 Currently only English language is supported, but nothing stops us from adding more languages.

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

# TODO: Implement

```

## License

MIT
