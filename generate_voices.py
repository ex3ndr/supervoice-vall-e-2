from supervoice_valle import Supervoice, Tokenizer
from encodec import EncodecModel
from pathlib import Path
import torch

# Load tokenizer
tokenizer = Tokenizer("./tokenizer_text.model")

# Load encodec
encodec_model = EncodecModel.encodec_model_24khz()
encodec_model.set_target_bandwidth(6.0)

# We don't need a real model for this task
model = Supervoice(None, None, encodec_model, None, tokenizer)

# Find all wav files in the voices directory
wav_files = list(Path('voices').glob('*.wav'))
wav_files = [f.stem for f in wav_files]

# Generate voices
for id in wav_files:
    print(f"Processing {id}")
    with open("./voices/" + id + ".txt", 'r') as f:
        text = f.read().strip()
    created_voice = model.create_voice(audio = "./voices/" + id + ".wav", text = text)
    torch.save(created_voice, f"./voices/{id}.pt")

# Generate index file
with open("supervoice_valle/voices_gen.py", "w") as f:
    f.write(f"available_voices = {wav_files}")
        