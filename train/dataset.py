import gzip
import json
import math
import random
import torch

def load_sampler(index, dir, batch_size, tokenizer = None):

    # Load ids
    ids = []
    with gzip.open(index, "r") as f:
        for line in f:
            cut = json.loads(line)
            id = cut["supervisions"][0]["id"]
            if id.startswith("small/"):
                id = id[len("small/"):]
            if id.startswith("medium/"):
                id = id[len("medium/"):]
            if id.startswith("large/"):
                id = id[len("large/"):]
            ids.append(id)

    def sample():
        loaded = 0
        res_encoded = []
        res_text = []
        while loaded < batch_size:
            # Pick ID
            id = random.choice(ids)

            try:

                # Load text
                with open(dir + id + ".txt", 'r') as file:
                    text = file.read()
                    if tokenizer is not None:
                        text = tokenizer.encode(text) if random.random() < 0.3 else tokenizer.encode_sample(text) # 30% chance of sampling optimal
                        if text.shape[0] == 0:
                            raise Exception("Empty file")

                # Load encoded
                encoded = torch.load(dir + id + ".pt")

                # Append
                res_text.append(text)
                res_encoded.append(encoded)
                loaded += 1
            except:
                print("Invalid file: " + id)
                pass
        
        return res_encoded, res_text
    
    return sample

def create_async_loader(sampler, num_workers = 1):

    # Dataset
    class AsyncDataset(torch.utils.data.IterableDataset):
        def __init__(self, sampler):
            self.sampler = sampler
        def generate(self):
            while True:
                yield self.sampler()
        def __iter__(self):
            return iter(self.generate())
    dataset = AsyncDataset(sampler)

    # Load loader
    loader = torch.utils.data.DataLoader(dataset, batch_size = 1, num_workers = num_workers, pin_memory = True, shuffle=False)

    return loader