import sentencepiece as spm
import gzip
import json

def clean_text(s: str) -> str:
    table = str.maketrans("’‘，。；？！（）：-《》、“”【】", "'',.;?!(): <>/\"\"[]")
    s = s.translate(table)
    return s.strip()

print("Prepare text...")
with open("train_tokenizer_text.txt", "w") as out:
    with gzip.open("./external_datasets/libriheavy/libriheavy_cuts_medium.jsonl.gz", "r") as f:
        for line in f:
            cut = json.loads(line)
            t = cut["supervisions"][0]["custom"]["texts"][0]
            t = clean_text(t)
            out.write(t + "\n")

print("Training text tokenizer")
spm.SentencePieceTrainer.train(
    input = "train_tokenizer_text.txt", 
    model_prefix = "tokenizer_text", 
    vocab_size = 4 * 1024, 
    character_coverage = 1.0, 
    num_threads = 32,
    # max_sentencepiece_length = 4,
    # This avoid binding spaces to tokens since we want to use them as a separate tokens
    add_dummy_prefix = False,
    allow_whitespace_only_pieces = True,
    user_defined_symbols = '▁',
    train_extremely_large_corpus = True
)