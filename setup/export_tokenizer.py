from transformers import MarianTokenizer
import json

# Load the original tokenizer

model_names = ["Helsinki-NLP/opus-mt-ROMANCE-en", "Helsinki-NLP/opus-mt-en-ROMANCE"]


for model_name in model_names:
    tokenizer = MarianTokenizer.from_pretrained(model_name)

    # Get the vocab and special tokens
    vocab = tokenizer.get_vocab()
    special_tokens = tokenizer.special_tokens_map

    # Create the tokenizer config that transformers.js expects
    tokenizer_config = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [],
        "normalizer": None,
        "pre_tokenizer": None,
        "post_processor": None,
        "decoder": None,
        "model": {
            "type": "SentencePiece",
            "vocab": vocab,
            "unk_id": tokenizer.unk_token_id,
            "bos_id": tokenizer.bos_token_id,
            "eos_id": tokenizer.eos_token_id,
            "pad_id": tokenizer.pad_token_id
        }
    }

    # Save as tokenizer.json
    feature = "seq2seq-lm"
    with open(f"models/{model_name}-{feature}/tokenizer/tokenizer.json", "w") as f:
        json.dump(tokenizer_config, f, indent=2)