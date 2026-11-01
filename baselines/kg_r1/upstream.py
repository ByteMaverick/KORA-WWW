"""The authors' KG-R1 code and checkpoint, pinned (standard library only, so their environments can read it)."""
REPO = "https://github.com/junhongmit/KG-R1"
COMMIT = "a3c02facbb3f4a905c9cf24707867d1df0746595"      # their "Update README" commit
CHECKPOINT = "SCL2025/KG-R1-CQW"
CHECKPOINT_REVISION = "fa505b2e330843dbeebef7dd695141ebfc1af764"
#: The files the bf16 checkpoint needs (the index names only the -of-00002 shards; the -of-00003 set is not read).
CHECKPOINT_FILES = (
    "config.json", "generation_config.json", "model.safetensors.index.json",
    "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors",
    "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "added_tokens.json",
    "special_tokens_map.json",
)
#: The model name result rows carry (results/eval/<run>/<ENGINE>/...): KG-R1 answers with its own 3B, so its rows
#: never sit under a served model's name.
ENGINE = "kg-r1-cqw"
