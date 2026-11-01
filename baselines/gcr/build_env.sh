#!/bin/bash
# GCR's own environment (it pins transformers 4.44, whose group beam search it uses), installed into the conda
# environment at $GCR/env, which must exist (GCR defaults to $DATA_ROOT/gcr).
source "${CONDA_SH:?conda.sh}"
GCR="${GCR:-${DATA_ROOT:?the data folder}/gcr}"
conda activate "$GCR/env" || exit 1
pip install --no-cache-dir -q torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121 || exit 1
pip install --no-cache-dir -q transformers==4.44.2 accelerate==0.34.2 peft==0.12.0 datasets==2.21.0 marisa-trie==1.2.0 \
  openai==1.51.0 tiktoken python-dotenv scikit-learn sentencepiece protobuf==5.28.2 networkx tqdm httpx==0.27.2 || exit 1
python -c "import torch, transformers, marisa_trie; print(\"env ok\", torch.__version__, transformers.__version__)"
