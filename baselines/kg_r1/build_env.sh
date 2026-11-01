#!/bin/bash
# KG-R1's two environments, as the authors' README gives them (github.com/junhongmit/KG-R1 @ a3c02fa, "Installation"):
#   $KGR1/env_kgr1       the agent: Python 3.10, torch 2.4.0 (cu121), vLLM 0.6.3, their verl (`pip install -e .` in
#                        the clone), flash-attn, wandb, and their KG-processing extras
#   $KGR1/env_kg_server  the graph server: FastAPI, uvicorn, networkx, pandas, pyarrow (and ujson for their Freebase
#                        name extraction, which installs it itself when missing)
# Their README pins only torch and vLLM; what pip resolved is written to $KGR1/env_*.freeze.txt for the record.
#   KGR1=<folder> bash baselines/kg_r1/build_env.sh [kgr1|kg_server]   (KGR1 defaults to $DATA_ROOT/kg_r1)
set -u
KGR1="${KGR1:-${DATA_ROOT:?the data folder}/kg_r1}"
source "${CONDA_SH:?conda.sh}" || exit 1
CLONE="$KGR1/KG-R1"

kgr1() {
  [ -d "$CLONE/.git" ] || { echo "run baselines/kg_r1/fetch.sh code first" >&2; return 1; }
  [ -d "$KGR1/env_kgr1" ] || conda create -y -q -p "$KGR1/env_kgr1" python=3.10 || return 1
  conda activate "$KGR1/env_kgr1" || return 1
  pip install --no-cache-dir -q torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121 || return 1
  pip3 install --no-cache-dir -q vllm==0.6.3 || return 1
  (cd "$CLONE" && pip install --no-cache-dir -q -e .) || return 1
  conda install -y -q -c nvidia cuda-toolkit=12.1 || return 1
  pip3 install --no-cache-dir -q flash-attn --no-build-isolation || return 1
  pip install --no-cache-dir -q wandb fastapi uvicorn requests aiohttp networkx || return 1
  python -c "import torch, vllm, verl, flash_attn; print('env_kgr1 ok', torch.__version__, vllm.__version__)" || return 1
  pip freeze > "$KGR1/env_kgr1.freeze.txt"
  conda deactivate
}

kg_server() {
  [ -d "$KGR1/env_kg_server" ] || conda create -y -q -p "$KGR1/env_kg_server" python=3.10 || return 1
  conda activate "$KGR1/env_kg_server" || return 1
  pip install --no-cache-dir -q fastapi uvicorn pydantic requests transformers datasets huggingface_hub \
    networkx pandas pyarrow numpy scipy ujson || return 1
  python -c "import fastapi, uvicorn, pandas, pyarrow; print('env_kg_server ok')" || return 1
  pip freeze > "$KGR1/env_kg_server.freeze.txt"
  conda deactivate
}

case "${1:-all}" in
  kgr1) kgr1 ;;
  kg_server) kg_server ;;
  all) kg_server && kgr1 ;;
  *) echo "usage: $0 [kgr1|kg_server]" >&2; exit 2 ;;
esac
