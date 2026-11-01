#!/bin/bash
# GCR on our sets, on one machine with one GPU: step 1 (the released 8B writes paths constrained to each question's
# subgraph), step 2 (each answer model reads the paths), then the predictions as a harness run folder (UPSTREAM.md).
#   SETS="cwq webqsp" ANSWER_MODELS="qwen3-30b-a3b gpt-oss-120b" bash baselines/gcr/run_gcr.sh
# Set first: EVAL_ROOT, DATA_ROOT, CONDA_SH, HARNESS_ENV, and SERVER_<model> for each answer model (UPSTREAM.md,
# "Variables"). The paper's runs gave one job 1 GPU, 16 CPUs, 128 GB and at most a day.
# Every run writes into a new folder, $GCR/runs/<RUN>_<date>_<pid>: the authors' scripts resume from any
# predictions.jsonl already in their output folder, and those rows would be scored as this run's.

set -euo pipefail
: "${EVAL_ROOT:?this repository}" "${DATA_ROOT:?the data folder}" "${CONDA_SH:?conda.sh}" \
  "${HARNESS_ENV:?the harness conda environment}"
# healthy_list MODEL: the model's server, SERVER_<model> (e.g. SERVER_qwen3_30b_a3b=http://<host>:<port>/v1), if it answers
healthy_list() {
  local v addr
  v="SERVER_$(printf '%s' "$1" | tr -c 'A-Za-z0-9' '_')"
  addr="${!v:-}"
  if [ -n "$addr" ] && curl -fsS --max-time 10 "$addr/models" >/dev/null 2>&1; then echo "$addr"; fi
  return 0
}
GCR="${GCR:-$DATA_ROOT/gcr}"
DATA="${DATA:-$GCR/data}"   # the exports; a smoke reads its own (export.py --limit into $GCR/data-smoke)
: "${SETS:=cwq webqsp grailqa metaqa}" "${ANSWER_MODELS:=${ANSWER_MODEL:-qwen3-30b-a3b}}" "${RUN:=test}"
MODEL_DIR="$GCR/hf/GCR-Meta-Llama-3.1-8B-Instruct"
RUN_DIR="$GCR/runs/${RUN}_$(date +%Y%m%dT%H%M%S)_$$"
[ ! -e "$RUN_DIR" ] || { echo "[gcr] $RUN_DIR exists; a run never writes into an earlier run's folder" >&2; exit 2; }
mkdir -p "$RUN_DIR"
echo "[gcr] run folder $RUN_DIR"

# only DIR: the one predictions.jsonl the step wrote under DIR (a folder of its own, new in this run), or stop.
only() {
  local found n
  found="$(find "$1" -name predictions.jsonl -type f 2>/dev/null || true)"
  n="$(printf '%s' "$found" | grep -c . || true)"
  [ "$n" -eq 1 ] || { echo "[gcr] expected one predictions.jsonl under $1, found $n" >&2; return 2; }
  printf '%s\n' "$found"
}

# undirected SET: GCR's --undirected for the set. Freebase (CWQ, WebQSP, GrailQA, and both halves of WebQSP- and
# CWQ-Unseen, cut from the extended Freebase by GrailQA's rule): False, the authors' default and how
# they ran; Freebase stores most relations both ways. MetaQA and the BioKG layout (BioCQ, KQA Pro, their copies): True,
# GCR's own option, since a directed walk from many of their topic entities finds no path.
undirected() {
  case "$1" in
    cwq|webqsp|grailqa|webqsp-unseen|webqsp-unseen-real|cwq-unseen|cwq-unseen-real) echo False ;;
    *) echo True ;;
  esac
}

for d in $SETS; do   # every set is exported (its questions and the sidecar convert.py reads) before the GPU is spent
  [ -s "$DATA/$d/test.jsonl" ] && [ -s "$DATA/$d.test.harness.json" ] \
    || { echo "[gcr] $d: no export at $DATA/$d/test.jsonl and $DATA/$d.test.harness.json (export.py)" >&2; exit 2; }
done
for m in $ANSWER_MODELS; do   # every answer model's server is up before step 1 spends the GPU
  [ -n "$(healthy_list "$m" | sed -n 1p)" ] || { echo "[gcr] no live $m server" >&2; exit 2; }
done
source "$CONDA_SH"; conda activate "$GCR/env"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OPENAI_API_KEY=EMPTY
cd "$GCR/graph-constrained-reasoning"; export PYTHONPATH="$PWD:$EVAL_ROOT"   # its scripts import src.*; ours baselines.*
grep -q "patch_upstream.py v3" src/llms/chatgpt.py || { echo "[gcr] restore the clone (git checkout -- src/llms) and run baselines/gcr/patch_upstream.py (v3) on it" >&2; exit 2; }
mkdir -p tmp   # their TIKTOKEN_CACHE_DIR ('./tmp')

for d in $SETS; do
  # Both steps under the harness's 200 s per question (timed.py): step 1's times are shared by every answer model,
  # step 2's are per model; convert.py adds them and misses the over-limit and any prediction this run did not time.
  out="$RUN_DIR/$d"
  times1="$out/times/step1.jsonl"
  echo "[gcr] $d: step 1, path generation ($(date -Is))"
  python -m baselines.gcr.timed --step 1 --times "$times1" workflow/predict_paths_and_answers.py --data_path "$DATA" --d "$d" --split test \
    --index_path_length 2 --model_name GCR-Meta-Llama-3.1-8B-Instruct --model_path "$MODEL_DIR" \
    --k 10 --prompt_mode zero-shot --generation_mode group-beam --attn_implementation sdpa \
    --undirected "$(undirected "$d")" --predict_path "$out/GenPaths"
  paths="$(only "$out/GenPaths")"
  for m in $ANSWER_MODELS; do
    # Step 2's decoding from the settings table (max tokens, and gpt-oss's reasoning effort), read by the patched client.
    eval "$(PYTHONPATH="$EVAL_ROOT" python -m baselines.gcr.step2_env "$m")"
    # sed, not head: head closes the pipe on a second server and pipefail ends the run (exit 141)
    answer_url="$(healthy_list "$m" | sed -n 1p)"
    [ -n "$answer_url" ] || { echo "[gcr] no live $m server" >&2; exit 2; }
    export OPENAI_BASE_URL="$answer_url"
    times2="$out/times/$m.step2.jsonl"
    echo "[gcr] $d: step 2, answer with $m at $answer_url, max_tokens=$GCR_MAX_TOKENS top_p=$GCR_TOP_P top_k=$GCR_TOP_K reasoning_effort=${GCR_REASONING_EFFORT:-none} ($(date -Is))"
    # -n 8: their thread pool (multiprocessing.dummy); timed.py bounds each question in a worker thread too.
    python -m baselines.gcr.timed --step 2 --times "$times2" --after "$times1" workflow/predict_final_answer.py --data_path "$DATA" --d "$d" --split test \
      --model_name "$m" --reasoning_path "$paths" --add_path True -n 8 \
      --predict_path "$out/KGQA/$m"
    answers="$(only "$out/KGQA/$m")"
    conda activate "$HARNESS_ENV"
    (cd "$EVAL_ROOT" && python -m baselines.gcr.convert --predictions "$answers" --paths "$paths" --times "$times1" "$times2" \
      --exported "$DATA/$d.test.harness.json" --set "$d" --model "$m" --run "$RUN")
    conda activate "$GCR/env"
  done
done
echo "[gcr] done $RUN_DIR $(date -Is)"
