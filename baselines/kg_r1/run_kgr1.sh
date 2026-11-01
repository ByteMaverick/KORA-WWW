#!/bin/bash
# KG-R1 (the authors' code and checkpoint) on our sets, on one machine with one GPU: per set, their graph server on
# the exported subgraphs, the preflight of its four operations, their turn-5 evaluation of SCL2025/KG-R1-CQW
# (MAX_TURNS=5), then the per-example records as a harness run folder (results/eval/<run>/kg-r1-cqw/<set>-test/kg_r1).
#   CUDA_VISIBLE_DEVICES=<gpu> SETS="metaqa cwq webqsp" bash baselines/kg_r1/run_kgr1.sh
# Set first: EVAL_ROOT, DATA_ROOT, CONDA_SH, HARNESS_ENV (UPSTREAM.md, "Variables"); setup once: UPSTREAM.md, steps
# 1-5. KG-R1 calls none of our model servers: its own 3B answers. The paper's runs gave one job 1 GPU, 16 CPUs,
# 128 GB and at most a day. Every run writes into a new folder, $KGR1/runs/<RUN>_<date>_<pid>.
# LIMIT=<n> runs the first n questions of each set only (the smoke; RUN=smoke).

set -euo pipefail
: "${EVAL_ROOT:?this repository}" "${DATA_ROOT:?the data folder}" "${CONDA_SH:?conda.sh}" \
  "${HARNESS_ENV:?the harness conda environment}"
KGR1="${KGR1:-$DATA_ROOT/kg_r1}"
: "${SETS:=metaqa cwq webqsp}" "${RUN:=test}" "${EXPORT:=$KGR1/export}" "${LIMIT:=}"
CLONE="$KGR1/KG-R1"
MODEL_DIR="$KGR1/hf/KG-R1-CQW"
PIN_COMMIT=a3c02facbb3f4a905c9cf24707867d1df0746595
PIN_CHECKPOINT="SCL2025/KG-R1-CQW@fa505b2e330843dbeebef7dd695141ebfc1af764"
RUN_DIR="$KGR1/runs/${RUN}_$(date +%Y%m%dT%H%M%S)_$$"
[ ! -e "$RUN_DIR" ] || { echo "[kg_r1] $RUN_DIR exists; a run never writes into an earlier run's folder" >&2; exit 2; }
mkdir -p "$RUN_DIR"
echo "[kg_r1] run folder $RUN_DIR"

# The code and checkpoint the rows will name (convert.py records the pins): the clone at the pinned commit with this
# patch, the checkpoint at the pinned revision.
[ "$(git -C "$CLONE" rev-parse HEAD)" = "$PIN_COMMIT" ] || { echo "[kg_r1] $CLONE is not at $PIN_COMMIT" >&2; exit 2; }
for f in eval_scripts/kg_r1_eval_main/eval_qwen_3b_turn5_hf.sh verl/trainer/ppo/ray_evaluator_kg.py; do
  grep -q "patch_upstream.py v1" "$CLONE/$f" || { echo "[kg_r1] run baselines/kg_r1/patch_upstream.py (v1) on the clone" >&2; exit 2; }
done
[ "$(cat "$MODEL_DIR/REVISION" 2>/dev/null)" = "$PIN_CHECKPOINT" ] || { echo "[kg_r1] $MODEL_DIR is not $PIN_CHECKPOINT (fetch.sh checkpoint)" >&2; exit 2; }

# KGR1_PORT: a port of its own, when this runs beside other servers; the default is from the pid
PORT="${KGR1_PORT:-$((20000 + $$ % 10000))}"
URL="http://127.0.0.1:$PORT"
# Their client routes a query by dataset name (generation.py _init_server_routing) and falls back to the search_url
# eval_env.py passes: every route goes to this run's server.
for v in CWQ WEBQSP METAQA GRAILQA SIMPLEQA MULTITQ TREX QALD10EN ZERO_SHOT_RE; do export "${v}_SERVER_URL=$URL/retrieve"; done
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false
# Some schedulers set ROCR_VISIBLE_DEVICES (AMD's) beside CUDA_VISIBLE_DEVICES on NVIDIA nodes; their verl refuses both
# set (verl/single_controller/base/worker.py), and every Ray worker dies before loading the model
unset ROCR_VISIBLE_DEVICES
source "$CONDA_SH"
SERVER=""
# wait returns the killed server's 143: as the last of an && list under set -e it would end the run after its first set
stop_server() { if [ -n "$SERVER" ]; then kill "$SERVER" 2>/dev/null || true; wait "$SERVER" 2>/dev/null || true; fi; SERVER=""; return 0; }
trap stop_server EXIT

for d in $SETS; do
  out="$RUN_DIR/$d"
  mkdir -p "$out"
  parquet="$EXPORT/parquet/$d/test.parquet"
  [ -f "$parquet" ] || { echo "[kg_r1] no export for $d ($parquet): run export.py" >&2; exit 2; }
  kg="$d"; [ "$d" = cwq ] && kg=CWQ
  conda activate "$KGR1/env_kg_server"
  eval "$(PYTHONPATH="$EVAL_ROOT" python -m baselines.kg_r1.eval_env --env)"
  run_parquet="$parquet"
  if [ -n "$LIMIT" ]; then             # the smoke: the first LIMIT questions
    run_parquet="$out/limit.parquet"
    python -c "import pyarrow.parquet as pq,sys; t=pq.read_table(sys.argv[1]); pq.write_table(t.slice(0,int(sys.argv[3])),sys.argv[2])" \
      "$parquet" "$run_parquet" "$LIMIT"
  fi
  n="$(python -c "import pyarrow.parquet as pq,sys; print(pq.read_metadata(sys.argv[1]).num_rows)" "$run_parquet")"

  echo "[kg_r1] $d: graph server on $URL ($(date -Is))"
  (cd "$CLONE" && exec python -m kg_r1.search.kg_retrieval_server --base_data_path "$EXPORT/kg" --host 127.0.0.1 \
    --port "$PORT" --workers 2 --thread_workers 16 --kgs "$kg" \
    --actions get_head_relations get_tail_relations get_head_entities get_tail_entities get_conditional_relations \
    --relation_format "$KGR1_RELATION_FORMAT" --use_entities_text) > "$out/server.log" 2>&1 &
  SERVER=$!
  for _ in $(seq 1 360); do curl -fsS --max-time 5 "$URL/health" >/dev/null 2>&1 && break; sleep 5; done
  PYTHONPATH="$EVAL_ROOT" python -m baselines.kg_r1.preflight --url "$URL" --export "$EXPORT" --set "$d" \
    || { echo "[kg_r1] $d: the graph server failed its preflight ($out/server.log)" >&2; exit 2; }

  conda activate "$KGR1/env_kgr1"
  mapfile -t OVERRIDES < <(PYTHONPATH="$EVAL_ROOT" python -m baselines.kg_r1.eval_env --overrides "$PORT")
  # their unused train loader drops a last partial batch of 64; their config check then wants the (unused, training)
  # ppo_mini_batch_size of 64 no larger than it (ray_trainer_kg._validate_config) and a multiple of the per-GPU micro
  # batch of 8 (fsdp_workers), so a smaller set lowers both, the mini batch to a multiple of 8 (all unused in evaluation)
  if [ "$n" -lt 64 ]; then
    mini=$(( n >= 8 ? n / 8 * 8 : n ))
    OVERRIDES+=("data.train_batch_size=$n" "actor_rollout_ref.actor.ppo_mini_batch_size=$mini")
    [ "$n" -lt 8 ] && OVERRIDES+=("actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$n")
  fi
  echo "[kg_r1] $d: $n questions, turn-5 evaluation of $MODEL_DIR ($(date -Is)); overrides: ${OVERRIDES[*]}"
  KGR1_TEST_FILE="$run_parquet" KGR1_OUT_DIR="$out/eval" \
    bash "$CLONE/eval_scripts/kg_r1_eval_main/eval_qwen_3b_turn5_hf.sh" "$MODEL_DIR" "$d" "${OVERRIDES[@]}"
  stop_server
  # Their script pipes the evaluation through tee, so its exit status is lost: the records are checked instead.
  detailed="$out/eval/test_detailed_results.jsonl"
  got=0
  [ -f "$detailed" ] && got="$(grep -c . "$detailed" || true)"
  [ "$got" -eq "$n" ] || { echo "[kg_r1] $d: $got records for $n questions in $detailed ($out/eval/evaluation.log)" >&2; exit 2; }

  conda activate "$HARNESS_ENV"
  (cd "$EVAL_ROOT" && python -m baselines.kg_r1.convert --detailed "$detailed" --export "$EXPORT/parquet/$d" --set "$d" --run "$RUN")
done
echo "[kg_r1] done $RUN_DIR $(date -Is)"
