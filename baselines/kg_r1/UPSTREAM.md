# KG-R1

**Upstream.** <https://github.com/junhongmit/KG-R1>, commit `a3c02facbb3f4a905c9cf24707867d1df0746595` ("Efficient and
Transferable Agentic Knowledge Graph RAG via Reinforcement Learning", Lin, Liu et al., arXiv 2509.26383 v5). Licence:
Apache License 2.0 (the repository's `LICENSE` at that commit). The commit is the full hash. Nothing of theirs is
vendored: `fetch.sh code` clones the repository at the commit and `patch_upstream.py` edits two of its files (listed
below, each checked against the authors' sha256 in `patch_upstream.UPSTREAM_SHA256`).

**Checkpoint.** `SCL2025/KG-R1-CQW` on Hugging Face at revision `fa505b2e330843dbeebef7dd695141ebfc1af764`
("cwq-KG-r1-grpo-qwen2.5-3b-it_Aug11_f1_turn5-merged-step400": Qwen2.5-3B-Instruct trained with GRPO on CWQ; "CQW" is
the repository's spelling). It is the checkpoint the authors' `run/run.sh` evaluates with
`eval_scripts/kg_r1_eval_main/eval_qwen_3b_turn5_hf.sh`. `fetch.sh checkpoint` downloads the bf16 files only (the
two shards `model.safetensors.index.json` names, 6,794,207,232 bytes, and the tokenizer and config files), each
checked against the hub's sha256, and writes `REVISION`. Nothing is trained here. There is no plain WebQSP checkpoint.

## How it runs

KG-R1 is one agent. Each turn it thinks, then either sends one query to a graph server (`get_relations_out/in(entity)`,
`get_entities_out/in(entity, relation)`) or answers; after 5 query turns a final turn must answer. The server holds one
subgraph per question and finds entities by name. It calls none of our model servers: its own 3B answers, so its rows
sit under its own model name (`kg-r1-cqw`), the same on every backbone.

| file | what it does |
|---|---|
| `upstream.py` | the pins: repository, commit, checkpoint, revision, the checkpoint's files |
| `fetch.sh` | `code` (the clone at the commit), `checkpoint`, `freebase` (their CWQ / WebQSP subgraphs and the Freebase dump their entity names come from) |
| `build_env.sh` | their two environments as their README gives them: `env_kgr1` (torch 2.4.0, vLLM 0.6.3, their verl, flash-attn), `env_kg_server` (FastAPI, pandas) |
| `patch_upstream.py` | the edits below |
| `export.py` | our questions in their format: the server's per-question subgraphs and the evaluation parquet |
| `eval_env.py` | the settings below as shell exports and Hydra overrides |
| `preflight.py` | checks that their server answers the four operations on an export |
| `convert.py` | their per-example records as harness rows |
| `run_kgr1.sh` | per set: their server, the preflight, their turn-5 evaluation, the conversion |

**Settings** (the paper's value; the code's where the paper is silent):

- turns: 5 (Table 1, H = 5; the turn-5 script's `MAX_TURNS=5`);
- decoding: temperature 0, no sampling (App. E.4), top_p 1.0, no top_k, 256 tokens per turn (their
  `evaluation_k_beam.yaml`), one rollout per question. The turn-5 script itself samples at the rollout's temperature
  1.0 with four rollouts for pass@k; that is not run;
- server: their `kg_retrieval_server` as their launch script starts it (flat relations, entity names, the same
  actions); a 3,500-token prompt window and 512-token observations (their `evaluation_k_beam.yaml`); their
  "detailed_flat" guideline;
- prompt: theirs, unchanged (`export.py`, checked against their builder): `Question: <q>? (Initial entities: "a",
  ...)`, the initial entities being the split's gold topic entities, as for every system.

**Data** (`export.py`):

- CWQ and WebQSP: the authors' own subgraphs (`fetch.sh freebase`: their Google Drive archives, index tuples; names
  from the Freebase dump by their `process_entitites_freebase.sh` and `convert_entities.py`), with the split's
  questions, ids and gold topic entities (named from their entity file when the topic is in the question's subgraph;
  otherwise the split's name, counted in `export_meta.json`).
- MetaQA (and, in the code, BioCQ and KQA Pro, which the paper's tables do not use): the same per-question 2-hop
  subgraphs GCR gets, read from GCR's export file (`--gcr-export`, its sha256 recorded) or built with GCR's functions
  (`--kb`, `--endpoint`). The paper builds subgraphs from the anchors and the gold answers (App. C.2); ours, as GCR's,
  from the anchors alone, capped at 2,000 triples.
- GrailQA: not run. Their release has no GrailQA subgraphs or builder.
- The Unseen pairs (`webqsp-unseen`, `cwq-unseen` and their `-real` halves): the same per-question 2-hop subgraphs
  GCR's export cuts live from the extended graph for both halves (GrailQA's rule; `baselines/gcr/UPSTREAM.md`), read
  from GCR's export (`--gcr-export`) or built with GCR's function (`--endpoint`). Never the authors' subgraphs: a twin
  has none, and its real half is cut by the twin's rule. A question the graph was too slow for has no subgraph (a
  miss).
- No gold in the input: the parquet's `reward_model.ground_truth` and the server's `answers` are written empty; their
  reward and pass@k summary therefore read 0 and are not used. A question without a subgraph is listed in
  `missing.json`, never run, and becomes a miss row (`no_subgraph`).

**Rows** (`convert.py`): the answer is the authors' (`extract_answer_kg`): the first `<answer>` once the
`<information>` blocks are removed, scored as text; `kgr1_answer_set` keeps their own comma split, unscored.
`kgr1_kg_results` counts the queries the server answered with relations or entities; none means the answer was written
without the graph. A block saying the server could not be reached makes the row an error (`graph_outage`).
`kgr1_batch_rejected` counts turns their server refused whole (one request of the turn named an action its schema does
not take); that is their code's behaviour in their own runs too, counted, not corrected.

**Time.** Their evaluation is batched (128 questions, each advancing one turn per step), so a question's own time is
not measured: the patched evaluator times each batch, a row's `wall_time_s` is its batch's seconds (an upper bound on
its own), and a batch over 200 s makes its questions timeouts (lower `KGR1_VAL_BATCH_SIZE` if one does).
`kgr1_amortized_seconds` is the batch's seconds per question. The method's own bound is its turn limit.

## Edits to their clone (`patch_upstream.py`)

Paths and output only: nothing in the prompt, the agent loop, the server or the decoding. Each edited line carries
`# patched by baselines/kg_r1/patch_upstream.py v1`; with none of the `KGR1_*` variables set the script runs as
released.

`eval_scripts/kg_r1_eval_main/eval_qwen_3b_turn5_hf.sh`:

1. `K_VALUES` from `KGR1_K_VALUES` when set (theirs `"1 2 3 4"`, which also sets the rollouts per question);
   `run_kgr1.sh` sets 1, one rollout at the paper's deterministic decoding;
2. `CUDA_VISIBLE_DEVICES` from the environment when set (theirs is hard-coded to 4);
3. the evaluation parquet from `KGR1_TEST_FILE` when set (theirs knows only their CWQ and WebQSP files);
4. the output folder from `KGR1_OUT_DIR` when set, so each run writes into a folder of its own;
8. `VAL_BATCH_SIZE` from `KGR1_VAL_BATCH_SIZE` when set (theirs 128); throughput only.

`verl/trainer/ppo/ray_evaluator_kg.py`:

5. `import time`;
6. around each batch: the batch's question ids, in order, and its wall time;
7. every record of their detailed results file carries the question id it answers (`harness_sample_id`) and its
   batch's seconds (`harness_batch_seconds`); their records have neither, and are placed only by batch and position.
   The paper's own run records carry these two fields with our internal repository's name as the prefix in place of
   `harness_`: rename the prefix to `harness_` to convert such a file with this `convert.py`.

## Edits to our files for this release

Our files come from the paper's evaluation harness: `export.py` and `run_kgr1.sh` from the version the final runs used
(`85e4608`, abbreviated), which adds the Unseen export and `KGR1_PORT`; the others from `4eebf1b` (abbreviated).
Comments and docstrings shortened; the code is the paper harness's apart from these edits (each checked: the files are
AST-identical to the source with exactly these edits applied):

| file | change |
|---|---|
| `convert.py`, `eval_env.py` | imports from the paper harness rewired to this release's modules (table below) |
| `export.py`, `convert.py` | the questions read with `harness.dataset` |
| `convert.py` | `--results` default: a cluster folder became `results/eval` (relative to the folder it runs from, `$EVAL_ROOT` in `run_kgr1.sh`); the two record fields it reads are `harness_sample_id`, `harness_batch_seconds` (they were named after our internal repository); it also writes the folder's `run.json` (the fields the harness's runner writes), which `python -m harness score` needs |
| `patch_upstream.py` | the mark no longer names our internal repository; the inserted names are `harness_ids`, `harness_start`, `harness_batch_seconds`, `harness_sample_ids`, `harness_sample_id` (same reason) |
| `fetch.sh`, `build_env.sh` | the cluster paths became `$KGR1` / `$DATA_ROOT` and `$CONDA_SH` |
| `run_kgr1.sh` | the cluster job script without its scheduler lines: the same commands in the same order; the job id in the run folder and port became the shell's pid; the cluster's shared shell library became the variables below |

| imported in the paper's harness | here |
|---|---|
| `common.settings` | `harness.settings` (the `gcr`, `kg_r1` and `kbbinder_official` rows are in its `SYSTEMS` and `DECODING` tables, with the paper's values) |
| `common.main._score_text_answer` | `harness.metrics.reasoning.score_text` (the same function, imported as `_score_text_answer`) |
| `common.manifest.run_manifest` | `harness.manifest.run_manifest`: the functions it calls, moved unchanged, except that the row's code record is keyed `code` (it was named after the paper's internal repository), a copy without `.git` reads a `COMMIT` file at the repository's root, and `kora` is always None (KORA is this repository's `kora/`) |
| `data/kora/splits/<set>/<split>.json` | `harness.dataset.records(set, split)` and `path_of` (`data/questions/<set>/`, and the Unseen halves from `data/unseen/<set>/`); GrailQA's `test.json` is the 1,000 every system ran (the paper's `test-rep1000.json`, the same ids) |
| `baselines.gcr.export` | `baselines/gcr/export.py` |

**Record differences from the paper's rows** (values only; scoring is unchanged): the run manifest's code record is
keyed `code` (the paper's used our internal repository's name), its `kora` is None and its `upstream` names only
KB-BINDER's folder; the Unseen halves' questions come from one pair file per set (`data/unseen/<set>/test.json`), so
`-unseen` and `-unseen-real` record the same split path and sha256 (the paper's had one file per half); GCR's export
sidecar names GrailQA's `question_file` as `test.json` (the paper's `test-rep1000.json`, the same 1,000 questions).

## Variables

| variable | meaning |
|---|---|
| `DATA_ROOT` | the data folder; `KGR1` defaults to `$DATA_ROOT/kg_r1` (the clone `KG-R1/`, `hf/KG-R1-CQW/`, the environments, `export/`, `runs/`) |
| `EVAL_ROOT` | this repository's root: on `PYTHONPATH` for `eval_env.py` and `preflight.py`; `convert.py` runs there |
| `CONDA_SH` | the `conda.sh` the scripts source |
| `HARNESS_ENV` | the conda environment `convert.py` runs in |
| `CUDA_VISIBLE_DEVICES` | the GPU (one) the evaluation runs on; unset, their script's 4 |
| `KGR1_PORT` | the graph server's port, when the run shares a machine with other servers; else from the pid |
| `SETS`, `RUN`, `EXPORT`, `LIMIT`, `KGR1_VAL_BATCH_SIZE` | what `run_kgr1.sh` runs; `SETS` defaults to MetaQA, CWQ and WebQSP |

## Steps

1. Code and checkpoint, with internet: `bash baselines/kg_r1/fetch.sh code`, then `bash baselines/kg_r1/fetch.sh
   checkpoint` (about 6.8 GB; every line must say `ok`).
2. Environments: `bash baselines/kg_r1/build_env.sh` (both; or `kg_server` / `kgr1`). Their README pins only torch and
   vLLM; what pip resolved is written to `$KGR1/env_*.freeze.txt`.
3. Patch: `python -m baselines.kg_r1.patch_upstream $KGR1/KG-R1` (prints `patched`).
4. CWQ and WebQSP data, with `env_kg_server` active: `bash baselines/kg_r1/fetch.sh freebase` (their subgraph
   archives into `$KGR1/KG-R1/data_kg/{CWQ,webqsp}` and the 30 GB Freebase dump), then their name extraction (a large
   job: give it 32 CPUs and 256 GB, as the paper's run had): `cd $KGR1/KG-R1/scripts && ln -sf $KGR1/freebase-rdf-latest.gz . && bash
   process_entitites_freebase.sh && python convert_entities.py`.
5. Export, after GCR's export (`baselines/gcr/UPSTREAM.md`, step 2):
   ```
   python -m baselines.kg_r1.export --set metaqa --gcr-export $DATA_ROOT/gcr/data --out $KGR1/export
   python -m baselines.kg_r1.export --set cwq    --data-kg $KGR1/KG-R1/data_kg --out $KGR1/export
   python -m baselines.kg_r1.export --set webqsp --data-kg $KGR1/KG-R1/data_kg --out $KGR1/export
   for s in webqsp-unseen webqsp-unseen-real cwq-unseen cwq-unseen-real; do
     python -m baselines.kg_r1.export --set $s --gcr-export $DATA_ROOT/gcr/data --out $KGR1/export; done
   ```
6. Smoke, one GPU: `SETS=metaqa LIMIT=10 RUN=smoke bash baselines/kg_r1/run_kgr1.sh`. Check: `server.log` says
   "Initialized KG: <set> with N subgraphs"; the log says `kg_r1 preflight ok`; `eval/evaluation.log` shows
   `[PROMPT_AUGMENTATION] Enabled: True, Level: detailed_flat`, `n_rollout_eval: 1`, `'temperature': 0.0`, and every
   `[KG_ROUTING]` line sends to this run's port; `eval/test_detailed_results.jsonl` has 10 records, each with
   `harness_sample_id` and `harness_batch_seconds` under 200; the convert line has no `graph_outage`.
7. Runs, one GPU each: `SETS="metaqa"`, then `SETS="cwq webqsp"`, then
   `SETS="webqsp-unseen webqsp-unseen-real cwq-unseen cwq-unseen-real"`. Run folder (`run.json`, `s0.jsonl`):
   `$EVAL_ROOT/results/eval/<run>/kg-r1-cqw/<set>-test/kg_r1`; score it with `python -m harness score <that folder>`.

## Disclosed

- **A test question in the authors' prompt.** Their "detailed_flat" guideline
  (`verl/trainer/ppo/prompt_augmentation_kg.py` in their repository), which their dataset code adds to every prompt,
  lists "New York City" and "United Nations" on one line among its example entity names: the topic and the gold of
  WebQSP test question WebQTest-1501 ("where is the un based"). Kept as released; the question stays in the scored set.
- **In-domain on CWQ.** The checkpoint was trained on CWQ's train split; WebQSP runs on the same checkpoint (there is
  no plain WebQSP one), and on MetaQA it transfers without retraining, as the paper transfers it by swapping only the
  server's graph (§3.4, §4.4, Table 2).
- **Batched time.** A row's time is its batch's (above), an upper bound on the question's own. The paper reports
  6.4 s per question unbatched on an H100 (App. F.1).
- **No token counts.** Their records keep text; KG-R1's rows count no LLM calls or tokens.
