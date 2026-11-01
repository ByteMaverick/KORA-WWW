# Graph-constrained Reasoning (GCR)

**Upstream.** <https://github.com/RManLuo/graph-constrained-reasoning>, commit `9518e8e` (abbreviated; the full hash is not recorded) ("Graph-constrained Reasoning:
Faithful Reasoning on Knowledge Graphs with Large Language Models", Luo et al., ICML 2025). Licence: MIT (the
repository's `LICENSE`, copied in `upstream/`).

**Checkpoint.** `rmanluo/GCR-Meta-Llama-3.1-8B-Instruct` on Hugging Face, the authors' released weights; nothing is
trained here. `fetch.sh` downloads every file of the repository's `main` branch; no revision is pinned.

**Data.** RoG's released subgraphs for CWQ and WebQSP, the data the authors ran on (paper App. C): the Hugging Face
datasets `rmanluo/RoG-cwq` and `rmanluo/RoG-webqsp` (`fetch.sh`, `main` branch, not pinned).

**Vendored unchanged** in `upstream/`: the authors' LLM client files at `9518e8e`, which `patch_upstream.py` is
written against and `step2_client.py` patches in a temporary folder when no clone is given.

| file | sha256 |
|---|---|
| LICENSE | `d981694a61972fe3a9e082f4638c283d06a152ecf018ffe4ae2eaaeb6d524c74` |
| src/llms/__init__.py | `2256d23744b81b1f29994dc35516f6b2c692c7aea5c05c437469e6fba270a613` |
| src/llms/base_language_model.py | `fdcbdc8e074507a6c48bca7516dc92b13b34d0748d15b39fb08cc8e922870491` |
| src/llms/chatgpt.py | `d16f56b2738eb92241c1de3172a05d2b514c44d487211874dfcad578bb63bfff` |

The three `.py` checksums are also `patch_upstream.UPSTREAM_SHA256`. The rest of their repository is cloned at the
commit (step 1 below), not vendored.

**Our files** come from the paper's evaluation harness: `export.py` and `run_gcr.sh` from the version the final runs
used (`85e4608`, abbreviated), which adds the Unseen export and the slow-graph handling to the one the other files come
from (`4eebf1b`, abbreviated).

## How it runs

GCR does not search a graph. Each question carries a subgraph; GCR indexes every path of up to 2 hops from the
question's entities, the fine-tuned 8B writes paths that must exist in that index (step 1, one GPU), and a general LLM
reads the paths and the 8B's hypothesis answers and answers (step 2, our served model, so every system is compared on
the same model).

| file | what it does |
|---|---|
| `fetch.sh` | the 8B and RoG's CWQ / WebQSP data into `$GCR/hf/`, one file at a time with curl |
| `build_env.sh` | GCR's own environment in `$GCR/env` (transformers 4.44, whose group beam search it uses) |
| `patch_upstream.py`, `step2_env.py` | let step 2 call our model servers with the settings below (edits listed below) |
| `export.py` | our questions in GCR's format, each with its subgraph, the gold fields empty |
| `timed.py` | runs their two workflow scripts with each question timed and held to the harness's 200 s |
| `convert.py` | their predictions as harness rows |
| `run_gcr.sh` | steps 1 and 2 and the conversion, per set |
| `step2_client.py` | one step-2 request through the patched client, for a preflight of an answer server |

**Settings**, all GCR's own (paper §5.1 and App. E: K = 10, L = 2): k = 10 paths, group beam search, index path length
2, zero-shot prompt. Step 2: the authors' temperature 0.0; max_tokens 1,024 (the harness's cap; their client sends
none), on gpt-oss 4,096 with reasoning effort `low`; top_p 1.0 and no top_k (the OpenAI API's defaults their client
relied on), sent explicitly.

**The walk's direction** (`undirected` in `run_gcr.sh`): on Freebase (CWQ, WebQSP, GrailQA) `--undirected False`,
the authors' default and how they ran (Freebase stores most relations both ways, e.g. `people.person.parents` /
`people.person.children`); on MetaQA and the BioKG-layout graphs `--undirected True`, GCR's own option, since MetaQA's
edges all point from a film to its attributes, so a directed walk from many topic entities finds nothing.
Undirected, an edge is walked against its direction under its own relation name.

**Subgraphs** (`export.py`):

- CWQ and WebQSP (GCR's home sets: its 8B was fine-tuned on their train splits) read RoG's released subgraphs whole,
  but start from the harness's gold topic entities, as every system does: a topic entity whose harness name is a node
  of RoG's subgraph is that node; one named by its MID is RoG's name for it when RoG's `q_entity` lines up; one with
  no node is not a start node and is listed on the row (`gcr_topic_absent`). RoG's own `q_entity` is never the start
  list. The 11 WebQSP test questions RoG has no subgraph for are not exported; `convert.py --exported` writes each as
  a miss (`no_subgraph`).
- GrailQA (Freebase, not in GCR's paper; RoG released no subgraphs for it) is exported live from the Freebase SPARQL
  endpoint, as the paper builds subgraphs where RoG has none (§5.4, App. C): every triple leading out of the topic
  entity and out of each neighbour, by English name (a CVT by its MID, a literal by its value), names, keys,
  descriptions, aliases and web pages left out, at most 2,000 triples, every query sorted before its LIMIT.
  The paper's GrailQA rows for GCR were produced with the export from before the slow-question rules: a question's
  queries tried once more after 30 s, then a `no_subgraph` miss; a partial Virtuoso answer (its `X-SQL-State` header)
  counted as a timeout; five such misses in a row stop the export. With this export a rerun can mark slow or partially
  answered GrailQA questions as misses where the earlier export kept a truncated subgraph or stopped. CWQ, WebQSP and
  MetaQA are unaffected (their exports do not query a live endpoint).
- The Unseen pairs (`webqsp-unseen`, `cwq-unseen` and their `-real` halves) by GrailQA's live rule, both halves
  from the extended graph's endpoint, walked directed. A question whose queries the graph is too slow for (a hub; the
  query limit is `GCR_EXPORT_TIMEOUT`, 300 s) is tried twice, 30 s apart, then left without a subgraph (a miss row,
  `no_subgraph`), never written truncated; an outage stops the export, as do five slow questions in a row, and
  `--resume` continues it. GCR's rows in the paper's Unseen tables are not reported; KG-R1 reads these subgraphs.
- MetaQA from its `kb.txt` (the code also exports BioCQ and KQA Pro from their SPARQL endpoints, which the paper's
  tables do not use): every triple within 2 hops of
  the topic entities, in either direction, one-hop first, capped at 2,000 triples (the harness's cap; the paper states
  none). The export prints how often the cap bit and how often a gold answer is inside the subgraph at all.
- The gold is not in GCR's input: RoG's `a_entity` and `answer` fields are exported empty. GCR's inference never reads
  them (only the authors' own scoring and their gold-path ablation, not run, do), so `timed.py` skips that scoring and
  the harness scores the rows `convert.py` writes against the split's gold.

**Time.** Both steps run under `timed.py`, which writes each question's seconds per step; `convert.py` adds the two,
and a question over 200 s in either step or both together is a miss (`timeout`). A prediction this run did not time in
both steps is a miss (`untimed`), charged 200 s: the authors' scripts resume from any `predictions.jsonl` already in
their output folder, so `run_gcr.sh` writes every run into a new folder and stops if it exists.

**Rows** (`convert.py`): one per exported or missing question, with `gcr_paths` (paths step 2 was given), `no_paths`
(none: step 2 answered without the graph), the per-step seconds and, on CWQ / WebQSP, the topic entities started from.
GCR's rows count no LLM calls or tokens (step 1 is its own 8B).

## Edits to their clone (`patch_upstream.py`)

None to the algorithm or the prompts; each inserted line is marked `# patched by baselines/gcr/patch_upstream.py v3`.

1. `src/llms/__init__.py`: our served models, by name (`qwen`, `gpt-oss`), go to the OpenAI-client class.
2. `src/llms/chatgpt.py`: `SERVED`, the names of our served models.
3. A served model's context window is the servers' 32,768 tokens, instead of an error.
4. Token counting for a served model is characters / 3 and never asks tiktoken (it would download an encoding, which an
   offline machine cannot); for any other name a failure to count is estimated the same way (the count only decides
   whether the prompt fits).
5. The decoding comes from the environment `step2_env.py` writes: `GCR_MAX_TOKENS` as max_tokens, `GCR_TOP_P` and
   `GCR_TOP_K` as top_p and top_k, and `GCR_REASONING_EFFORT`, when set, as reasoning_effort. The temperature is
   their 0.0.
6. `TIKTOKEN_CACHE_DIR` is theirs (`./tmp`) unless the environment sets one.
7. A question `timed.py` has dropped (its 200 s spent; their thread pool cannot stop a thread, so `timed.py` marks it)
   makes no further attempt: their retry loop returns None at its next turn.

## Edits to our files for this release

Comments and docstrings shortened; the code is the paper harness's apart from these edits (each checked: the files are
AST-identical to the source with exactly these edits applied):

| file | change |
|---|---|
| all | imports from the paper harness rewired to this release's modules (table below) |
| `export.py` | FiDeLiS's helpers it imported (`SKIP`, `TRIES`, `PAUSE`, `SLOW_IN_A_ROW`, `timed_out`, `with_retries`, `_complete_last_line`) moved into it unchanged (FiDeLiS is not in this release); the questions read with `harness.dataset`; `QUESTION_FILE` empty (GrailQA's `test.json` here is the paper's 1,000); an unused variable (`m`) dropped |
| `convert.py` | the questions read with `harness.dataset`; `--results` default: a cluster folder became `results/eval` (relative to the folder it runs from, `$EVAL_ROOT` in `run_gcr.sh`); it also writes the folder's `run.json` (the fields the harness's runner writes), which `python -m harness score` needs; an unused import (`os`) dropped |
| `patch_upstream.py` | the patch's mark (`MARK`, `OLD_MARK`) no longer names our internal repository; one inserted comment lost an internal label |
| `step2_client.py` | `PINNED`: the authors' client files are read from `upstream/` (they were in a test folder) |
| `fetch.sh`, `build_env.sh` | the cluster's data and conda paths became `$GCR` / `$DATA_ROOT` and `$CONDA_SH` |
| `run_gcr.sh` | the cluster job script without its scheduler lines: the same commands in the same order; the cluster's shared shell library became the variables below and a `healthy_list` that reads `SERVER_<model>` |

| imported in the paper's harness | here |
|---|---|
| `common.settings` | `harness.settings` (the `gcr`, `kg_r1` and `kbbinder_official` rows are in its `SYSTEMS` and `DECODING` tables, with the paper's values) |
| `common.main._score_text_answer` | `harness.metrics.reasoning.score_text` (the same function, imported as `_score_text_answer`) |
| `common.manifest.run_manifest` | `harness.manifest.run_manifest`: the functions it calls, moved unchanged, except that the row's code record is keyed `code` (it was named after the paper's internal repository), a copy without `.git` reads a `COMMIT` file at the repository's root, and `kora` is always None (KORA is this repository's `kora/`) |
| `data/kora/splits/<set>/<split>.json` | `harness.dataset.records(set, split)` and `path_of` (`data/questions/<set>/`, and the Unseen halves from `data/unseen/<set>/`); GrailQA's `test.json` is the 1,000 every system ran (the paper's `test-rep1000.json`, the same ids) |
| `common.subgraph_topics.topic_nodes` | `harness.subgraph_topics.topic_nodes` (moved unchanged) |
| `common.dataset.UNSEEN` | `harness.dataset.UNSEEN` (the same four WebQSP / CWQ halves) |
| `baselines.fidelis_official.subgraphs.*` | moved into `export.py` (above) |

**Record differences from the paper's rows** (values only; scoring is unchanged): the run manifest's code record is
keyed `code` (the paper's used our internal repository's name), its `kora` is None and its `upstream` names only
KB-BINDER's folder; the Unseen halves' questions come from one pair file per set (`data/unseen/<set>/test.json`), so
`-unseen` and `-unseen-real` record the same split path and sha256 (the paper's had one file per half); GCR's export
sidecar names GrailQA's `question_file` as `test.json` (the paper's `test-rep1000.json`, the same 1,000 questions).

## Variables

| variable | meaning |
|---|---|
| `DATA_ROOT` | the data folder; `GCR` defaults to `$DATA_ROOT/gcr` (weights in `hf/`, environment in `env/`, the clone, exports in `data/`, runs in `runs/`) |
| `EVAL_ROOT` | this repository's root: `run_gcr.sh` puts it on `PYTHONPATH` and runs `convert.py` there |
| `CONDA_SH` | the `conda.sh` that `build_env.sh` and `run_gcr.sh` source |
| `HARNESS_ENV` | the conda environment `convert.py` runs in |
| `SERVER_<model>` | an answer model's OpenAI-compatible server, `http://<host>:<port>/v1`; `<model>` is the model name with every other character than letters and digits as `_` (`SERVER_qwen3_30b_a3b`) |
| `SETS`, `ANSWER_MODELS`, `RUN`, `DATA` | what `run_gcr.sh` runs; `SETS` defaults to the four standard sets |
| `GCR_CLONE` | `step2_client.py`: a patched clone of theirs to read their client files from (unset: the authors' files in `upstream/`, patched in a temporary folder) |
| `SUBGRAPH_TRIES`, `SUBGRAPH_RETRY_PAUSE` | `export.py`: how many times a question's graph queries are tried (unset: 2) and the seconds between two tries (unset: 30) |
| `CF_BIOKG_GRAPH` | `export.py`: the named graph the BioKG-layout export reads (`biocq_records`; unset: `http://biokg.org/`); no shipped set uses it |

## Steps

1. Once, with internet: `bash baselines/gcr/fetch.sh`; create an empty conda environment at `$GCR/env`, then
   `bash baselines/gcr/build_env.sh`; clone and patch theirs:
   ```
   git clone https://github.com/RManLuo/graph-constrained-reasoning $GCR/graph-constrained-reasoning
   git -C $GCR/graph-constrained-reasoning checkout 9518e8e
   python -m baselines.gcr.patch_upstream $GCR/graph-constrained-reasoning
   ```
2. Export every set (where the graph services are reachable; `METAQA_KB` is the `kb.txt` the MetaQA graph was built
   from; `--endpoint` is the set's SPARQL endpoint):
   ```
   python -m baselines.gcr.export --set cwq     --rog $GCR/hf/RoG-cwq    --out $GCR/data
   python -m baselines.gcr.export --set webqsp  --rog $GCR/hf/RoG-webqsp --out $GCR/data
   python -m baselines.gcr.export --set grailqa --endpoint <Freebase SPARQL endpoint> --out $GCR/data
   python -m baselines.gcr.export --set metaqa  --kb "$METAQA_KB" --out $GCR/data
   for s in webqsp-unseen webqsp-unseen-real cwq-unseen cwq-unseen-real; do   # for KG-R1's Unseen rows
     python -m baselines.gcr.export --set $s --endpoint <extended graph's SPARQL endpoint> --out $GCR/data --resume; done
   ```
   Each prints its counts: questions, how often the cap bit, how often a gold answer is inside the subgraph, and (CWQ,
   WebQSP) how many topic entities were renamed or not found.
3. A smoke, one set and 10 questions: `python -m baselines.gcr.export --set webqsp --rog $GCR/hf/RoG-webqsp --limit 10
   --out $GCR/data-smoke`, then `DATA=$GCR/data-smoke SETS=webqsp RUN=gcr-smoke bash baselines/gcr/run_gcr.sh`.
   Check: a new run folder; the step-2 line shows `max_tokens=1024 top_p=1.0 top_k=-1 reasoning_effort=none` for Qwen
   and `max_tokens=4096 ... reasoning_effort=low` for gpt-oss; the times files have one line per question; the convert
   line reports 0 over 200 s, 0 untimed, and "with no prediction" 0.
4. The runs, one GPU, a group of sets at a time:
   `SETS="cwq" ANSWER_MODELS="qwen3-30b-a3b gpt-oss-120b qwen3-4b" bash baselines/gcr/run_gcr.sh`, then
   `SETS="webqsp grailqa"` and `SETS="metaqa"`. A run stopped part-way leaves its unfinished set unconverted: run that
   set again (every run writes a new folder). Run folder (`run.json`, `s0.jsonl`):
   `$EVAL_ROOT/results/eval/<run>/<model>/<set>-test/gcr`; score it with `python -m harness score <that folder>`.

## Disclosed

- **Attention kernel.** Step 1 runs `--attn_implementation sdpa`; the authors' script uses `flash_attention_2` (the
  paper names none). Both are exact attention, but bf16 kernels round differently and group beam search can break a
  near tie the other way, so a path or two may differ from a flash-attention run. flash-attn is not in GCR's
  environment.
- **Step-2 threads.** `-n 8` questions answered at once; the authors' script runs `-n 10`. It changes only how many
  requests reach the server together; each question is timed on its own.
- **Hypothesis answers come from the 8B's own knowledge.** In step 1 only the path is held to the graph: after
  `</PATH>` the 8B decodes its `# Answer:` freely (paper §4.3), and step 2 reads those hypotheses beside the paths. A
  row with paths counts as answered over the graph even when its answer is a hypothesis the 8B wrote from what it
  learned on Freebase.
- **In-domain on CWQ and WebQSP.** The 8B was fine-tuned on their train splits; on GrailQA and MetaQA it transfers
  without retraining, as its paper runs it on graphs it was not trained on (§5.4, Table 6).
- **The cap on step 2**: 1,024 tokens on Qwen (gpt-oss: 4,096), the harness's; the authors' client sends none.
- **Model load outside the 200 s**: loading the 8B before step 1's first question is not counted.
- **Reach.** Paths have at most 2 hops (MetaQA's 3-hop questions cannot be answered over them); there is no counting or
  filtering; GCR cannot answer what its subgraph does not hold.
