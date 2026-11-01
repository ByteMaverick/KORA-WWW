# KORA: Knowledge Graph Question Answering with Learned Stopping

**The LLM plans and searches. The graph and plain code produce the answer. A small learned estimator decides when to stop.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)
[![Paper: under review (WWW '27)](https://img.shields.io/badge/paper-under%20review%20%28WWW%20%2727%29-lightgrey.svg)](#citation)
[![Model: Hugging Face](https://img.shields.io/badge/model-Hugging%20Face-orange.svg)](https://huggingface.co/anonymous-kgqa/kora-stopping-estimator)
[![Data: in this repository](https://img.shields.io/badge/data-in%20this%20repository-green.svg)](data/)

Anonymous release for the paper, under review. Everything the paper used is here or linked:

- **Code**: this repository. KORA, the Stopping Estimator, the harness, seven baselines, the Unseen builder.
- **Data**: [`data/`](data/) and [`estimator/data/`](estimator/data/). The five question sets with splits, the Unseen pairs, the estimator's training data.
- **Model**: the Stopping Estimator, on Hugging Face at [anonymous-kgqa/kora-stopping-estimator](https://huggingface.co/anonymous-kgqa/kora-stopping-estimator).

## What KORA does

KORA answers questions over a knowledge graph (Freebase, MetaQA) with an LLM in the loop, but never lets the LLM
write the answer or decide when it has found it.

- **The LLM plans and searches.** It reads the question, writes a plan, and proposes graph queries from relations it
  has seen on the graph.
- **The graph and code produce the answer.** Every query is executed; filters, counts and superlatives run in code
  over the values the graph returned. Each answer value is traceable to the query that produced it.
- **A small learned model decides when to stop.** A 400M encoder reads the question, the retrieved path and the
  candidate answers and says whether that is enough. No prompt ever asks the LLM "is this sufficient?", so its own
  memory cannot decide the answer.
- **If nothing is accepted**, KORA keeps searching or splits the question into dependent sub-questions from the plan.

Why it matters: on questions about entities that were added to Freebase after the models were trained, systems that
let the LLM stop and answer collapse (ToG 70 → 23 Hits@Any); KORA keeps 80 of its 83, and every answer traces to
the graph. The same stopping model, dropped into ToG, PoG and CoR unchanged, lifts them too.

## Try it

You need Python 3.11+, a model on an OpenAI-compatible server (vLLM or a hosted API), and a graph: Freebase in
Virtuoso (WebQSP, CWQ, GrailQA) or MetaQA in Neo4j. [`data/README.md`](data/README.md#serving-the-graphs) says how to
load both. Run everything from the repository root.

**1. Install.**

```
pip install -e '.[embed,estimator]'
```

Add `baselines` to the extras for ToG, PoG, CoR and SoG. GCR, KG-R1 and KB-BINDER have their own environments (see
their `UPSTREAM.md`).

**2. Configure.** Copy [`.env.example`](.env.example) to `.env` and fill in the first block: the model server and
model id, the key, and the graph. Everything else has the paper's defaults.

```
cp .env.example .env
```

**3. Start the two local servers**, one terminal each. The checkpoint is the folder from the Hugging Face link above
(`hf download anonymous-kgqa/kora-stopping-estimator --local-dir estimator-ckpt`). Add `--device cpu` without a GPU.

```
python -m kora.embed_server --port 8001
python -m estimator.server --checkpoint estimator-ckpt --port 8100
```

**4. Ask a question.** `--topic` is the topic entity, as an id or `id=name`.

```
python -m kora answer "where is jamarcus russell from" --topic m.0cjcdj="JaMarcus Russell"
```

It prints one JSON object: the `answers`, and a `provenance` record with the status, the stop reason, the LLM calls,
tokens, seconds and graph calls, and every query each task ran. This one comes back `Mobile`, the gold answer, in
3 LLM calls. On MetaQA the topic is just the name, and `KORA_STOP` picks who stops: the estimator (default), the
model (`llm`), or the gold answers (`gold`, a ceiling; needs `--gold`).

```
python -m kora answer "what does Olga Baclanova star in" --graph metaqa --topic "Olga Baclanova"
KORA_STOP=llm  python -m kora answer "where is jamarcus russell from" --topic m.0cjcdj="JaMarcus Russell"
KORA_STOP=gold python -m kora answer "where is jamarcus russell from" --topic m.0cjcdj="JaMarcus Russell" --gold Mobile
```

**5. Run a system on a few questions and score it.**

```
python -m harness run --system kora --stop estimator --dataset webqsp --max-samples 5 --out runs/kora
python -m harness run --system tog  --stop native    --dataset webqsp --max-samples 5 --out runs/tog
python -m harness run --system tog  --stop estimator --dataset webqsp --max-samples 5 --out runs/tog-est
python -m harness score runs/kora runs/tog runs/tog-est
python -m harness compare runs/tog-est runs/tog
```

`run` writes one row per question to `<out>/s0.jsonl` and the model and graph calls under `<out>/traces/`; `score`
prints a CSV row per run (hit, F1, support, calls, tokens, latency); `compare` gives the paired difference with a 95%
interval. A wrong URL does not abort a run, it writes an error row per question, so check the `error` field first.

`--system`: `kora`, `tog`, `pog`, `cor`, `sog`, `kbbinder`, `direct_llm`. `--stop`: `estimator | llm | gold` for
KORA, `native | estimator | gold` for ToG, PoG and CoR, `native` for the rest. `--dataset`: `webqsp`, `cwq`,
`grailqa`, `metaqa`, or an Unseen set. KB-BINDER needs its data first (`KBB_DATA=<folder> bash
baselines/kbbinder/fetch.sh`, 250 MB). The `*-unseen` sets need the extended Freebase ([below](#the-unseen-benchmark)).

**6. The paper's numbers**: [`results/`](results/), one CSV per table and figure. `python results/make_tables.py`
prints them.

## What is where

| folder | contents |
|---|---|
| [`kora/`](kora/) | the agent. Planner → Query Agent → graph operations → Stopping Estimator, with a Controller that decomposes the question when nothing is accepted. Prompts in one file, [`kora/prompts/prompts.yaml`](kora/prompts/prompts.yaml); the run's record in [`kora/provenance/`](kora/provenance/). |
| [`estimator/`](estimator/) | the Stopping Estimator: server, renderer, training-data sampler, notebook, threshold. [README](estimator/README.md), [model card](estimator/MODEL_CARD.md), [data by stage](estimator/data/README.md). |
| [`harness/`](harness/) | one runner and scorer for every system. Same question sets, model client, 200 s limit and stop switch; the gold reaches a system only through `--stop gold`. Scoring in four parts under [`harness/metrics/`](harness/metrics/): reasoning (Hits@Any, F1), groundedness (graph-trace support), efficiency (calls, tokens, latency), unseen (the source-minus-twin gap). [README](harness/README.md). |
| [`baselines/`](baselines/) | ToG, PoG, CoR, SoG, GCR, KG-R1, KB-BINDER: the authors' code with our adapter and an `UPSTREAM.md` naming source, commit, licence, settings and edits. |
| [`data/`](data/) | the five question sets (WebQSP, CWQ, GrailQA, MetaQA, KQA Pro) with train, val and test; the Unseen pairs. [README](data/README.md). |
| [`unseen_builder/`](unseen_builder/) | builds the extended Freebase the twins are asked on, and loads it. [README](unseen_builder/README.md), [manifest](unseen_builder/MANIFEST.md). |
| [`results/`](results/) | the paper's CSVs, `make_tables.py`, and `GAPS.md` (683 of 724 printed cells match the submitted paper; the rest are support values the paper re-scored later). |

## The Unseen benchmark

Each pair is a WebQSP or CWQ question and its twin, the same question about entities added to Freebase: the gold
path's entities are cloned under new ids and names, padded with their sources' relation structure, dates and numbers
shifted by one order-preserving transform, and the question and gold query rewritten. 1,000 test and 200 validation
pairs per set, as `webqsp-unseen` / `webqsp-unseen-real` and `cwq-unseen` / `cwq-unseen-real` in the harness. The
extended graph is a Freebase copy with the new triples loaded next to it (`python -m unseen_builder.main load`); the
triples are released with the camera-ready version, so until then only the source halves run.

## Baselines

| system | folder | upstream |
|---|---|---|
| ToG (ICLR 2024) | [`baselines/tog/`](baselines/tog/UPSTREAM.md) | [IDEA-FinAI/ToG](https://github.com/IDEA-FinAI/ToG) |
| PoG (NeurIPS 2024) | [`baselines/pog/`](baselines/pog/UPSTREAM.md) | [liyichen-cly/PoG](https://github.com/liyichen-cly/PoG) |
| CoR (Findings of ACL 2026) | [`baselines/cor/`](baselines/cor/UPSTREAM.md) | [Chenhui7au/Chain-of-Relations](https://github.com/Chenhui7au/Chain-of-Relations) |
| SoG | [`baselines/sog/`](baselines/sog/UPSTREAM.md) | [OpenReview supplementary](https://openreview.net/forum?id=aAyONRMcdB) |
| GCR (ICML 2025) | [`baselines/gcr/`](baselines/gcr/UPSTREAM.md) | [RManLuo/graph-constrained-reasoning](https://github.com/RManLuo/graph-constrained-reasoning) |
| KG-R1 | [`baselines/kg_r1/`](baselines/kg_r1/UPSTREAM.md) | [junhongmit/KG-R1](https://github.com/junhongmit/KG-R1) |
| KB-BINDER (ACL 2023) | [`baselines/kbbinder/`](baselines/kbbinder/UPSTREAM.md) | [TIGER-AI-Lab/KB-BINDER](https://github.com/TIGER-AI-Lab/KB-BINDER) |

GCR and KG-R1 run from the steps in their folders; their last step writes a run folder that `harness score` reads.
The other five run as `--system`.

## Results at a glance

Qwen3-30B-A3B as the backbone for every system. Hits@Any / F1, in percent, from the paper's Tables 1 and 3 (the
Unseen F1 values are from the shipped CSVs; the paper's Table 3 prints Hits@Any and support).

| | WebQSP · CWQ · GrailQA · MetaQA (mean) | WebQSP-Unseen | CWQ-Unseen | LLM tokens per question, pooled over the four sets |
|---|---|---|---|---|
| **KORA** | **82.0 / 74.7** | **79.6 / 72.2** | **64.3 / 59.6** | 18.4K in, 1.2K out |
| SoG | 79.0 / 70.5 | 73.9 / 64.5 | 63.3 / 59.4 | 71.0K in, 1.1K out |
| PoG | 64.2 / 48.6 | 64.0 / 43.5 | 42.3 / 36.5 | |
| ToG | 59.6 / 45.6 | 22.8 / 17.4 | 15.7 / 14.1 | |
| Direct LLM, no graph | 37.1 / 22.5 | 4.2 / 2.5 | 3.1 / 1.7 | |

The Unseen sets ask the same questions about entities that were added to Freebase after the models were trained, so
the model's memory cannot help. Every KORA answer item there traces back to its own graph calls (100% graph-trace
support). The same
frozen estimator, dropped into ToG, PoG and CoR in place of their own stop rule, lifts them on the twins too (ToG 22 →
56 Hit on WebQSP-Unseen). Every number is in [`results/`](results/).

## Using KORA

One question: `python -m kora answer "who plays ken barlow in coronation street" --topic m.01_2n="Coronation Street"`.
It prints the answers and a `provenance` record tracing every value to the query that returned it. `KORA_STOP=llm|gold`
changes who stops; `.env.example` has every setting with the paper's value. Under the harness: `--system kora`.
Details and layout: [`kora/`](kora/).

## Using the Stopping Estimator

`hf download anonymous-kgqa/kora-stopping-estimator --local-dir estimator-ckpt`, then
`python -m estimator.server --checkpoint estimator-ckpt --port 8100`. `POST /score` with the question, the executed
query and its result ids returns P(yes); stop when it reaches the threshold from `GET /health`. The API, the training
data and retraining: [`estimator/README.md`](estimator/README.md); scoring without the server:
[`estimator/MODEL_CARD.md`](estimator/MODEL_CARD.md).

## Using the Unseen data

The pairs are in [`data/unseen/`](data/unseen/) and run as `--dataset webqsp-unseen` / `cwq-unseen` (twins) and
`*-unseen-real` (sources). They need the extended graph: a Freebase copy with the inserted triples loaded by
`python -m unseen_builder.main load` (the triples are released with the camera-ready version). Fields and setup:
[`data/README.md`](data/README.md#unseen-pairs); building new pairs: [`unseen_builder/README.md`](unseen_builder/README.md).

## License

MIT ([`LICENSE`](LICENSE)). Vendored code under `baselines/` stays under its authors' terms, named in each `UPSTREAM.md`.

## Citation

To be added after review.
