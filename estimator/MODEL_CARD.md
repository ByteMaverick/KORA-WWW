---
license: mit
base_model: jhu-clsp/ettin-encoder-400m
library_name: transformers
pipeline_tag: text-classification
language:
- en
tags:
- knowledge-graph
- kgqa
- stopping
- agent
---

# KORA Stopping Estimator

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-under%20review-lightgrey)](#citation)
[![Base model](https://img.shields.io/badge/🤗%20Base-Ettin--400M-blue)](https://huggingface.co/jhu-clsp/ettin-encoder-400m)

> **TL;DR**: A 400M encoder that tells a knowledge-graph QA agent when it has found the answer. Drop it into an agent in
> place of "ask the LLM whether this is enough" and the agent gets more accurate, more grounded, and cheaper. It was
> trained once, on KORA's states, and the same frozen weights improve three other published agents without retraining.

This is the Stopping Estimator from "KORA: Knowledge Graph Question Answering with Learned Stopping" (under review).
The code that trains and serves it, the agent it was built for, the harness and the baselines are in the paper's
anonymous repository.

## Table of contents

- [Why a stopping estimator](#why-a-stopping-estimator)
- [Results](#results)
- [Quick start](#quick-start)
- [What it reads](#what-it-reads)
- [Threshold](#threshold)
- [Training data](#training-data)
- [Training](#training)
- [Validation numbers](#validation-numbers)
- [Files](#files)
- [Limits](#limits)
- [Citation](#citation)

## Why a stopping estimator

An LLM-guided agent searching a knowledge graph has to decide, at every step, whether what it has retrieved answers
the question. Most agents ask the LLM, and most prompts for that question invite the model to use "your knowledge" as
well. That is where the model's memory leaks into the answer: it stops when it recognises the answer, not when the
graph has shown it. This model makes that decision instead. It only ever sees the question and the retrieved evidence,
it never queries the graph, and it never writes an answer. It just says yes or no.

## Results

All numbers are from the paper, with Qwen3-30B-A3B as the backbone. Hits@Any (Hit) and set F1, in percent; the F1
values of the transfer table come from the released CSVs, since the paper's Table 2 prints Hit and support.

**Inside KORA.** The same agent with three stop rules: the LLM judging its own evidence, this estimator, and the
gold answers (a ceiling, not a system).

| stop rule | WebQSP | CWQ | GrailQA | MetaQA | mean Hit | mean F1 |
|---|---|---|---|---|---|---|
| the LLM decides | 81.2 | 64.1 | 76.8 | 90.9 | 78.2 | 68.5 |
| **this estimator** | **81.8** | **66.7** | **82.6** | **97.1** | **82.0** | **74.7** |
| gold answers (ceiling) | 86.8 | 74.8 | 90.6 | 95.1 | 86.8 | 79.5 |

On the Unseen twins, questions about entities added to Freebase that no model has seen in training, the gap is
wider: 79.6 vs 69.5 Hit on WebQSP-Unseen and 64.3 vs 56.2 on CWQ-Unseen, estimator vs LLM stop.

**Dropped into other agents, frozen.** ToG, PoG and CoR, each run as published and then with only their stop decision
replaced by this estimator. 500 pairs per set; "twins" is the half about new entities. Support is the share of answer
items that trace back to the agent's own graph calls.

| agent | set | their own stop: Hit / F1 / support | with this estimator: Hit / F1 / support |
|---|---|---|---|
| ToG | WebQSP-Unseen twins | 22.0 / 17.3 / 15.7 | **56.4 / 44.2 / 52.6** |
| ToG | CWQ-Unseen twins | 16.4 / 14.6 / 18.9 | **46.8 / 43.6 / 62.0** |
| PoG | WebQSP-Unseen twins | 67.0 / 47.2 / 46.6 | **71.2 / 60.2 / 91.6** |
| PoG | CWQ-Unseen twins | 44.4 / 37.6 / 59.1 | **51.8 / 45.7 / 86.7** |
| CoR | WebQSP-Unseen twins | 41.0 / 35.3 / 82.0 | **75.8 / 71.5 / 96.0** |
| CoR | CWQ-Unseen twins | 29.4 / 25.8 / 89.5 | **59.0 / 54.2 / 97.6** |

ToG's Hit on the WebQSP twins goes from 22.0 to 56.4 and CoR's from 41.0 to 75.8, with nothing else about those
agents changed. The source halves, the original questions, improve too, by smaller margins: ToG 71.8 to 77.0 and PoG
70.6 to 74.4 on the WebQSP-Unseen sources.

**On its own.** Validation AUROC .975, F1 .835 at the released threshold (details [below](#validation-numbers)).

## Quick start

The folder is laid out for the paper's server. With the repository checked out:

```
hf download anonymous-kgqa/kora-stopping-estimator --local-dir estimator-ckpt
python -m estimator.server --checkpoint estimator-ckpt --port 8100        # add --device cpu without a GPU
```

`GET /health` returns the threshold; `POST /score {"states", "questions", "extras"}` returns P(yes) per state. On a
CPU with 8 threads the 4,709 validation rows score in about 5 minutes.

Without the server, it is a plain `transformers` sequence classifier:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tok = AutoTokenizer.from_pretrained("estimator-ckpt/tokenizer")
model = AutoModelForSequenceClassification.from_pretrained("estimator-ckpt/model").eval()

question = "the films written by Elizabeth Myers were in which genres"
evidence = """EVIDENCE (facts retrieved so far, as subject | relation | object)
  Blackmailed | WRITTEN_BY | Elizabeth Myers

CANDIDATE ANSWER (1 items)
Blackmailed"""

enc = tok(question, evidence, return_tensors="pt", truncation=True, max_length=7999)
with torch.no_grad():
    p_yes = model(**enc).logits.softmax(-1)[0, 1].item()
print(p_yes, p_yes >= 0.04942165136337313)   # low: the path stops one hop short of the genres
```

The evidence text has to be in the exact format below. The repository's `estimator/render.py` writes it from an
agent's state, and it is the same renderer that wrote the training rows.

## What it reads

A text pair, `[CLS] question [SEP] evidence [SEP]`. The evidence is one `subject | relation | object` line per hop
(an unnamed intermediate set is written `?x<k>`, a constraint goes on its own line), relation names exactly as the
graph writes them, then the candidate answers as at most 30 names with the true count. The graph is never named. Up to
7,999 tokens; nothing was truncated in training (the longest validation row is 1,132 tokens). The example above is a
real validation row, and its label is no.

Output: two classes, P(yes) = softmax[1].

## Threshold

tau = 0.0494 (0.04942165136337313). It is the cut on the validation scores at which the larger of the false-negative
and false-positive rates is smallest, searched over 401 quantiles. It was chosen on validation only and frozen for
every test run in the paper.

## Training data

Rows from the training questions of WebQSP, CWQ, GrailQA, MetaQA and KQA Pro, 1,000 per set, labelled by the gold
rule: yes if the candidate holds at least half of the gold answers and at least half of it is gold. The repository's
`estimator/data/README.md` walks through every stage with its counts.

1. **Raw.** Data A: paths sampled from the graph and the gold (the gold path, ranked enumerated paths, and edits of
   the gold path: one hop short or past the answer, a sibling relation, a dropped or wrong constraint, a same-name
   start, random walks), each executed and its end set labelled. Data B: the states KORA (Qwen3-30B) showed its judge
   while the gold stopped it.
2. **Cleaned.** Rows no agent would send, or whose label cannot be trusted, dropped.
3. **Rendered.** Every row rendered to (question, evidence, label).
4. **Training rows.** A and B merged, plus copies of a random third with every entity name replaced by an invented
   one, so that names carry no signal: 150,087 rows, 16.9% yes.

The checkpoint was also trained on rows from BioCQ, a set that is not in the paper: 24,061 of the 150,087. The
repository ships the other 126,026 (`estimator/data/train.parquet`) and keeps BioCQ out of every data file.

Validation: the states of a gold-stopped KORA run on 666 validation questions (5,625 states, 916 of them BioCQ); the
repository ships the 4,709 states of the 537 non-BioCQ questions. Train, validation and holdout share no question.

## Training

The ModernBERT / Ettin GLUE recipe: a two-class head on the `[CLS]` token, full fine-tuning, AdamW with decoupled
weight decay (lr 3e-5, weight decay 8e-6), batch 32, 6% linear warm-up then linear decay, 2 epochs, bf16, seed 0,
micro-batches of 4,096 tokens. One GPU, 27 minutes. `checkpoint.json` records the run.

## Validation numbers

At tau 0.0494:

| rows | FN rate | FP rate | precision | recall | F1 | accuracy | AUROC |
|---|---|---|---|---|---|---|---|
| all 5,625 (the run's own scores, `validation_scores.npy`) | .073 | .073 | .759 | .927 | .835 | .927 | .975 |
| the 4,709 shipped (BioCQ removed) | .085 | .078 | .744 | .915 | .821 | .921 | .971 |

`estimator/train/threshold.py` on `validation_scores.npy` and the 5,625 labels gives tau 0.0494 and the first row
exactly. Scored again from these weights on CPU in float32, the scores differ from the run's bf16 GPU scores by .0004
on average (at most .041) and flip 2 of the 5,625 decisions; the threshold rule then picks tau .0487 with the same
FN rate, FP rate, precision, recall, F1 and accuracy, and AUROC .975. The shipped `validation.parquet` has an `idx`
column into `validation_scores.npy`, so `scores[idx]` gives the second row exactly; on those rows alone the rule would
pick tau .039. The released tau stays .0494.

## Files

| file | what it is |
|---|---|
| `model/config.json` | the encoder's configuration |
| `model/model.safetensors` | the fine-tuned weights, float32, 1.58 GB |
| `tokenizer/tokenizer.json` | Ettin's tokenizer |
| `tokenizer/tokenizer_config.json` | its configuration |
| `threshold.json` | what the server reads: tau, max length 7,999, `cls` pooling |
| `checkpoint.json` | the run: tau, max length, pooling, lr, epochs, seed, rows, validation metrics |
| `validation_scores.npy` | P(yes) for the 5,625 validation rows, in the order of the full validation file |

## Limits

- It was trained on the states of one agent, KORA, and on sampled paths, over Freebase (WebQSP, CWQ, GrailQA), MetaQA
  and KQA Pro. The threshold was chosen on KORA's validation states. The transfer results above are what happened when
  it was handed other agents' states unchanged; other graphs and other evidence formats have not been tried.
- It reads only what the agent shows it. It does not query the graph, and it cannot tell that evidence is missing if
  the agent never retrieved it.
- The evidence has to be rendered in the format above. A different rendering is a different input distribution.

## Licence

MIT, like the repository. The base encoder, Ettin-400M, is MIT.

## Citation

To be added after review.
