# The Stopping Estimator

A fine-tuned Ettin-400M encoder that reads a state (the question, the path retrieved so far as subject | relation |
object lines, and the candidate answer set) and gives P(the candidate answers the question). An agent stops when
P >= tau. KORA calls it over HTTP (`ESTIMATOR_URL`); the baselines' stop adapters call the same server.

| part | what it does |
|---|---|
| `render.py` | the one renderer: writes every training row and every live request, so the two cannot differ |
| `server.py` | serves a checkpoint over HTTP |
| `sampler/` | generates Data A from the graph and the gold, with no agent run (Freebase, MetaQA, KQA Pro) |
| `train/stop_judge.ipynb` | the training notebook: splits, checks, cleaning and rendering of the data, model selection, fine-tuning, threshold, DAgger rounds, transfer |
| `train/rename.py`, `train/threshold.py` | the renamed copies of the released estimator's training rows; tau from scores |

Install the extra first: `pip install '.[estimator]'` (torch, transformers, pandas, pyarrow, scikit-learn, scipy,
matplotlib).

## Serve a checkpoint

```
python -m estimator.server --checkpoint DIR --port 8100        # or set ESTIMATOR_CHECKPOINT
```

`DIR` holds `model/`, `tokenizer/` and `threshold.json` (`{"tau": 0.0494}` for the paper's estimator). Then point
KORA at it with `ESTIMATOR_URL=http://localhost:8100`; KORA reads tau from `/health` unless `ESTIMATOR_TAU` is set.

- `POST /score` `{"states": [...], "questions": [...], "extras": [...]}` -> `{"p": [...]}`. Each extra carries the
  candidate ids, their count, the executed query and the chain of queries the state stands on; the server names the
  ids from the graph and renders the pair. A state it cannot render scores 0.0 (keep searching).
- `GET /health` -> `{"ok": true, "threshold": tau, ...}`.
- Freebase ids are named from `FREEBASE_SPARQL_URL` (`--freebase-url`). For the Unseen sets point it at the extended
  graph, so a cloned entity gets the name of the graph that holds it. MetaQA ids are their own names.
- `--log FILE` appends every scored state (score, rendered text) as JSON lines; `--device cpu` runs without a GPU.

The paper's checkpoint is on Hugging Face, `https://huggingface.co/anonymous-kgqa/kora-stopping-estimator`; its model card is `estimator/MODEL_CARD.md`, and
`estimator/checkpoint.json` records its run (tau, max length, pooling, lr, epochs, seed, rows, validation metrics).
The download is a checkpoint folder as the server reads it: `model/config.json`, `model/model.safetensors`,
`tokenizer/tokenizer.json`, `tokenizer/tokenizer_config.json`, `threshold.json` (tau 0.0494), `checkpoint.json` and
`validation_scores.npy`. Serve it as it is:

```
python -m estimator.server --checkpoint <downloaded folder> --port 8100        # add --device cpu without a GPU
```

On 8 CPU threads it scores the 4,709 shipped validation rows in about 5 minutes. It was trained on BioCQ rows too (24,061 of
its 150,087 training rows), which this release does not ship (`estimator/data/README.md`).

## Training data

A row is `{"question", "evidence", "label", "meta"}`: the rendered pair, the label, and metadata the model never sees.
The label is the gold rule: the candidate holds at least half of the gold answers and at least half of it is gold.

- **Data A** is generated from the graph and the gold of training questions by `sampler/`: the gold path, enumerated
  relation paths ranked by embedding similarity to the question, and edits of the gold path (one hop short or past the
  answer, a sibling relation, a dropped or wrong constraint, a same-name start, random walks). Each path is executed and
  its end set labelled.
- **Data B** is the states KORA (Qwen3-30B) reached on training questions while the gold stopped it, as its judge
  was shown them. Validation and holdout states were harvested the same way on their questions.
- BioCQ is not in the paper, so its rows are removed from every parquet file below; KQA Pro stays. The released
  checkpoint was trained on the full 150,087 training rows, 24,061 of them BioCQ; the shipped `train.parquet` holds
  the other 126,026, each a row of the checkpoint's training file, unchanged.

### The data in `estimator/data/`

Everything the released estimator was built from, BioCQ aside, by training stage: raw (1), cleaned (2), rendered (3),
and the training, validation and holdout rows (4). `estimator/data/README.md` walks through the stages, with the
command or notebook section that made each file and the row counts at each step. The `.jsonl.gz` files are gzipped JSON lines; the notebook
reads them uncompressed, so run `gunzip -k estimator/data/raw/*/*.gz` first. The notebook's `DATA_DIR` is
`ESTIMATOR_DATA_DIR`, default `estimator/data`, a path relative to the working directory: start Jupyter at the
repository root, or set `ESTIMATOR_DATA_DIR` to an absolute path.

| file | rows | columns | made by | read by |
|---|---|---|---|---|
| `raw/generated/<set>.rows.jsonl.gz` (webqsp, cwq, grailqa, metaqa, kqapro) | 21,253 / 22,885 / 22,065 / 16,621 / 18,318 | `state` (the generator's text), `questions`, `expected.sufficient` (the label) | `python -m estimator.sampler.build` | notebook §1 |
| `raw/generated/<set>.meta.jsonl.gz` | the same, line for line | `qid`, `bench`, `question`, `evidence`, `candidate`, `n_candidate`, `label`, `source` (spine, enum, sibling, extend, ...), `hops`, `recall`, `precision`, `score`, `relations`, `directions`, `constraints`, `constraint_lines`, `lucky` | the same command | notebook §1 |
| `raw/oracle_runs/<set>.jsonl.gz` (webqsp, cwq, grailqa, metaqa) | 9,386 / 16,887 / 9,505 / 7,421 | one judged state per line: `ts`, `qid`, `site` (leaf/root), `state`, `ids`, `names`, `goal`, `recall`, `precision`, `f1`, `n_gold`, `n_got`, `names_truncated`, `hops`, `label`, `label_f1`, `hit1`, and the judge's own verdict (`llm_ok`, `llm_reason`); a leaf line adds `task`, `cand`, `query`, `judge` and `llm_best`, a root line `target` (the task offered to the root) and `llm_conf` (the judge's confidence) instead | KORA with `--stop gold` on 1,000 training questions per set | notebook §1 |
| `cleaned/A.parquet` | 71,850 | `qid`, `bench`, `source`, `question`, `evidence[]`, `candidate[]`, `n_candidate`, `relations[]`, `directions[]`, `constraints`, `constraint_lines[]`, `label`, `recall`, `precision`, `text` | notebook §4.1 (cleaning) | §4.1b, §4.2 |
| `cleaned/B.parquet` | 31,291 | `qid`, `bench`, `site`, `task`, `goal`, `is_subquestion`, `state`, `query`, `ids[]`, `names[]`, `n_got`, `n_gold`, `label`, `recall`, `precision`, `names_truncated`, `ts`, `cand` | notebook §4.1 | §4.1b, §4.2 |
| `rendered/A.parquet` | 71,715 (17.2% yes) | `idx`, `question`, `evidence`, `label`, `meta` {`qid`, `bench`, `source`, `site`} | notebook §4.2 (rendering) | §6, `rename.py` |
| `rendered/B.parquet` | 30,596 (22.9% yes) | as A, `meta` also {`ts`, `cand`, `page_round`, `page_dropped`}, plus `page` | notebook §4.2, `page` added in §4.4 | §6, `rename.py` |
| `validation.parquet` | 4,709 (19.8% yes) | as B | notebook §4.3 and §4.4, from a gold-stopped KORA run on 666 validation questions (537 without BioCQ) | §6, §7, `threshold.py` |
| `holdout.parquet` | 6,418 (24.2% yes) | as B | notebook §4.4b, from a gold-stopped KORA run on 956 holdout questions (756 without BioCQ) | §6.3 |
| `train.parquet` | 126,026 (17.8% yes) | `question`, `evidence`, `label`, `meta` | `python -m estimator.train.rename` (below) on the full rendered A and B, BioCQ rows then removed | the final fine-tuning |
| `stats.json` | | the renaming statistics of the full run, BioCQ included: the three renamed validation copies (5,625 rows each), A + B merged (112,589 rows), the renamed copies (37,530 drawn, 37,498 kept after deduplication) and the result (112,589 + 37,498 = 150,087) | `rename.py` (`renamed` for the validation copies, `with_renamed_copies` for the training rows) | |

Validation, holdout and train questions are disjoint. Only the parquet results of the validation and holdout runs are
shipped, not their run logs. KQA Pro has generated rows only (Data A): KORA was not run on it for Data B. The sampler's
`<set>.stats.json` files are not shipped.

Beyond these files, the notebook reads, under `DATA_DIR/raw/`:

- the question files, `raw/questions/<set>/train.json` (every set) and `raw/questions/<set>/val.json`: shipped as
  `data/questions/<set>/`. Unpack the two gzipped train files (`gunzip -k data/questions/cwq/train.json.gz
  data/questions/grailqa/train.json.gz`), then link or copy `data/questions` to `DATA_DIR/raw/questions`;
- not shipped: the decision states of the live runs, `raw/onpolicy/*.decisions.jsonl` (§1's `LIVE` cell joins them and
  fails without them), the test questions' ids and texts, `raw/test_questions.json` (§2 and §7 keep them out), the
  split id lists (`raw/qids/`), the encoders (`kora/models/`) and, in its graph checks (§3.6–3.9), the live graphs.

Its §1 reads BioCQ's raw files too (`SETS` lists `biocq`); on the shipped files, drop `biocq` from `SETS` there.

The holdout questions are drawn from the training questions (§2), so they are in `data/questions/<set>/train.json`:
data rebuilt from `train.json` alone holds the holdout and leaks it into training. Leave out the qids of
`holdout.parquet` (and of `validation.parquet`), for example as a JSON list of ids passed to the sampler's `--exclude`.

### Generate Data A

The sampler (`sampler/`) reads a question file (`{"questions": [{"id", "question", "answers", "answer_ids",
"topic_entity", "gold"}]}`; KQA Pro's carry their KoPL `program`) and writes `<set>.rows.jsonl`, `<set>.meta.jsonl` and
`<set>.stats.json`, the files the notebook reads from `raw/generated/`. The training questions are
`data/questions/<set>/train.json` (unpack CWQ's and GrailQA's first, as above):

```
python -m estimator.sampler.build webqsp freebase --questions webqsp_train.json --out $DATA_DIR/raw/generated --exclude val.json test.json
python -m estimator.sampler.build cwq freebase --questions cwq_train.json --out $DATA_DIR/raw/generated --exclude val.json test.json
python -m estimator.sampler.build grailqa freebase --questions grailqa_train.json --out $DATA_DIR/raw/generated --exclude val.json test.json
python -m estimator.sampler.build metaqa metaqa --questions metaqa_train.json --out $DATA_DIR/raw/generated --exclude val.json test.json
python -m estimator.sampler.build kqapro kqapro --questions kqapro_train.json --out $DATA_DIR/raw/generated --exclude val.json test.json
python -m estimator.sampler.rows $DATA_DIR/raw/generated/webqsp.meta.jsonl --holdout val.json test.json --rows $DATA_DIR/raw/generated/webqsp.rows.jsonl
```

Freebase is read from `FREEBASE_SPARQL_URL`, MetaQA from `NEO4J_METAQA_URI`, KQA Pro from `KQAPRO_SPARQL`. Paths are
ranked with KORA's embedder (`KORA_EMBED_BASE_URL`, `KORA_EMBED_MODEL`); `--scorer lexical` ranks by word overlap
instead. `--exclude` keeps every held-out question (by id and by text) out of the sample. The other flags:
`--qids FILE` (only these ids), `--max-hops 2`, `--workers 8`, `--limit N`, `--seed 0`.

| file | what it holds |
|---|---|
| `graph.py` | what the sampler needs from a graph (read-only: ids and names), an in-memory graph and a random walk |
| `paths.py` | the paths of one question: the enumeration, the edits of the gold path, and the pick of rows |
| `rows.py` | a row: the path rendered as evidence lines, its label from the gold, the files; and the checks command |
| `build.py` | the command: one set, every question in parallel |
| `adapters/` | one graph family each: `freebase` (WebQSP, CWQ, GrailQA), `metaqa`, `kqapro` (read through `kqapro_graph.py`, a Virtuoso client for the BioKG layout); each has `make_graph(bench)` |

How a question becomes rows:

- **The spine.** The adapter executes the gold (Freebase: the gold SPARQL, or GrailQA's s-expression; MetaQA: its
  hops; KQA Pro: its KoPL program, walked step by step) as a chain of hops ending on the gold set.
- **The enumeration.** From each anchor, relation paths ranked by how their names read against the question: hop 1
  follows the 50 best relations, the 10 best hop-1 paths continue through 15 relations, the 5 best hop-2 paths
  through 10; an unnamed set (a mediator) is followed one more step without counting as a hop. About 250 executed
  paths a question.
- **The edits** of the gold path, the near misses a stop decision has to reject: `truncate` (one hop early),
  `extend` (one hop past the answer), `sibling` (the last relation swapped for a close-reading sibling),
  `drop_constraint` (the gold with one constraint removed), `wrong_value` (a filter reversed or given another value),
  `same_name` (the same chain from another node with the anchor's name) and `random` (a walk of the spine's length).
- **The pick.** Every positive up to 8; every edit up to 20; 8 plausible enumerated wrong paths, small answers first;
  4 easy ones, random walks first. Rows are deduplicated by (relation chain, candidate set).
- **The row.** The path as `subject | relation | object` lines (an unnamed set as `?x<k>`, each constraint on its own
  line after the hop it restricts), the candidate (30 names shown, the true count kept) and the label: the candidate
  holds at least half the gold and at least half of it is gold. A yes that shares no relation with the gold path is
  marked `lucky` and left out of training.
- **The checks** (`python -m estimator.sampler.rows`): no question of a held-out file in the rows, the share of gold
  paths that are not positive, rows and positives per source, look-alike yes / no pairs, and rendered lengths. It exits
  1 when a held-out question leaks.

**KQA Pro.** Its graph is KQA Pro's closed subset of Wikidata, served by Virtuoso in the layout BioCQ's graph uses:
`<BASE>/node/KQAPro:<n>` nodes with an `rdfs:label` and an `rdf:type` per concept, and relations and attributes under
`<BASE>/relation/<slug>`. `kqapro_graph.py` reads it in the dialect KORA's query agent writes, bare tokens rather than
URIs: an entity token is the benchmark's own short id (`KQAPro:13213`), a predicate token is `<graph>.<relation>` with
spaces and other punctuation written as underscores (`kqapro.<relation>`), which a model can be asked to produce
without copying a long URI, and `:type` stands for a kind; the client translates the tokens to URIs.

A question's gold is its KoPL program, which the adapter walks step by step on the endpoint: Find (every entity with
the name), Relate, And, Or, FilterConcept (a class constraint), FilterStr / FilterNum / FilterYear / FilterDate
(filters on an attribute), and the final What, Count, QueryAttr, SelectAmong or SelectBetween. A FindAll, a qualifier
step or a set over 300 nodes is not walked: the walk raises `Unwalkable` with its reason instead of guessing. Each final
step renders as a path; a constraint narrows its hop's objects and is not a line of its own:

| final step | the path | the candidate |
|---|---|---|
| What | the hops | the last set |
| Count | the hops, then `set \| count \| n` | the number n |
| QueryAttr | the hops, then `entity \| kqapro.<attribute> \| value` | the value(s) |
| SelectAmong | the hops, `set \| kqapro.<attribute> \| v1, v2, ...` (the set's order), then `v1, v2, ... \| kqapro.<attribute> (largest) \| winner` (or `(smallest)`) | the winner |
| SelectBetween | the two compared entities as anchors, then the same two hops | the winner |

A value node carries the literal as the graph writes it (`114 metre`, `1844/1/1`), so a name match with the split's
answers is a match on the graph's own text.

The notebook then drops rows no agent would send (no named candidate, a path starting from several entities at once,
a path that returns to its start), renders the rest with `render.from_path`, and drops texts that carry both labels.

## Train

The notebook trains with the ModernBERT / Ettin GLUE recipe: a two-class head on the first token, full fine-tuning,
AdamW with decoupled weight decay (lr 3e-5, weight decay 8e-6), batch 32, 6% linear warm-up then linear decay,
2 epochs, bf16, seed 0, at most 7,999 tokens a row with nothing truncated. It needs one CUDA GPU.

The released estimator adds renamed copies: A and B are merged (B first; a text in both kept once; a text with both
labels dropped), then copies of a random third of the rows with every entity name replaced by an invented one are
added, 150,087 rows for the released checkpoint (`estimator/data/train.parquet` holds its 126,026 non-BioCQ rows):

```
python -m estimator.train.rename --rows estimator/data/rendered/A.parquet estimator/data/rendered/B.parquet --fraction 1/3 --seed-tag j1h --out train.parquet
```

and the notebook's `train_and_save` fine-tunes on that file as on A + B (the paper's run used micro-batches of 4,096
tokens, which changes memory only, not the gradients). It writes the model and tokenizer to `ckpt/` and tau to
`judge.json`; to serve, put them in `DIR/model/`, `DIR/tokenizer/` and `DIR/threshold.json` (`{"tau": ...}`). `train/rename.py` also has the other two name sources (random letter strings, other
real names from the rows) for scoring a checkpoint on renamed validation states.

On the shipped files the `rename` command reproduces the merged A + B part of `train.parquet` (its first 94,523 rows) row for row,
but not its renamed copies: the third that was renamed was drawn over the full rows, BioCQ included, and the draw
changes with its input (`estimator/data/README.md`, stage 4). On the full rendered A and B it reproduces the
checkpoint's 150,087 rows row for row.

## Pick tau

tau is the cut on the validation scores at which the larger of the false-negative and false-positive rates is
smallest, searched over 401 quantiles of the scores. It is chosen on validation only and frozen for every test run.

```
python -m estimator.train.threshold --scores validation_scores.npy --labels estimator/data/validation.parquet --out DIR/threshold.json
```

The downloaded checkpoint's `validation_scores.npy` holds 5,625 scores, BioCQ's rows included, so it does not line up
with the shipped file's 4,709 labels. `validation.parquet`'s `idx` column indexes those scores: `scores[idx]` gives the
model card's second row at the released tau, and the rule picks tau .039 on the shipped rows (the released tau stays
.0494):

```
python -c "import numpy as np, pandas as pd; from estimator.train.threshold import metrics; v = pd.read_parquet('estimator/data/validation.parquet'); s = np.load('DIR/validation_scores.npy')[v.idx.to_numpy()]; np.save('shipped_scores.npy', s); print(metrics(v.label.to_numpy(), s, 0.04942165136337313))"
python -m estimator.train.threshold --scores shipped_scores.npy --labels estimator/data/validation.parquet
```
