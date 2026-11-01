# The estimator's data, stage by stage

The files below are what the Stopping Estimator was trained and validated on, in the order the training used them.
"Notebook §n" is a section of `estimator/train/stop_judge.ipynb`. The columns of each file are in `estimator/README.md`.
Paths are relative to `$DATA_DIR` (`ESTIMATOR_DATA_DIR`, default `estimator/data`). The `.jsonl.gz` files are gzipped;
the notebook reads them uncompressed (`gunzip -k estimator/data/raw/*/*.gz`).

| stage | file | what it is |
|---|---|---|
| 1 | `raw/generated/<set>.{rows,meta}.jsonl.gz` | Data A as the sampler wrote it |
| 1 | `raw/oracle_runs/<set>.jsonl.gz` | Data B as KORA's gold-stopped runs logged it |
| 2 | `cleaned/A.parquet`, `cleaned/B.parquet` | A and B after the notebook's cleaning |
| 3 | `rendered/A.parquet`, `rendered/B.parquet` | A and B rendered to (question, evidence, label) |
| 4 | `train.parquet` | the training rows: rendered A + B, plus renamed copies of a third of them |
| 4 | `validation.parquet` | the validation rows: model selection and the threshold |
| 4 | `holdout.parquet` | the held-out test rows, read once |
| 4 | `stats.json` | the renaming statistics |

**BioCQ is not in this release.** It is not in the paper, so its rows are removed from every parquet file; KQA Pro
stays. Every row of a shipped parquet file is a row of the file the estimator was built from, unchanged: the files
lost only their BioCQ rows. The released checkpoint was trained on the full 150,087 training rows, 24,061 of them
BioCQ; `train.parquet` holds the other 126,026.

## Per set

Rows (questions). Data A: generated → `cleaned/A` → `rendered/A`. Data B: oracle run → `cleaned/B` → `rendered/B`.

| set | generated | cleaned/A | rendered/A | oracle run | cleaned/B | rendered/B | train | validation | holdout |
|---|---|---|---|---|---|---|---|---|---|
| WebQSP | 21,253 (947) | 16,448 (947) | 16,437 | 9,386 (1,000) | 7,752 (947) | 7,308 | 28,925 | 1,089 (125) | 1,680 (198) |
| CWQ | 22,885 (997) | 12,189 (996) | 12,158 | 16,887 (1,000) | 10,804 (997) | 10,699 | 29,057 | 1,549 (138) | 2,105 (193) |
| GrailQA | 22,065 (996) | 17,151 (996) | 17,059 | 9,505 (999) | 7,840 (995) | 7,718 | 30,498 | 1,257 (136) | 1,430 (165) |
| MetaQA | 16,621 (903) | 15,381 (903) | 15,381 | 7,421 (1,000) | 4,895 (904) | 4,871 | 23,262 | 814 (138) | 1,203 (200) |
| KQA Pro | 18,318 (1,000) | 10,681 (793) | 10,680 | none | none | none | 14,284 | none | none |
| total | 101,142 | 71,850 | 71,715 | 43,199 | 31,291 | 30,596 | 126,026 | 4,709 | 6,418 |

Train, validation and holdout share no question.

## Stage 1: what the sampler and the oracle runs produced

- **Data A**, `raw/generated/<set>.rows.jsonl.gz` and `<set>.meta.jsonl.gz` (the same rows, line for line): made by
  `python -m estimator.sampler.build <set> <adapter> --questions <train.json> --out $DATA_DIR/raw/generated --exclude
  <validation> <test>` (adapter `freebase` for WebQSP, CWQ and GrailQA, `metaqa`, `kqapro`) on the 1,000 training
  questions per set of notebook §2. A row is a path executed on the graph, its end set, and its label from the gold:
  101,142 rows. How the sampler picks the paths is in `estimator/README.md` ("Generate Data A").
- **Data B**, `raw/oracle_runs/<set>.jsonl.gz`: every state KORA (Qwen3-30B) showed its judge while the gold stopped
  it, on the same 1,000 training questions per set: 43,199 states. KQA Pro has none: KORA was not run on its training
  questions.

The notebook reads both in §1 (exploration), §2 (the splits) and §3 (the audit).

## Stage 2: cleaning (notebook §4.1)

Rows no agent would send, or whose label cannot be trusted, are dropped:

- **A, 101,142 → 71,850.** A path starting from several anchors at once, a path that returns to its start, no named
  candidate, a yes that shares no relation with the gold path ("lucky"), and duplicates.
- **B, 43,199 → 31,291.** Sub-question states, states on held-out questions, duplicates, and states with no nameable
  candidate.

The notebook's counts per reason are over its six sets, BioCQ included. A, by first reason: several anchors 17,604,
back to its start 12,269, no named candidate 2,817, lucky 808, duplicates 148. B: sub-question states 13,789, held-out
questions 1,756, duplicates 1,092, no nameable candidate 61.

Every row of `cleaned/A` is a line of the generated files and every row of `cleaned/B` is a state of the oracle runs.
§4.1b audits the result.

## Stage 3: rendering (notebook §4.2–4.4)

The one renderer, `estimator/render.py`, turns a row into the pair the model reads: the question, and the evidence
(the path as `subject | relation | object` lines, then the candidate answer set). The server renders every live
request with the same functions, so training and serving cannot differ.

- **A, 71,850 → 71,715**, with `render.from_path`: rows whose path lines cannot be read back into hops, rows that
  fail validation, yes rows whose candidate set stopped at the sampler's 200 limit, and texts that carry both labels.
- **B, 31,291 → 30,596**, with `render.from_kora`: states that fail validation, states that render the same as
  another state of the question, and texts with both labels. §4.4 adds the `page` column (the schema the query agent
  saw); the model does not read it.
- **Validation** (§4.3, §4.4) and **holdout** (§4.4b) are rendered the same way from gold-stopped KORA runs on 666
  validation and 956 holdout questions, BioCQ included; the shipped files hold 537 and 756 of them. Only the rendered
  rows are shipped, not the runs.

## Stage 4: the training rows, the validation rows, the threshold

**`train.parquet`** is rendered A and B together, then renamed copies:

1. A and B merged (B first; a text in both kept once; a text with both labels dropped): its first 94,523 rows.
2. Copies of a random third of the merged rows with every entity name replaced by an invented one, so names carry no
   signal: its other 31,503 rows.

```
python -m estimator.train.rename --rows estimator/data/rendered/A.parquet estimator/data/rendered/B.parquet --fraction 1/3 --seed-tag j1h --out train.parquet
```

What reproduces: on the shipped `rendered/` files this command gives the first 94,523 rows of `train.parquet`, row for
row. What does not: the renamed third was drawn over the full merged rows, BioCQ included. There it drew 37,530
copies and kept 37,498 (the 32 copies with no name to rename equal their original and are dropped as duplicates):
112,589 + 37,498 = 150,087 rows, of which `train.parquet` holds the 31,503 non-BioCQ copies. The draw depends on its
input, so on the shipped files the command draws another third and keeps 31,479 copies, none of which is a row of
`train.parquet`. `train.parquet` is therefore shipped as the checkpoint's own training
file with its BioCQ rows removed, not regenerated. On the full inputs the command reproduces the checkpoint's 150,087
rows row for row.

`stats.json` holds the statistics of that full run: the merged rows (`j1b`, 112,589), the renamed copies drawn (`j1h copies`,
37,530; 37,498 kept), the result (`j1h`, 150,087), and the three renamed copies of the full validation file (5,625 rows each, BioCQ
included; `val_invented`, `val_random`, `val_real`), used to check that the estimator does not lean on names.

**`validation.parquet`** picks the model and tau: tau is the cut on the validation scores at which the larger of the
false-negative and false-positive rates is smallest (`estimator/train/threshold.py`), frozen for every test run.
**`holdout.parquet`** is read once, in §6.3.

## The notebook's sections and the stages

| notebook | stage | reads | writes |
|---|---|---|---|
| §1 Exploratory analysis | 1 | `raw/generated/`, `raw/oracle_runs/` | |
| §2 Splits | 1 | the question files and id lists | `splits/` (not shipped) |
| §3 Audit of A and B | 1 | the raw files, the live graphs | |
| §4.1 Cleaning, §4.1b audit | 2 | the raw files | `cleaned/A.parquet`, `cleaned/B.parquet` |
| §4.2 Pre-processing, §4.2b | 3 | `cleaned/` | `rendered/A.parquet`, `rendered/B.parquet` |
| §4.3 Validation states, §4.4 the page, §4.4b Holdout states | 3 | the validation and holdout runs (not shipped) | `validation.parquet`, `holdout.parquet`, the `page` column |
| §5 Feature engineering, §6 Model selection | 4 | `rendered/`, `validation.parquet` | models, scores |
| §6.3 The holdout | 4 | `holdout.parquet` | |
| §7 DAgger rounds, §8 Transfer | after | runs not shipped | |
| §9 The final judge | 4 | describes the released estimator: `train_and_save` on `train.parquet` (made by `rename.py`, outside the notebook) | the checkpoint, tau |

On the shipped files, drop `biocq` from `SETS` in the notebook's first code cell: its raw files are not shipped.

## Notes

- The sampler's `<set>.stats.json` files are not shipped, and the exact sampler flags of the paper's run (for example
  `--max-hops`) are not recorded with the files; CWQ, GrailQA, MetaQA and KQA Pro rows reach 3 to 5 hops.
