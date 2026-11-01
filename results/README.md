# Results

The paper's reported numbers, one CSV per table and figure, and `make_tables.py`, which prints each table with the
paper's row and column labels (standard library only):

```
python results/make_tables.py               # every table and figure
python results/make_tables.py table1 fig7   # only those
```

`GAPS.md` lists every printed cell that differs from the submitted paper, with the count of matching cells per table.
All Hits@Any, F1 and LLM-cost cells match; the differences are graph-trace support values (see `GAPS.md`). No value
in these CSVs was edited to match the paper.

## Files

| file | paper | rows | values |
|---|---|---|---|
| `table1_standard.csv` | Table 1 | Qwen3-30B; WebQSP, CWQ, GrailQA, MetaQA, and `mean` | `hit`, `f1`; per set `llm_calls`, `input_tokens`, `output_tokens` |
| `table2_transfer.csv` | Table 2 | Qwen3-30B; 500 pairs per set, both halves | `hit`, `f1`, `support` |
| `table3_unseen.csv` | Table 3 | Qwen3-30B; 1,000 pairs per set, both halves | `hit`, `f1`, `support` |
| `table4_backbones.csv` | Table 4 | Qwen3-4B and gpt-oss-120B; the four sets, `mean`, and both halves of the Unseen pairs | `hit`, `f1`, `support` |
| `table5_stop_rules.csv` | Table 5 | KORA with each stop, Qwen3-30B and Qwen3-4B; the four sets, `mean`, unseen halves | `hit`, `f1` |
| `table7_unseen_validation.csv` | Table 7 | one row per check | `webqsp`, `cwq`, `basis` |
| `fig5_routes.csv` | Figure 5 | KORA, Qwen3-30B; one row per set and route | `count`, `n`, `pct`, `tokens` |
| `fig6_latency.csv` | Figure 6 | Qwen3-30B; one row per system and run (three runs of 200 twins, 100 per Unseen set) | `hit`, `tokens`, `llm_calls`, `latency_median_s`, `latency_p90_s` |
| `fig7_support.csv` | Figure 7 | Qwen3-30B; unseen halves of the 1,000 pairs | question counts per support-by-correctness segment, `support` |
| `paired_comparisons.csv` | Section 5 | the paired differences the text quotes | `difference`, `ci_lo`, `ci_hi`, `n` |

## Columns

- `backbone`: the LLM every training-free system and KORA called (Qwen3-30B-A3B-Instruct-2507, Qwen3-4B,
  gpt-oss-120B). GCR and KG-R1 run their own released models (Table 1's note).
- `dataset`: the harness's set name. `webqsp`, `cwq`, `grailqa`, `metaqa` are the standard test sets;
  `webqsp-unseen` / `cwq-unseen` are the unseen halves of the pairs (the twins), `webqsp-unseen-real` /
  `cwq-unseen-real` their source halves, both asked on the extended graph. `mean` is the mean over the four standard
  sets, taken over the unrounded per-set values, so it can differ by 0.1 from the mean of the printed values; it is
  given only for systems evaluated on all four.
- `system`: the paper's name. A stop other than the system's own is in brackets: `KORA` (the Stopping Estimator),
  `KORA (LLM stop)`, `KORA (gold stop)`, `ToG (estimator)`, `ToG (gold stop)`, and so on.
- `stop`: the harness's `--stop` for that row (`native`, `estimator`, `llm`, `gold`); empty for KB-BINDER, GCR and
  KG-R1, which take only their own stop (a KB-BINDER run folder records `native`; GCR's and KG-R1's converted folders
  an empty stop).
- `status`: `ok`, `not applicable` (KB-BINDER has no CWQ configuration, CoR does not support MetaQA, KG-R1 is not
  evaluated on GrailQA), `not run` or `incomplete` (KB-BINDER with gpt-oss-120B). Only `ok` rows carry numbers.
- `n`: questions (pairs for the Unseen halves), every one in the denominator: a failure, an empty answer or a
  timeout (200 s) is a miss.
- `hit`, `f1`: Hits@Any and set-level F1, in %.
- `support`: graph-trace support (Appendix B.3), in %, averaged over answered questions with logged graph calls;
  empty where a system makes no graph calls.
- `llm_calls`, `input_tokens`, `output_tokens`: LLM calls and tokens per question, failures included. Table 1's
  `In`, `Out` and `Calls` pool them over every question of the sets a system ran on (weighted by `n`); GCR and KG-R1
  log none (`note`).
- `note`: the gpt-oss-120B KORA rows use the estimator trained without renamed copies (Section 5.5).
- `table7_unseen_validation.csv` `basis`: `builder validation records` (the extension builder's validation outputs),
  `table3_unseen.csv (Direct LLM)` (the closed-book row is the direct LLM's Hits@Any on the two halves), or `paper text`
  (copied from the paper's Table 7; no result file for the check is in this release).
- `fig5_routes.csv`: `route` is how KORA settled the question (Figure 5's legend); `pct` is `count / n`; `tokens` is the
  mean LLM tokens (input and output) per question on that route.
- `fig6_latency.csv`: `tokens` is the mean LLM tokens per question; latencies are seconds, a timed-out question
  counted as 200 s.
- `fig7_support.csv`: `correct_supported` + `correct_not_fully_supported` + `wrong_supported` +
  `wrong_not_fully_supported` + `no_answer` = `n`; correct is Hits@Any, supported means every answer item passes the
  support rule.
- `paired_comparisons.csv`: A minus B over the same questions (or pairs), 95% paired bootstrap interval, 10,000
  resamples. The four standard sets' mean is resampled within each set.

## Reproducing a row

Each row of the Table 1–5 and Figure 6 CSVs is one run of the harness, scored:

```
python -m harness run --system kora --stop estimator --dataset cwq --model <served model id> --out runs/kora-cwq
python -m harness score runs/kora-cwq --csv scored.csv
python -m harness compare runs/kora-cwq runs/sog-cwq        # a paired_comparisons.csv row
```

The score CSV has the columns used here under the same names (`backbone`, `dataset`, `stop`, `n`, `hit`, `f1`,
`support`, `llm_calls`, `input_tokens`, `output_tokens`, `latency_median_s`, `latency_p90_s`); its `system` is the
harness id, mapped as follows.

| `system` here | `--system` | `--stop` |
|---|---|---|
| KORA / KORA (LLM stop) / KORA (gold stop) | `kora` | `estimator` / `llm` / `gold` |
| ToG, PoG, CoR / their (estimator) / (gold stop) rows | `tog`, `pog`, `cor` | `native` / `estimator` / `gold` |
| SoG | `sog` | `native` |
| Direct LLM | `direct_llm` | `native` |
| KB-BINDER | `kbbinder` | `native` |
| GCR, KG-R1 | not run by the harness: `baselines/gcr/`, `baselines/kg_r1/` | |

- GCR and KG-R1: their rows came from each folder's steps (`UPSTREAM.md`, "Steps"), scored with the same scorer:
  GCR exports each question with its subgraph, runs its released 8B (paths) and the backbone (answers), and converts
  the predictions to a run folder (`run_gcr.sh`); KG-R1 runs its released 3B against the authors' graph server on
  per-question subgraphs (`run_kgr1.sh`). Each `convert.py` writes `run.json` and `s0.jsonl`, which
  `python -m harness score` reads. KB-BINDER runs in the harness (`--system kbbinder`, `run_kbbinder_official` in
  `baselines/kbbinder/run.py`, called once per question as in the paper's harness) once its data and dependencies are
  installed. Each `UPSTREAM.md` lists the edits that let the adapters run from this repository.
- Tables 1, 4 and 5: `--dataset webqsp|cwq|grailqa|metaqa`. The MetaQA rows are over 515 questions, all of
  `data/questions/metaqa/test.json` (the paper's appendix states 517; see `GAPS.md`).
- Tables 3, 4 and 5 (Unseen columns): `--dataset webqsp-unseen` and `--dataset webqsp-unseen-real` (and the CWQ
  ones); `compare` on the two halves gives the source-minus-unseen gap over the pairs.
- Table 2 and Figure 6 run on subsets of the Unseen test pairs (`--ids`); the subsets' id lists are not in this
  release.
- Figures 5 and 7 split questions by KORA's route and by support and correctness; `score` reports a run's support as
  one number and does not produce these splits.
- Table 7 is the extension builder's validation (`unseen_builder/MANIFEST.md`).

## Gaps

`GAPS.md` lists every difference between these files and the submitted paper. In short: the paper's support values for
KORA (100.0) and CoR come from a later re-scoring of the runs' graph-call logs than the one these CSVs hold; Figure 7's
KORA and CoR segments follow from that; Table 4 omits ToG's support with gpt-oss-120B, which the CSV keeps.
