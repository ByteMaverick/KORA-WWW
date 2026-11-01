# The extension: what it contains and how it was validated

The extension adds, beside the real Freebase graph, a clone of every entity on each question's gold path, under new ids
and new names. A clone has its source's shape: the same relations in the same counts, literals moved by one
order-preserving transform per question, and fresh real entities of the required type as the objects of padded edges.
No clone is ever the target of an edge from the real graph, and no real entity gains an edge.

| | WebQSP-Unseen | CWQ-Unseen |
|---|---|---|
| released pairs, test | 1,000 | 1,000 |
| released pairs, validation | 200 | 200 |
| source split of the test pairs | the set's test questions | the set's test questions |
| source split of the validation pairs | the set's training questions | the set's training questions |

The released pairs are drawn from the validated twins, stratified by query shape (hops, operations and projection); where
a stratum had too few valid pairs, the shortfall was drawn from other strata (8 WebQSP and 34 CWQ pairs). The clones'
triples are not in this repository (see below); the pair files in `data/unseen/` carry every pair's `clone_ids`.

Naming: persons and places are named from recombined real name parts, with no model; every other entity is named by a
held-out Qwen3-14B checkpoint that is not one of the evaluated models and never sees the source name. `--names-model` and
`--names-url` select the model when rebuilding (any OpenAI-compatible server).

## Validation report

Checks run on the served extended graph with every pair loaded (builder `checks.py`; the table is the paper's Table 7).

| check | WebQSP | CWQ |
|---|---|---|
| source query does not reproduce the recorded answers (dropped) | 45 | 337 |
| source answers unchanged on the extended graph | 1,200 / 1,200 | 1,200 / 1,200 |
| twin query returns exactly the planted answers | 1,200 / 1,200 | 1,200 / 1,200 |
| inserted triples with a real subject | 0 | 0 |
| clone-to-clone edges between pairs | 0 | 0 |
| excluded by closed-book and source-identification screening | 42 of 1,631 | 24 of 2,408 |
| screened-out twins released | 0 | 0 |
| shared twins, sources, texts or clones between validation and test | 0 | 0 |
| closed-book accuracy, source to twin | 53.9 to 4.2 | 34.3 to 3.1 |

The 1,200 are the 1,000 test and 200 validation pairs of the set. Closed-book accuracy is a model answering the question
with no graph; the twin's figure is what remains of parametric knowledge once the entities are new.

## What is not here

The inserted triples (`clones.nt`, one file per set and split), the builds' mapping files and the naming logs are released
with the camera-ready version. The builder in this directory regenerates them from a Freebase endpoint, but the names
of the entities it needs a model for are not reproducible by re-running (the naming model samples at temperature 0.8):
the released names are the ones in the pair files.

The builder covers the Freebase sets only. The pair files load through the harness loader as `webqsp-unseen`,
`webqsp-unseen-real`, `cwq-unseen` and `cwq-unseen-real`.
