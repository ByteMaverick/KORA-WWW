# The Unseen builder

Builds an Unseen twin for each WebQSP or CWQ question: the question asked again about new entities that exist only in an
extended copy of Freebase. The entities on the question's gold path are cloned beside the real graph under new ids and
new names, so the twin has the same shape, the same relations and the same kind of answer as its source. The source and
its twin form a pair; both halves run on the same extended graph, and the drop from the source half to the twin half is
the knowledge-advantage gap. `MANIFEST.md` lists what the released extension contains and its validation report.

## The flow

```
 a benchmark question with its gold SPARQL
        │
        ▼
 ① EXPOSE + EXECUTE     every variable projected, value filters removed, run on the real Freebase (read only);
        │               the gold must reproduce the recorded answers, or the question is dropped
        ▼
 ② CLONE                every node on the gold path but the classes; the topic and every mediator always;
        │               a clone may point at a real node, a real node never points at a clone
        ▼
 ③ MOVE LITERALS        one order-preserving transform per question: dates shift by whole years, numbers scale
        ▼
 ④ PAD                  each clone gets its source's outgoing relations in the same counts; hubs capped at 2,000
        ▼
 ⑤ NAME                 people and places from recombined real name parts; other kinds by a model that never sees
        │               the source's name; every name checked against every real label and alias
        ▼
 ⑥ REWRITE              the question with each mention swapped (no model writes text), the gold with ids swapped
        ▼
 write clones.nt, load it into the copy
        ▼
 ⑦ VALIDATE             the suite of eight checks on the extended copy; a twin it rules out leaves every file
        ▼
 ⑧ RELEASE              every split loaded on a fresh copy and checked again → held-out filter → trim (stratified
                        by shape) → the released files on another fresh copy: they ship only if every real answer
                        is unchanged and every planted gold executes → the pair files
```

`main.py` runs this flow and reads in this order: `Build.question` is steps ①–⑥ for one question, `_run` takes the
questions through it, then `build_main` writes the files, loads and validates them; the release steps are in
`steps.py`, and `main` runs any step.

| file | holds |
|---|---|
| `main.py` | the flow: `Build` and `_run` (①–⑥ per question), the `build` step, the `main` dispatch of every step |
| `steps.py` | the other steps: `merge`, `heldout`, `pollution`, `trim`, `pairs`, `load` |
| `twin.py` | one question into one twin: ① the gold exposed, ② the clone, ③ the literals moved, ④ the padding, ⑥ the rewrite |
| `names.py` | ⑤ the name pools and the naming model |
| `checks.py` | the build's load and suite run, ⑦ the eight checks, ⑧ the held-out filter, the pollution check, the trim |
| `graph.py` | the real Freebase read (SELECTs only), its hub and name lookups and their per-run cache (`_cached`), the benchmark nodes a twin must avoid (`benchmark_nodes`: every topic and answer id of WebQSP's and CWQ's train, val and test files), the N-Triples writer, the bulk load into a copy |
| `prompts.py` | the five prompts the builder sends to the naming and checking models |

## ① Expose and execute the gold

The gold query is parsed (a query that does not parse is dropped) and rewritten to project every variable, so its rows
return every node the query touches: the topic, its constants, intermediates, mediators (CVTs) and every candidate
answer. Its value filters are removed, so the candidates a filter would discard come back too, with the values they
were compared on; a filter's `EXISTS` group becomes an `OPTIONAL`, and each `UNION` branch is marked so that a row
instantiates only its own branch's triples. A filter that tests only languages or identities (`?x != ?c`) is kept.

The gold must first return the question's recorded answers on the real graph, or the question is dropped ("gold differs
on the real graph"). A path of more than 200 nodes, or a reply past the server's row cap, drops it ("path too large").
The dataset's `xsd:datetime(` is sent as `xsd:dateTime(`.

## ② What is cloned: the direction rule

- Only classes stay real: a `type.type` node, or the object of `type.object.type` / `common.topic.notable_types`.
  Everything else on the path is cloned, countries, languages and units included when the gold binds them.
- The topic is always cloned, and so is every mediator.
- A constant the question names (its label or an alias found in the text) is cloned and its mention swapped. A constant
  it does not name (the node "Marriage" behind "married") is a schema constraint: it stays real and the twin is flagged
  `constant_not_in_question`.
- The direction rule: a clone may point at a real node; a real node never points at a clone. A kept-real node that
  would point at a clone is cloned too; if it is a class, or a constant the text does not name, the question is dropped
  ("a class points at a clone", "real constant would point at a clone").
- A hub (2,000 or more edges, housekeeping relations not counted) the path only points at ("actors from France") stays
  real and flags the twin `hub_kept_real`; a hub the path leaves toward a clone is cloned, its padding capped, and flags
  `hub_cloned`.
- New ids are `m.x` and a base-36 counter (`m.x1`, `m.x2`, ...); real ids start `m.0` or `g.1`, so they never collide.
  The counter continues across the builds on one copy (`<replica-root>/load/next_id.txt`).

Every topic must be on the path and mentioned in the text, or the question is dropped ("topic not on path", "topic
mention not found").

## ③ Move the literals

One monotone transform per question: every date shifts by a whole number of years δ in ±1..40 (month, day and precision
kept), every number scales by one s in [0.5, 2.0] (decimals kept, integers rounded half up). Of up to 200 draws, the
first is kept under which every moved value lies in its property's range on the graph, no 29 February lands in a common
year, and every pairwise order among the moved values and the query's constants holds. So a filter keeps every
candidate on its side, a superlative keeps its winner, a count is unchanged, and a birth stays before a death. A
literal moves only when every subject it sits on is cloned. The query's constants move with it, and so do their
surfaces in the question text (a date's year, a number's text, an integer's comma-grouped form). Strings never move.

Dropped, never repaired: a superlative whose values tie at the cut, a literal not of its property's kind, a constant
not of its variable's kind, no transform in range, one surface that two constants would map to different texts, and a
moved constant the question shows in a form no surface maps ("1.2 million").

## ④ Pad in the source's shape

Each clone gets its source's outgoing relations in the real graph, in the same counts, the path's own relations
included. An object is filled with another clone (when the source pointed at a node that is cloned), a clone mediator
(padded one level deep, at most 30 per relation per clone), a moved literal, or a fresh real node of the relation's
target type. The target type is read from data: the commonest specific type among up to 200 of the relation's objects.

Fresh targets come from a 500-node sample that is even across the type (one count, then a hash bucket with a fixed
salt, then a seeded draw), never the server's first rows, which are the best connected. A fresh target is never the
topic or an answer of any real WebQSP or CWQ question, never one of the source's own objects on the relation, and never
a node the path reads. Incoming edges and housekeeping relations are not padded; a topic with more than 2,000 out-edges
is padded to 2,000 (`topic_hub`). Everything padding cannot fill is counted per clone, relation and reason in
`mapping.json`'s `unpadded` record (hub cap, mediator cap and depth, string, no range, out of range, duplicate,
self-loop, no target).

## ⑤ Names

A name must read like a real name of its kind, exist nowhere in the graph, and never be derived from the source's
name: the source's name is never shown to a model.

- People are `<First> <Last>` and places one or two place parts, recombined from token pools read off the graph's own
  English labels of `people.person` and `location.location` (`build --pools-only`).
- Every other kind asks the model (`prompts.NAME`) for five names of "a new, fictional {kind}", seeded with its kept
  attributes; up to five batches per clone, then the question is dropped ("naming failed"). A symptom-like kind is
  named by a clinical phrase (`prompts.NAME_PHRASE`).
- A clone whose source label contains another clone's source label is named after it ("Vancouver Canucks" ->
  "Metronis Canucks").
- A kind whose labels carry a year or an edition number ("2009 World Series") is named in that form: the clone's own
  moved year around a series name the model writes (`prompts.NAME_SERIES`).
- Every candidate is refused when it is a placeholder, a label or alias of any node on the real graph or the extended
  copy, a name already handed out (shared across concurrent shards with `--names-shared`), or a fourth name on a
  five-letter stem already used three times for its kind.

The paper's twins used a held-out Qwen3-14B checkpoint for the model-named kinds; `--names-model` and `--names-url`
select the model (any OpenAI-compatible server). Every candidate, rejection, prompt and reply is logged.

## ⑥ The twin question and its gold

Each cloned constant's mentions are found by its labels and aliases, case-insensitive, at word boundaries, longest first
("Jamaica's" holds Jamaica, "Jamaican" does not). One pass writes each clone's name over its spans and each moved
literal surface over its occurrences; a possessive keeps its form, and an all-lowercase question takes its names
lowercased. The twin's gold is the source's query with the ids and moved constants swapped; its answers are the
source's answers mapped through the clones (a literal answer through the transform). A twin's id is its source's id
plus `_unseen`. Each twin carries its `shape`, `clone_ids`, `answer_class` and flags (`constant_not_in_question`,
`hub_kept_real`, `hub_cloned`, `topic_hub`, `unresolved_text_entities`, `mention_repeated`,
`literal_surface_repeated`).

## Write and load

The clones go to `clones.nt`: each clone's name, its source's types and notable type, every edge among clones and to
kept-real nodes, and every moved literal with its property's datatype (a name literal of the source is never copied).
The file is bulk-loaded into the extended copy (`ld_dir` / `rdf_loader_run` through `isql`), one load at a time per
copy under a file lock.

The real graph is never written: the build sends only SELECTs to `--real`. The load refuses unless the server at
`--isql-host:--isql-port` reports an ini file inside `--replica-root`, refuses a directory with no `virtuoso.db`, and
refuses the paths in `$FREEBASE_DB_SRC` and `$FREEBASE_LIVE_REPLICA` (set them to the database the real runs read).

## ⑦ Validate

The suite runs after every build, on the extended copy, and writes `report.json`. Each check passes, fails or is
skipped; a check made of sub-checks passes when every one that ran passed.

| # | check | rules a twin out? |
|---|---|---|
| 1 | real questions unchanged: every source's gold returns its recorded answers on the extended copy | yes: the twins whose clones changed it |
| 2 | planted gold executes: every twin's gold returns exactly its planted answers; every clone has an out-edge | yes |
| 3 | names unknown: no new name is a real label or alias; advisory, the attribute test on a sample | no |
| 4 | not a renamed source: a judge shown a clone's facts never names its source | no |
| 5 | readable: `human_read.jsonl`, up to 100 twins beside their sources, for a human read | no |
| 6 | structure: shape, hop, constraint and answer-size histograms equal the sources'; each clone's out-degree per relation equals its source's less `unpadded`; no insert line has a real subject; literals in range; births before deaths | no |
| 7–8 | prior empty, difficulty matched: measured by the harness on the evaluated models | no |

A twin check 1 or 2 rules out leaves every file, counted as "polluted real question", "planted gold mismatch", or
"padding satisfied the gold" when a padded node or literal is among the wrong answers. Samples are seeded, so a report
reproduces. `build --suite-only` re-runs the suite on an earlier stage's files.

## ⑧ Release

One shard's suite cannot see pollution arriving through another shard's clones, so the release is checked in rounds.

1. Shards are joined with `merge` in source order; a later shard's twin with a clone named as an earlier one's is
   dropped ("duplicate name across shards"), and so is a twin a shard's suite found changing another shard's source.
2. Round 2: every merged split's `clones.nt` loaded on a fresh copy (`load`), one `build --suite-only` per split;
   `pollution` maps a change made by another split's or set's clone to the twin that owns it.
3. The held-out filter (`heldout`): a model outside the evaluated set gets every twin twice. Closed book, the twin fails
   when the model names a gold answer with no graph; shown a topic clone's facts, it fails when the model names the real
   source.
4. The trim (`trim`): the test split is `--test` twins stratified by shape (hops, operations and projection), each
   stratum its share of the source test split by largest remainder, a shortfall given to strata with twins to spare;
   val is the first `--val` twins of the val wave (`--val-from`), else the train twins after the released train.
   Excluded and surplus twins stay in `raw.jsonl` with their reason.
5. Round 3: the released files loaded on another fresh copy and checked again; `pollution --gate` exits 1 unless every
   report passes checks 1 and 2.
6. `pairs` writes each released split as the harness's pair file, `data/unseen/<set>/{test,validation}.json`:
   `{"pairs": [{"source": record, "twin": record}]}`, the source records unchanged, in the twins' order.

## Run

Every step is `python -m unseen_builder.main STEP [flags]`; `STEP --help` lists its flags.

```
python -m unseen_builder.main build --pools-only --real http://<host>:8890/sparql
python -m unseen_builder.main build --set webqsp --split test --limit 100 \
    --real http://<host>:8890/sparql --ext http://<host2>:8890/sparql \
    --isql-host <host2> --isql-port 1111 --replica-root <copy of the Freebase database> \
    --out <dir> --names-model <model> --names-url <chat completions URL> --judge-url <server>/v1
```

| flag of `build` | meaning |
|---|---|
| `--set webqsp\|cwq`, `--split test\|val\|train` | the source split, `<--splits>/<set>/<split>.json` (default `data/questions`, which ships train, val and test; the released test pairs were built from `test`, the validation pairs from `train`). Every build also reads the WebQSP and CWQ train, val and test files, and `pairs --split val` reads `train.json` and `val.json`, so first unpack the gzipped train file: `gunzip -k data/questions/cwq/train.json.gz` |
| `--offset N`, `--limit N` | skip the first N questions whose gold parses, then take the next N (a later wave of a split) |
| `--real URL` | the real Freebase SPARQL endpoint: only read |
| `--ext URL`, `--isql-host`, `--isql-port`, `--replica-root DIR` | the extended copy: queried, and loaded through `isql` (the binary in `ISQL_BIN`; unset: `isql` on the `PATH`) |
| `--out DIR` | the build's files |
| `--names-model`, `--names-url`, `--names-extra`, `--names-shared FILE` | the naming model (any OpenAI-compatible chat completions server; key in `OPENAI_API_KEY` when it needs one); `--names-extra` merges JSON into each request; `--names-shared` is a file of names handed out, shared by concurrent shards |
| `--judge-url URL`, `--judge-model` | an OpenAI-compatible server for the model checks (attribute test, guess the source); `--no-model-checks` skips them |
| `--seed` | the run's seed, plus the CRC-32 of `<set>/<split>` |
| `--pools DIR`, `--pools-only` | the name pools; `--pools-only` builds them from `--real` and exits: run it once before any shard |
| `--shard k/N`, `--id-start N`, `--no-next-id` | build every N-th question from the k-th, with its own range of clone ids |
| `--dry-run` | everything up to the load, files under `<out>/preview/` |
| `--suite-only --splits-in DIR [--build-in DIR]` | re-run the suite on an earlier stage's files, writing to `--out` |

The other steps:

```
python -m unseen_builder.main merge --set webqsp --split test --shards <dir1> <dir2> ... --out <merged>/test \
    [--replica-root DIR]
python -m unseen_builder.main load --replica-root DIR --isql-host H --isql-port P <clones.nt> ...
python -m unseen_builder.main pollution --checked <dir> --out <dir>          # and --gate <dir>
python -m unseen_builder.main heldout --raw <release>/raw.jsonl --mapping <merged>/<split>/mapping.json ... \
    --ext URL --model M --url URL --out <dir>
python -m unseen_builder.main trim --set webqsp --merged <dir> --out <release> [--val-from DIR] \
    [--exclude heldout_excluded.txt] [--polluted polluted.txt] [--test 1000 --val 200 --train 2000]
python -m unseen_builder.main pairs --set webqsp --split test --twins <release>/test [--ids released.txt]
```

`load` is also how a user rebuilds the extended graph from released insert files: copy the Freebase database, serve
the copy, load every `clones.nt` into it, and validate with `build --suite-only` per set and split.

## Outputs

A build writes, in `--out`: `<split>.json` (the twins), `<split>_meta.jsonl` (one line per source question read, with
its drop reason), `drops.json` (every drop by reason, with the question ids), `clones.nt` (the inserts),
`mapping.json` (source -> clones, clone -> name, type, attributes, labels, unpadded edges, literals, and the build's
record), `names.log.jsonl`, `report.json` and `human_read.jsonl` (the suite), `errors.log`, and `build.json` (last).
No stage rewrites another stage's files.

The released pairs record `data/extension` as their generator (the builder's previous location); it is kept so the
output stays identical.
