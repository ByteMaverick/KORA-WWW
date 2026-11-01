# Data

The question sets the paper evaluates on and the Unseen pairs. The builder of the extension the pairs are asked on is in
`unseen_builder/`.

```
questions/<set>/{train,val,test}.json   webqsp, cwq, grailqa, metaqa, kqapro
unseen/<set>/test.json          webqsp, cwq: 1,000 pairs each
unseen/<set>/validation.json    webqsp, cwq: 200 pairs each
```

## Question sets

Each file is `{"dataset", "split", "source", "graph", "questions": [record, ...]}`; the four `val.json` files of WebQSP,
CWQ, GrailQA and MetaQA carry `size` (the question count) instead of `graph`, and `grailqa/test.json` has no `graph`.
`source` is a note on how the file was drawn; no code reads it.

A WebQSP, CWQ, GrailQA or MetaQA record has `id`, `question`, `answers` (names), `topic_entity` (`{id: name}`),
`anchor_kinds` (`{topic id: its type}`) and a `gold` block: the gold query, its type, projection, answer kind, shape
(hops, operations, projection, answer class), hops, constraints and `masked` (the question with the topic entity's
mention, where it is found, replaced by a type placeholder such as `?person`). The sets add:

- WebQSP and CWQ: `answer_ids` (Freebase ids), `topic_entity_ids` and `in_dump` (whether the topic entity is in the
  Freebase dump); a CWQ `gold` also keeps `machine_question` and `webqsp_id`;
- GrailQA: `hops`, `level`, `function`, `s_expression`, `domains` and `relations`;
- MetaQA: `hops`.

A KQA Pro record has no `gold` block. It has `id`, `question`, `answers`, `answer_ids` (`KQAPro:` ids),
`topic_entity`, `hops`, `question_type`, `kg_source`, `shape` (the reasoning shape it was stratified by) and `program`
(the KoPL program); train and val records also keep `published_answer`, the answer as KQA Pro released it.

| set | split | file | questions | graph |
|---|---|---|---|---|
| WebQSP | train | `questions/webqsp/train.json` | 3,098 | Freebase |
| WebQSP | val | `questions/webqsp/val.json` | 297 | Freebase |
| WebQSP | test | `questions/webqsp/test.json` | 1,624 | Freebase |
| CWQ | train | `questions/cwq/train.json.gz` | 31,158 | Freebase |
| CWQ | val | `questions/cwq/val.json` | 287 | Freebase |
| CWQ | test | `questions/cwq/test.json` | 3,487 | Freebase |
| GrailQA | train | `questions/grailqa/train.json.gz` | 44,337 | Freebase |
| GrailQA | val | `questions/grailqa/val.json` | 271 | Freebase |
| GrailQA | test (scored) | `questions/grailqa/test.json` | 1,000 | Freebase |
| GrailQA | test (all) | `questions/grailqa/test_full.json` | 5,609 | Freebase |
| MetaQA | train | `questions/metaqa/train.json` | 2,000 | MetaQA's own movie graph |
| MetaQA | val | `questions/metaqa/val.json` | 300 | MetaQA's own movie graph |
| MetaQA | test (scored) | `questions/metaqa/test.json` | 515 | MetaQA's own movie graph |
| MetaQA | test (all) | `questions/metaqa/test_full.json` | 517 | MetaQA's own movie graph |
| KQA Pro | train | `questions/kqapro/train.json` | 1,000 | KQA Pro's own graph |
| KQA Pro | val | `questions/kqapro/val.json` | 200 | KQA Pro's own graph |
| KQA Pro | test | `questions/kqapro/test.json` | 282 | KQA Pro's own graph |

The two `.json.gz` files are gzipped because of their size. The harness reads the test files only (and the Unseen
pairs), never train; the extension builder, the estimator's sampler and its notebook read `train.json`, so unpack them
first: `gunzip -k data/questions/cwq/train.json.gz data/questions/grailqa/train.json.gz`. Train and val are the splits
the estimator's training data and the prompts' tuning drew on; the test files are read only for the reported numbers.
The estimator's own validation and holdout questions are the qids of `estimator/data/validation.parquet` and
`estimator/data/holdout.parquet`; its holdout questions are drawn from `train.json`, so data rebuilt from `train.json`
alone holds them and has to exclude them. KQA Pro is not run by the harness; its files are here for the estimator's
data.

WebQSP and CWQ are the questions of the public releases that name a linked topic entity. In the test files, 56 WebQSP
and 7 CWQ questions have no gold answer as released, and 3 WebQSP records have answers but no `gold` block; all are
kept. GrailQA's test is its public development set (the hidden test is leaderboard-only): `test_full.json` keeps the
5,609 questions that name a topic entity, and the scored `test.json` is a 1,000-question sample of them in their own
proportions of level, function and hops (seed 20261002). MetaQA's test is 517 questions of its 1/2/3-hop test set, drawn
a third per hop with templates in proportion within the hop (seed 20260928); `test_full.json` holds all 517, and the
scored `test.json` omits the two that also appear in the benchmark's training or validation split (`MetaQA-3hop-5282`,
`MetaQA-3hop-10264`), as in the paper's runs. (The `source` note of both MetaQA test files says 500 drawn; the files
hold 515 and 517.) MetaQA's train is a 2,000-question draw from its training file in the same way, and KQA Pro's test a
stratified draw by reasoning shape from its validation file. The val files of WebQSP, CWQ, GrailQA and MetaQA are small
draws from each set's train questions, and those questions are also in `train.json`; KQA Pro's val is disjoint from its
train.

## Unseen pairs

`unseen/<set>/<split>.json` is `{"pairs": [{"source": record, "twin": record}, ...]}`. A source is the original question,
unchanged. Its twin asks the same question shape about entities that were added to the graph (new ids `m.x...`, new
names, new literal values), so the gold queries differ only in the entity ids and constants. A twin record has the fields
of a source plus `twin_id` (its source's id), `clone_ids` (the added entities of the pair), `answer_class` (`closed` when
every answer is a class or a hub kept as it was, `open` otherwise), `flags` (`topic_hub`, `hub_kept_real`,
`constant_not_in_question`, ...), `text_entities_unresolved` and `provenance`. The `shape` inside `gold` is the query
shape the pairs were stratified by. In the test pairs, 6 WebQSP twins, 24 WebQSP sources and 1 CWQ source have no gold
answer; in the validation pairs, 4 WebQSP twins and 8 WebQSP sources.

The harness reads the twin half as `webqsp-unseen` / `cwq-unseen` and the source half, in the same order, as
`webqsp-unseen-real` / `cwq-unseen-real`; both are asked on the extended graph (the validation split as `--split
validation`). `python -m harness run --dataset webqsp-unseen --split test ...` is the entry point.

## Serving the graphs

**Freebase.** The sets above were built on the Freebase RDF dump (graph `http://freebase.com`) loaded into a Virtuoso
Open Source instance. Load the dump with Virtuoso's bulk loader (`ld_dir` / `rdf_loader_run` in `isql`), list the
directory it sits in under `DirsAllowed` in `virtuoso.ini`, and serve it; the SPARQL endpoint is then
`http://<host>:8890/sparql` (set `FREEBASE_SPARQL_URL`, or pass `--sparql-endpoint`). The harness only reads.

**The extension.** The Unseen twins are asked on a copy of that Freebase database with the pairs' added entities loaded
beside it. Never load into the database the real runs read: stop Virtuoso, copy its database directory to a new one,
start a second instance on it, and load the extension there. `unseen_builder/` describes what the extension contains and
how it was checked; `unseen_builder/README.md` is the builder (`python -m unseen_builder.main build`) and the loader
(`python -m unseen_builder.main load`, which copies the `clones.nt` files into the copy's `load/` directory and
bulk-loads them). The inserted triples (`clones.nt`) are not part of this repository: they are released with the
camera-ready version. Without them the twin half cannot be answered from the graph, and only the source half
(`*-unseen-real`) and the four standard sets run.

**MetaQA.** MetaQA's movie graph is served from Neo4j 5 Community (the paper's runs used 5.26) at `NEO4J_METAQA_URI`.
KORA, the baselines and the estimator's sampler expect:

- one node label, `Entity`, with properties `id` and `name`, both the entity's text (MetaQA has no identifiers), and an
  index on `Entity(id)`;
- nine relationship types, the relations of `kb.txt` upper-cased: `DIRECTED_BY`, `WRITTEN_BY`, `STARRED_ACTORS`,
  `RELEASE_YEAR`, `IN_LANGUAGE`, `HAS_TAGS`, `HAS_GENRE`, `HAS_IMDB_VOTES`, `HAS_IMDB_RATING`, each from the movie to
  its attribute, with no properties (KORA reads them from `kora/prompts/prompts.yaml`, section `kg.metaqa_notes`);
- the graph in the server's default database, `neo4j`, with `NEO4J_METAQA_DATABASE=neo4j` (as in `.env.example`). The
  baselines' client reads that variable (`metaqa-db` when unset); KORA asks for a database `metaqa` and, when the
  server has none, reads the default one.

To load it, take `kb/kb.txt` from the public MetaQA release and run, from the repository root:

```
python data/metaqa_to_csv.py PATH/TO/kb.txt metaqa/csv      # 43234 nodes, 134741 relationships
docker run --rm -v "$PWD/metaqa/neo4j:/data" -v "$PWD/metaqa/csv:/import:ro" neo4j:5-community \
    neo4j-admin database import full --nodes=Entity=/import/entity_nodes.csv \
    --relationships=/import/kb_rels.csv --overwrite-destination neo4j
docker run -d --name metaqa -p 7687:7687 -v "$PWD/metaqa/neo4j:/data" \
    -e NEO4J_AUTH=neo4j/PASSWORD neo4j:5-community
docker exec metaqa cypher-shell -u neo4j -p PASSWORD \
    "CREATE INDEX entity_id IF NOT EXISTS FOR (n:Entity) ON (n.id);"
```

then set `NEO4J_METAQA_URI=bolt://localhost:7687` and `NEO4J_PASSWORD=PASSWORD`. The harness only reads.
