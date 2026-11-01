# The harness

One runner and one scorer for every system: KORA and the baselines run on the same question sets, with the same
model client, the same per-question limit and the same stop switch, and are scored by the same rules.

```
python -m harness run --system kora --stop estimator --dataset webqsp --out runs/kora-webqsp
python -m harness score runs/kora-webqsp runs/sog-webqsp --csv results.csv
python -m harness compare runs/kora-webqsp runs/sog-webqsp
```

Settings come from the shell and from `.env` in the working directory (the shell wins); see `.env.example`.

## Files

| file | holds |
|---|---|
| `cli.py` | the command line: `run`, `score`, `compare` |
| `runner.py` | the run loop: every question of a set through one system, one row per question; `QueryMetrics`, a system's per-question totals of its LLM calls |
| `dataset.py` | the question sets: where each file is, its questions as `QuestionSample`s, and the gold-locked sample every system is handed |
| `settings.py` | the settings table: models, systems, decoding, the per-question limit and its timeout |
| `llm.py` | the model client every system shares, its per-call metrics and call trace |
| `graph_meter.py` | counts and logs every graph call |
| `neo4j_client.py` | MetaQA over Neo4j (also in the shape ToG's and PoG's code reads), and the graph-error rule |
| `manifest.py` | what produced a row (code, authors' code, settings, split), stored in GCR's and KG-R1's rows |
| `subgraph_topics.py` | the topic entities as a per-question subgraph keyed by name knows them (GCR's exports) |
| `stop_oracle.py` | the gold stop of the `--stop gold` arms |
| `verdict_log.py` | every stop decision, one JSON line per judged candidate |
| `systems/` | the systems and their handlers (`kora.py`, `direct_llm.py`; the baselines live in `baselines/`) |
| `metrics/` | scoring the runs, in the paper's four metric parts (see Scoring) |
| `metrics/groundedness/` | graph-trace support: `records.py` (the inputs; also the ids and the name resolution), `hook.py` (calls records from the graph log), `support.py` (the support rule), `grounding.py` (a run's support, on the same answer sets as correctness) |
| `metrics/reasoning.py` | correctness: normalization, matching, Hits@Any and F1, the text scorer, the reading rule, the statistics, the reader of a run's shard files |
| `metrics/efficiency.py` | cost: LLM calls, tokens, graph calls and latency per question |
| `metrics/unseen.py` | the Unseen gap: source minus twin on the paired questions |
| `metrics/common/` | what the four share: `answers.py` (answer sets), `extract.py` and `extract_prompt.md` (the free-text extractor), `cells.py` (a run's rows), `report.py` (`score` and `compare`) |

## The run folder

One run is one folder (`--out`): `s<N>.jsonl` (the rows of shard N), `traces/s<N>/` (each question's LLM calls,
graph calls and stop decisions) and `run.json` (what ran). Only the live shard files `sN.jsonl` are read, in shard
order; files set aside beside them (`sN.failed.jsonl`, ...) are never read, and a question with two live rows is an
error (`DuplicateRowError`), not "the last file wins". When the scorer reads a run (`metrics/common/cells.py`), every
question of the set gets one row slot, and a question without a row is missing; a row for a question outside the set
is counted, never scored, and a line of a shard file that does not parse is skipped and counted (`unparsed row lines`).
`--ids` filters after sharding, so a question keeps the shard it has in the full set.

## Question sets

```
data/questions/<set>/test.json     WebQSP, CWQ, GrailQA, MetaQA: {"questions": [record, ...]}
data/unseen/<set>/<split>.json     WebQSP-Unseen, CWQ-Unseen (test, validation): {"pairs": [{"source": record, "twin": record}, ...]}
```

`<set>-unseen` is the twin half of the pairs (questions about entities added to the graph), `<set>-unseen-real` the
source half (the original questions), both in the pairs' order. A record has `id`, `question`, `answers` (names),
`answer_ids` and `topic_entity` ({id: name}); a twin also has `twin_id`, its source's id. A sample does not carry the
gold ids; the scorer reads them from the question file.

## Systems

| system | `--stop` | framework id | code |
|---|---|---|---|
| `kora` | estimator (default), llm, gold | `kora` | `harness/systems/kora.py` |
| `tog` | native (default), estimator, gold | `tog_<stop>` | `baselines/tog/run.py`, `run_tog` |
| `pog` | native (default), estimator, gold | `pog_<stop>` | `baselines/pog/run.py`, `run_pog` |
| `cor` | native (default), estimator, gold | `cor_<stop>` | `baselines/cor/run.py`, `run_cor` |
| `sog` | native | `sog_native` | `baselines/sog/run.py`, `run_sog` |
| `kbbinder` | native | `kbbinder_official` | `baselines/kbbinder/run.py`, `run_kbbinder_official` |
| `direct_llm` | native | `direct_llm` | `harness/systems/direct_llm.py` |

A handler is `handler(question, args, sample=...) -> (answer, fw_metrics)`; the answer is a string, or a list of
names (scored as its items, and joined with "; " in the row's text). A baseline's package is imported only when it
runs. A system that reports an error and returns no answer has failed the question: the row is an error row.

## The gold never reaches a system

Every system is handed `blind(sample)`: the sample with its gold fields (`answers`, `answer_ids`, `sparql`, `raw`)
locked. Reading or writing one raises `GoldAccessError`, which turns the question into an error row instead of a
number. The harness scores with the real sample, after the system returns. The gold-stop arms get the gold only as
an explicit callback, `args.stop_oracle_callback` (a `StopOracle`), set for those arms alone.

## The stop switch

`--stop` chooses who decides that the search has found enough: `native` (the system's own rule), `estimator` (the
learned judge, a server at `ESTIMATOR_URL`), `llm` (KORA's model judges) or `gold`.

Under `--stop gold` a system's search, prompts, judge and answer function stay as they are. At each point where it
asks its own judge "is this enough?", the gold arm also has it write the answer it would give at that moment, with
the answer function it uses when it stops, and the stop oracle says whether that answer is good enough to stop on.
The system stops on the oracle's word instead of its judge's; the judge is still asked and its verdict logged. The
gold never reaches the model. The rule is KORA's oracle, the same for every system:

* the answer's items, found as each system writes them: a {...} answer, a JSON list, a "Final answer:" list, a
  short comma list; an answer in sentences counts the graph names it mentions (the names in the system's own path),
  else it is one item;
* recall = the share of gold answers some item matches; precision = the share of the items (the first 50) that match
  some gold answer; accept when recall >= 0.5 and precision >= 0.5;
* a count question (the gold is one whole number and the question asks how many) is accepted when the first number
  in the answer is that number;
* names match when the words of one appear, in order and adjacent, in the other's; a gold that is only a number must
  lead the name (`9` is not `Apollo 9`); aliases are not resolved.

A plain hit (any gold named in the text) would accept a 30-name list with one right name, or "it is not X", which is
why the rule asks for precision too. Every step is logged in full: the answer, its items, the path the system had,
precision, recall, F1, accept, and the plain hit for reference.

## The verdicts file

Every stop decision of a stop arm, accepts and rejects, at every decision, is one JSON line in
`<HARNESS_TRACE_DIR>/<qid>.verdicts.jsonl`, beside the question's call trace. The stop tables (a system's own judge,
the learned judge, the gold over the same candidates) are computed from these after the runs. A record:

| field | value |
|---|---|
| `qid` | the question |
| `seq` | 0, 1, 2 ... per question file, in write order; a retry of the question appends after the first try (`pid` tells the tries apart) |
| `ts`, `pid` | when and by which process it was written |
| `arm` | the arm (`tog_estimator`, `pog_gold`, `cor_estimator`, `kora_estimator`, ...) |
| `stop` | `judge` (a judge decided: the learned judge, or a system's own) or `gold` |
| `judge_url` | the judge server asked; null for the gold, and for a judge that is not a server |
| `threshold` | the cut the arm applied to `score`; null for the gold |
| `depth` | the arm's depth (ToG's and PoG's loop depth, CoR's chain length, KORA's task depth), when it has one |
| `step` | the decision's number within the question, when the arm counts them |
| `index`, `n_candidates` | the candidate's place among those judged at this decision, and how many there were (0 and a null candidate: nothing to judge; the arm went on) |
| `candidate` | what was judged: `answer` (the text the gold scores), `names`, `ids` (each capped), `n` (how many in all), `query`, and the arm's own fields |
| `score` | the judge's p for this candidate; null for the gold, or when the judge gave none (KORA: read from its reason text, two decimals, the root only) |
| `accept` | true for the candidate the arm stopped on, false for every other |
| `refused` | the judge refused it (the server scores a state it refuses exactly 0.0) or could not be reached; null for the gold, and where the arm cannot tell |
| `gold_hit` | for the gold: its verdict on this candidate, with its figures in `gold` (precision, recall, F1 and the plain hit); null for a judge |
| `error` | why there is no score (the last error after the arm's retries), else null |

Arms add their own fields (CoR: `own_verdict`, its own judge's action at the same point; KORA: `level`, `judge`,
`source`). A write that fails never raises and never changes a decision; without `HARNESS_TRACE_DIR` nothing is
written. Writes are serialised by one lock, so concurrent decisions never interleave a line or repeat a seq.

## KORA under the harness

KORA runs as this repository's `kora` package, unchanged:

* every question runs through KORA's own per-question routine (`Harness.run` in `kora/run.py`): its recogniser,
  planner, controller, query agent, judges, decomposition and answer, with KORA's own scorer beside the harness's
  (its row is kept, trimmed, in `fw_metrics["kora"]`);
* KORA's model calls go through the harness's model client, like every baseline's: the same servers, retries, call
  traces, token counts and reasoning effort; each call is labelled with the KORA prompt that made it;
* KORA's graph access is its own (Freebase over SPARQL, Neo4j for MetaQA), pointed at the endpoint the harness picked;
  the Unseen sets run on the extended graph and read their original set's graph shape;
* the topic entities are the sample's, given to KORA as the root's entities, as every baseline gets them;
* the gold reaches KORA only under `--stop gold`, through the stop callback; every other stop runs with an empty
  gold, so its answer is KORA's own projection and the gold-scored fields of its row are empty;
* KORA's budget is 140 s, inside the harness's 200 s limit: one step in flight (a model call, often 30-60 s on
  gpt-oss-120B, and the queries it writes) runs past KORA's clock, and 140 s leaves it room to finish. KORA gets less
  time than the baselines, never more.

`--stop` sets KORA's `KORA_STOP` (`STOP_MODES` in `kora/app.py`):

| `--stop` | who decides | clock fallback (`KORA_CLOCK_FALLBACK`) |
|---|---|---|
| `estimator` | the Stopping Estimator at leaf and root | `best` |
| `llm` | the model judges (the knowledge judge at the root) | `held` |
| `gold` | the gold decides the root (recall and precision >= 0.5, `KORA_ORACLE=strict`) | `held` |

A `KORA_*` setting in the environment that the stop rule does not set, or a fixed one set to another value, is
refused, never silently overridden. There is no answer-without-the-graph step in KORA: its answer is written in code
from what the graph returned.

KORA's verdicts. KORA judges inside its own code, so its verdict lines are written after its run, from its row:

* the root's: the controller's log has one line per judgement: an offer to the root (`offer T2: 4 -> root, judged
  yes -- <reason>`) and a composed parent (`composed T1: and -> 3, judged no <reason>`). The estimator's reason
  carries its p to two decimals (`estimator p(sufficient)=0.62`), the knowledge judge's its levels score; no
  candidate names or ids;
* the leaf's: every attempt's executed candidates with the judge's outcome: `SUCCESS` (accepted, selected),
  unselected (accepted, another selected), `INSUFFICIENT_RESULT` (rejected, by the judge or by a rule in code: the row
  does not say which); no score. `EMPTY_RESULT`, `EXECUTION_ERROR` and rejected or unjudged candidates are not
  decisions.

Under `KORA_ROOT_JUDGE=oracle` the gold judges the root task (its offers, its composition, its own candidates) and the
model every other task.

## The graph-call log

`graph_meter.py`, installed once by the run loop, counts every graph query a system makes, whichever client it
uses, so each row can say how many graph queries a question cost beside its LLM calls and tokens. It wraps, without
changing what they return or raise: `requests`' `Session.request` for URLs that look like a SPARQL endpoint,
`urllib.request.urlopen` for SPARQL URLs (KORA's Freebase client; a module that bound `urlopen` to its own name before
install keeps the unwrapped one), `SPARQLWrapper.query`, and neo4j's `Session.run` (MetaQA). Each call is counted and
logged once: only the outermost wrapper records (a per-thread depth: `SPARQLWrapper.query` calls `urlopen` itself).

With `HARNESS_TRACE_DIR` set and `HARNESS_GRAPH_LOG` not `0`, each call appends one JSON line to
`<HARNESS_TRACE_DIR>/<qid>.graph.jsonl`, beside the question's LLM trace:

| field | value |
|---|---|
| `qid` | the question (`""` outside a question: the file is `no-qid.graph.jsonl`) |
| `seq` | 1, 2, ... per file, in write order; a question retried by a later process continues its file's numbering |
| `ts` | when the call started |
| `client` | `requests`, `urllib`, `sparqlwrapper` or `neo4j` |
| `endpoint` | the URL without its query string (neo4j: the server's host:port, or null) |
| `method` | the HTTP method (neo4j: `RUN`) |
| `query` | the query as sent: the SPARQL in the request's params, data or json, else the URL's `query=` parameter (SPARQLWrapper: its query string); neo4j: `{"text": the Cypher, "parameters": {...}}` |
| `status` | the HTTP status, or null (neo4j; a call that failed before a response) |
| `seconds` | the call's duration, reading the body included |
| `error` | repr of the exception the call raised, or null (a requests call answered 4xx / 5xx does not raise: its status says so) |
| `bytes` | the body's length in bytes, or null when no body was read |
| `body` | the whole response text (the response's charset, else UTF-8, undecodable bytes replaced); null for neo4j, whose rows the caller reads lazily, for an `HTTPError` urllib raises, and for a requests call made with `stream=True` |

There is no row cap; a warning goes to stderr when the process's log passes `HARNESS_GRAPH_LOG_ALERT_GB` (default 5)
and at every further multiple. Capturing a body never changes what the caller gets: urllib's response is read once
and handed back as a buffered copy that reads, iterates and closes like it and answers every other attribute from the
original; requests had already read the body. A read that fails is the call's error. A log that cannot be written
never raises and never changes the call; threads append under the log's lock. With the log off nothing is read,
only counted. A call interrupted by the question's timeout is neither
counted nor logged.

## The LLM call trace

With `HARNESS_TRACE_DIR` set, every model call is appended to `<HARNESS_TRACE_DIR>/<qid>.jsonl` (the qid made
file-safe, `no-qid` outside a question) as one JSON object: `ts`, `qid`, `engine`, `endpoint`, `system` (the system
prompt), `prompt`, `replies` (a list), `prompt_tokens`, `completion_tokens`, `latency_s`, `success`, `error` and
`extras` (the finish reason, the reasoning effort, ...). A baseline that calls its own OpenAI client (SoG) has its
calls traced the same way, the conversation flattened to text and tool calls written into the reply. A trace that
cannot be written never fails the call.

## The model client

`run_llm(prompt, engine, ...)` makes one call to a local OpenAI-compatible server (vLLM, Ollama, LM Studio), the
OpenAI API, or OpenRouter, with retries, and returns `(text, metrics)`; with `n > 1` it returns n texts (the OpenAI
API natively, the other paths as n sequential calls). The API key is the argument, else the provider's environment
variable; local servers need none. Reasoning models get the settings table's `reasoning_effort` on every path.
`stop` (a string or a list) ends the generation at the first occurrence of any of them. `top_p` and `top_k` are sent
on every call that has them (`top_k` in the request body, never to api.openai.com); their defaults are the OpenAI
API's (top_p 1, top_k -1: disabled, `settings.OPENAI_SAMPLING`), because a value a request leaves out is filled by a
vLLM server from the model's generation_config, which is not what the authors ran. A caller that passes `None` sends
nothing, and the server's default applies. Token counts come from the response, else tiktoken estimates.

## The settings table

`settings.py` holds what every system runs with, on every model; the adapters and the model client read it.

* `MODELS`, per model (matched on the engine name, any provider prefix, the longest name wins): `reasoning_effort`,
  sent on every call to a reasoning model (low for gpt-oss), and `max_tokens`, the allowance per call where a system
  sets none.
* `SYSTEMS`, per system, the settings of the authors' code a run takes from this table, not from the environment:
  ToG's `head_order` (`"paper"`: its incoming-relation search as the paper's §2.1.2 states it); CoR's
  `relation_width` (3) and `depth`, per question set by the set's family (`depth_by_set`: 4 on CWQ, `cwq` also for
  `cwq-test`), else the default 3 where the paper names no set; SoG's `max_rows` (p = 1,000), `high_degree` (k = 50)
  and `max_turns` (50).
* `DECODING`, per system: its temperatures, token cap and sampling; each system runs its authors' values, KORA its own.
  On a reasoning model every cap is at least the model's allowance, since the reasoning counts against it. Where the
  authors' code sends no sampling value it ran at its API's default, and the table states that default and the adapter
  sends it: a value a request leaves out is filled by the vLLM server from the model's generation_config (Qwen3-2507's:
  top_p 0.8, top_k 20) unless the server runs with `--generation-config vllm`.
* The rule for every value: the paper's stated value; the code's where the paper is silent.
* A framework id reads its system's rows: `tog_native`, `tog_estimator` and `tog_gold` all read ToG's.
* `TIMEOUT_SECONDS` (200) is the per-question limit every system runs under and the limit the scorer applies.
  The timeout is a `BaseException`, so a baseline's `except Exception` cannot swallow it.
* `LLM_REASONING_EFFORT`, when set, overrides the table for local experiments only; every call's metrics record the
  effort used.

## Graphs

Graph errors. A query the graph rejects as malformed (HTTP 400 from Virtuoso: SPARQLWrapper's `QueryBadFormed` or
requests' `HTTPError` 400; Neo4j's `CypherSyntaxError`) finds nothing, as such a query does in the authors' code, and
is counted (the row's `graph_rejected`). Every other error (connection refused, a timeout, a 5xx, Neo4j unavailable,
anything else) is a graph that cannot answer: the adapter raises it, so the row is an error and the question can be
rerun, instead of going on over "no edges" to an answer from the model's memory.

MetaQA is served from its own Neo4j database (`NEO4J_METAQA_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`,
`NEO4J_METAQA_DATABASE`). Every node has the label `:Entity` with properties `id` and `name`, always the same text, so
topic entities need no resolution; there are nine relationship types (`DIRECTED_BY`, `WRITTEN_BY`, `STARRED_ACTORS`,
`RELEASE_YEAR`, `IN_LANGUAGE`, `HAS_TAGS`, `HAS_GENRE`, `HAS_IMDB_VOTES`, `HAS_IMDB_RATING`), without properties.
ToG's and PoG's code read it through the functions they call on Freebase (`id2entity_name_or_type`,
`get_head_relations` / `get_tail_relations`, `get_tail_entities` / `get_head_entities`, `abandon_rels`): a "head"
relation has the entity as its subject (outgoing), a "tail" relation as its object (incoming). An unknown id is named
`UnName_Entity`, as ToG's code expects of Freebase, and `abandon_rels` never abandons a relation: MetaQA has no
infrastructure relations to drop. A neighbour query caps
each direction separately, since a LIMIT after a UNION binds only its last branch.

## Scoring

`python -m harness score` reads finished run folders; a run's questions are the whole set it ran on, and a question
without a row is a miss. It reports Hits@Any and set F1 with 95% bootstrap intervals over questions, precision and
recall, the answered / empty / error / timeout rates, graph-trace support, LLM calls, input and output tokens, graph
calls and latency. `compare` gives the paired difference of two runs on the same questions (or of the two halves of
an Unseen set, source with twin) with its paired bootstrap interval, and for Hits@Any on Unseen pairs McNemar's test.

The code follows the paper's four metric parts; all four read the same answer sets (Common, last):

| part | file | the paper's metric |
|---|---|---|
| groundedness | `metrics/groundedness/` | Support: the graph-trace support rate (provenance) |
| reasoning | `metrics/reasoning.py` | Hit (Hits@Any) and F1, with their intervals |
| efficiency | `metrics/efficiency.py` | LLM cost: tokens and calls; also graph calls and latency |
| unseen | `metrics/unseen.py` | source against Unseen: the paired gap |
| common | `metrics/common/` | answer sets, the extractor, a run's rows, `score` and `compare` |

### Groundedness: graph-trace support (`metrics/groundedness/`)

The support rate is computed on the same answer sets as correctness, with no model, and no gold in the verdict.

**Calls records.** Each logged call becomes one record, in seq order, from the query as sent and the body as received;
no graph is contacted and nothing is guessed:

* `anchor`: the entity ids the query writes as constants, in order of first appearance; a query with none (a lookup
  by name, a schema listing) has an empty anchor and never roots a chain, unless name anchoring resolves the name;
* `relation`: the query's one relation constant, not counting the label predicates and `type.object.type`; null when
  it names none or several;
* `result`: from a SPARQL JSON body, per binding row, each bound URI shaped like an entity id is an entity; each
  literal is a value of the row's one entity (else of the record's one anchor), unless it is the label of an entity
  the call returned (read with a label predicate, or a variable such as `xLabel` / `x_name` beside the entity `x`). A
  label names its entity in every record of the question;
* `relations`: bound URIs that are not entity ids (relation and type listings);
* `n_rows`: the number of binding rows; null when the body is not a SPARQL JSON SELECT result. An ASK returns no
  result.

The log holds no entity types, so the type guard passes every logged entity. MetaQA's neo4j calls are logged without
their rows, so a MetaQA row has no evidence and is left out (counted). A log line cut short by a timeout, or a seq
repeated by a second process, is skipped or counted, never fatal.

**The support rule.** A record is rooted if its anchor is non-empty and every anchor id is a topic entity or an entity an
earlier rooted record returned. Per answer item: a yes/no item is unsupported; an item is strict if it matches a
result of a rooted record, or (a number) equals the output of the question's one operation (count, sum, average or
date difference, chosen from the question's wording) over rooted results; else lenient if it matches a result of any
anchored record; else unsupported. An item with an entity id matches that entity only (ids that are literals are
matched by text); otherwise a literal matches by value (numbers numerically, dates at the item's precision) and an
entity by its normalized label, with the type guard. A question's support is strict items / items; the set's support
rate is the mean over answered questions. The gold is read only afterwards, to split items into correct and wrong.

**Name anchoring** (on by default; both settings are reported): a call without an id anchor whose query names its start
entity by a string on a label predicate (`?x ns:type.object.name "Germany"@en`, `rdfs:label`, or a FILTER equality on
such a label variable) is anchored on the topic entity of that name, else on an earlier result of that label; an
ambiguous or unknown name leaves it unanchored.

Also reported: the answer rate, the lenient and unsupported shares, the yes/no share, the support without yes/no
items, the chance rate (answers shuffled within strata, seeded) and grounded Hits@Any (a question counts when one of
its items matches a gold whose name is strict in the calls, or a single gold id is returned by a rooted call).
Diagnostics, never the headline: near misses (items neither strict nor yes/no nor type-rejected that the whole-word
matcher accepts against a rooted result either way round) and items rejected by the type guard alone. A graph-calling
system's answered row with graph queries but no log is counted.

### Reasoning: correctness (`metrics/reasoning.py`)

#### The reading rule

A missing row, a row with an `error`, or a row over the per-question limit (200 s, whatever `--timeout` the run used)
is a miss in every metric: an empty answer set. Error rows are told apart: an infrastructure failure (a connection
timed out, refused or reset, a gateway error, an LLM connection error, ...) is still a miss, and such rows should be
rerun before any number is read; a timeout; any other error; over budget.

#### Matching and the metrics

* normalize: lower case; punctuation removed (ASCII punctuation and every Unicode punctuation character, deleted, not
  replaced by a space); the articles a, an, the removed; whitespace collapsed.
* match: an item matches a gold if the normalized gold appears in the normalized item as whole words: "1" does not
  match "1960", "us" does not match "business". Numbers and dates match as text.
* one-to-one: m is the size of a maximum matching between items and golds, so neither a gold nor an item counts twice.
* metrics: Hits@Any = 1[m >= 1]; P = m / |items| (0 for none); R = m / |golds|; F1 = 2PR / (P + R) (0 if P + R = 0);
  size = |items|. Also Hit@1 (the first item matches) and the size-capped Hit (Hits@Any when |items| <= 2 |golds|).

#### The text scorer

The row's own `eval` field and the stop oracle's plain hit use the text scorer: a gold is named in the text as whole
words after lower-casing and dropping punctuation and articles (no aliases); recall is the share of golds named;
precision the share of the answer's items (the system's list, else the text split on commas, semicolons, bars and
lines, a gold holding a separator kept whole: "Washington, D.C." is one item) that name a gold. `hit_authors` is the
ToG / PoG / SoG authors' own Hits@1: the answers inside {braces} if any, else the whole answer, lower case, spaces
removed, a hit when either contains the other (an empty answer is no hit), without the alias files ToG and PoG add.

#### Statistics

The interval is a percentile bootstrap 95% interval of the mean (10,000 resamples, seed 0, one resample matrix per
length, so every system on a set is resampled alike); over paired differences it is the paired interval. McNemar's
exact two-sided test runs on the discordant pairs.

### Efficiency: cost (`metrics/efficiency.py`)

LLM calls (`fw_metrics.num_llm_calls`), input tokens (`total_prompt_tokens`) and output tokens
(`total_completion_tokens`), a row without a count adding 0: means over every question with a row, failed and
over-budget rows included; missing questions have no cost and are counted apart. Latency (`wall_time_s`) as median
and 90th percentile; a timeout counts as the 200 s budget. Graph calls: the lines of the question's graph-call log,
its final attempt only (the window of the row's own wall time); a row without a log reads none, and the cell's graph
calls are then "no log" (counted, never substituted).

### Unseen: the gap (`metrics/unseen.py`)

Source minus twin on the Unseen pairs: the gap is the mean over pairs of s(source) - s(twin), on Hits@Any or F1,
paired per question, with a paired bootstrap 95% interval over pairs and McNemar's test on Hits@Any. Every pair
counts: a timeout or error on either half is a miss on that half. While runs are in progress a pair is read only when
both halves are scored; `pairs_scored` says how many, and the gap is final only when every pair is.

### Common: answer sets and the extractor (`metrics/common/`)

#### Reading the answers

Each system's answer slot gives one answer set per question:

| slot | systems | the answer set |
|---|---|---|
| list | KORA, CoR | the returned list; a row without a list holds its answer text as one item (none if empty) |
| braces | SoG | every {braced} answer of the final reply; a reply without braces is one item, the whole reply |
| tog | ToG | the braced answer after its {Yes} (the authors' extraction); a bare "Yes" left by a reply with no second braces is replaced by its stated answer (the text after its last "answer is", else its bold spans) |
| field | PoG | the JSON "Answer" field; a list stays a list, a string is one item |
| free | the direct LLM | the extractor's items (below) |

For free text: an empty response, or the literal reply "No answer found.", is the empty set. Without a standing
extractor entry the question is not run and never scored. Each extracted item is kept only if it appears verbatim in
the response; when the extractor proposed no item the set is empty; when it proposed items and none survived, the
response is split on commas, semicolons and line breaks. A row that holds a list is not free text. Item ids, for the
support rule only: KORA's `answer_ids`, when they align one to one with the items.

#### The extractor

Free-text systems answer in prose. For each stored row a fixed open model, at temperature 0 with thinking off
(`EXTRACTOR_MODEL` at `EXTRACTOR_BASE_URL`), is asked to list the answer items the response gives; code keeps an item
only if it is a substring of the response (a line with a leading bullet or number is tried once without it). The
model sees the question and the response text, nothing else, so a gold field has no path to the prompt. The cache is
`extract.jsonl` in the run's folder, one line per qid: `{"qid", "prompt_hash", "extractor_model", "items",
"rejected", "status": "ok" | "empty" | "error"}`. A rerun skips a row cached with the same prompt hash and model; an
"error" line, or one with another prompt hash or model, is dropped and the row extracted again.
