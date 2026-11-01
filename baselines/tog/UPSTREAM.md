# Think-on-Graph (ToG)

**Upstream.** <https://github.com/IDEA-FinAI/ToG>, commit `7ccbb92e17579f934bb778386230de47eca0ab67` ("Think-on-Graph:
Deep and Responsible Reasoning of Large Language Model on Knowledge Graph", Sun et al., ICLR 2024). The repository has
no licence file.

**Vendored unchanged** in `upstream/`: the Freebase files of their `ToG/` folder, their README and requirements.

| file | sha256 |
|---|---|
| README.md | `caa0da912cb46bb92f4318520df8ca6bab77044890b6056395293197a269e84a` |
| freebase_func.py | `f895e5b1ee2f73487641c1f14a6102142284dabafd5fc0074cda890b3c97f5a8` |
| main_freebase.py | `36139c81008c0ec08f6f1d9b81957fec59d4d7af05d2563704576340246d3e96` |
| prompt_list.py | `954e7f6f79fb08010474340f1fade455bcc3c0cd294270923387d5e703de13f5` |
| requirements.txt | `4fbe226d34853538fa5ec79bced0ae86d363772b5c7d8248d9438d685d2a194c` |
| utils.py | `4460752d33b730b2ca084e352ea84c60bb89c93aea349cece11158a8ea5f15c5` |

**How it runs** (`run.py`). One question is the `for data in tqdm(datas):` body of their `main_freebase.py`, read from
that file and executed for that question, with their functions and prompts. Their setting: width 3, depth 3,
temperatures 0.4 / 0, max_length 256 (exploration / reasoning temperature), 5 entities kept of 20 or more, pruning by
the LLM.

**Harness integration** (no edit to their files; their functions are replaced in the loaded modules):

- The model: their `run_llm` is answered by the harness's model client with their system message. Temperatures and
  max_length come from `harness/settings.py` (`DECODING["tog"]`, which holds their values); on gpt-oss, whose reasoning
  counts against max_length, the cap is the model's allowance (4,096). top_p 1.0 and no top_k are sent explicitly,
  the OpenAI API's defaults their call relied on.
- The graph: Freebase at the harness's endpoint (`SPARQL_ENDPOINT`); MetaQA (Neo4j) through `graphs.py`, which answers
  their templates, `entity_search` and `id2entity_name_or_type` in the shapes their code reads.
- The incoming-relation search follows the paper (§2.1.2, `(?, r, e)`): the released `entity_search` fills that
  template's two arguments in the wrong order and finds nothing along an incoming relation; the adapter swaps them.
- Graph errors: a query the graph rejects as malformed (HTTP 400, Neo4j's `CypherSyntaxError`) finds nothing and is
  counted (`graph_rejected`); a graph that cannot answer (connection refused, a timeout, a 5xx) raises, so the
  question is an error and can be rerun.
- The result: their `save_2_jsonl` is captured. When their loop raises after an answer was saved (it goes on after
  some `half_stop` saves and can then index an empty list), the saved answer is kept and the crash is recorded
  (`tog_crash_after_save`).
- The answer: read from the saved reply with `extract_content`, a helper in their `eval/utils.py`: the second `{...}`
  when the first is `{Yes}`, else the first; the whole reply when it has no braces. Their `eval/eval.py` itself reads
  only the first `{...}` (`clean_results`), which is `Yes` for every question their loop stopped on.

**Stop modes** (`--stop`, framework ids `tog_native`, `tog_estimator`, `tog_gold`):

- `native`: ToG as above.
- `estimator`: after each depth ToG's `{Yes}/{No}` sufficiency call (`reasoning`) is skipped. The Stopping Estimator
  (`ESTIMATOR_URL`, POST /score) scores each (topic entity, relation, direction) group retrieved at that depth that
  holds an entity kept by ToG's pruning, with every entity its search returned for the group (after its entity
  sampling, before its width-3 pruning) and the path from the question's entity as a query. The best group with a
  score of at least τ (`ESTIMATOR_TAU`, else the threshold the server reports on GET /health) is returned: ToG stops
  and answers with that group's entity names. Otherwise ToG prunes and continues; at its maximum depth (3) its LLM
  answers as released, including its answer without explored paths. A failed estimator request counts as a
  rejection.
- `gold`: the same stopping points and candidates; a group is accepted when the reference answers cover it (coverage
  and precision at least 0.5 over its first 50 items, `harness/stop_oracle.py`); of the accepted groups the one with
  the best F1 is returned.

Every stop decision, accepted or not, is written to the question's verdicts file (`harness/verdict_log.py`).

**Test questions in the authors' prompts** (kept as released). Their exemplars (`prompt_list.py`) show the gold
answers of WebQSP test questions WebQTest-1836 (De Smet), WebQTest-447 (The Long Winter), WebQTest-1923 (So Undercover,
and its eight CWQ test variants), WebQTest-282 (five of Russia's neighbours) and WebQTest-501 (Staten Island Summer).
