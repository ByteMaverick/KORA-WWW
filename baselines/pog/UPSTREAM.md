# Plan-on-Graph (PoG)

**Upstream.** <https://github.com/liyichen-cly/PoG>, commit `6d1627e40a147aaa0abfefaef8fcc1c3fcd04c43`
("Plan-on-Graph: Self-Correcting Adaptive Planning of Large Language Model on Knowledge Graphs", Chen et al.,
NeurIPS 2024). The repository has no licence file.

**Vendored unchanged** in `upstream/`: the Freebase files of their `PoG/` folder, their README and requirements.

| file | sha256 |
|---|---|
| README.md | `b38a809412462cf9de971f2d3dd7b24e1a77564711a16a82c9de7fe7f618e516` |
| freebase_func.py | `0ec9c3dd5edd92a619d773a2afcf3d33578695ba61893faf5afd7744b9067a42` |
| main_freebase.py | `c1e2b2349f7b97c83cd2e9e79397a553e3f12d43e44e19dde996ef9a8df9dacd` |
| prompt_list.py | `2dfe1bd1ba8e3a978114d215f22da97a485251dba4219ddf6001eaa54b304841` |
| requirements.txt | `11cbb065e94086534ed9d786749798cb8baf35028df8c66ba1251b72e3bc7b7f` |
| utils.py | `de06e07ae951f3d9e1177ec639edfec233dfe933c3c8c93726239f09c100d09f` |

**How it runs** (`run.py`). One question is the `for data in tqdm(datas):` body of their `main_freebase.py`, read from
that file and executed for that question, with their functions and prompts (subobjectives, relation pruning, entity
pruning with memory, reflection up to 5 times, reasoning, `half_stop` and the answer without explored paths). Their
setting: depth 4, temperatures 0.3 / 0.3, max_length 1,024 (the paper's App. E; their script defaults to 4,096), the
`msmarco-distilbert-base-tas-b` sentence model (`POG_SENTENCE_MODEL` overrides the name).

**Harness integration**:

- The model: their `run_llm` is answered by the harness's model client with their system message and their
  (text, token counts) return shape. Temperatures and max_length come from `harness/settings.py` (`DECODING["pog"]`);
  on gpt-oss the cap is the model's allowance (4,096). top_p 1.0 and no top_k are sent explicitly.
- The graph: Freebase at the harness's endpoint; MetaQA through `graphs.py` (their templates,
  `id2entity_name_or_type` and `provide_triple`, with its Freebase `"m."` test read as "is an entity node").
- One line of their loop is read differently: `if entity.startswith("m."):` becomes `if is_entity(entity):`, the same
  test on Freebase and "an entity node" on MetaQA. It is the only change to their loop.
- Their per-question memory folder (`../mem/<dataset>/<model>/<question>`) lives in a temporary folder.
- Graph errors as for ToG; a graph that cannot answer is raised again after their loop, whose bare `except:`
  swallows it. An exception their loop swallowed with no answer saved makes the question an error
  (`pog_unanswered: ...`), as their own workflow reruns unanswered questions.
- The clock: their bare `except:` also catches the harness's per-question timeout; the adapter keeps it, makes every
  later model call raise it again, and raises it after their loop, so the question is a timeout.
- The answer: the saved reply read as their `eval/eval.py` reads it (the JSON `Answer`, its fallbacks, else the reply).

**Stop modes** (`--stop`, framework ids `pog_native`, `pog_estimator`, `pog_gold`):

- `native`: PoG as above.
- `estimator`: PoG's `Sufficient: Yes/No` reasoning call is skipped. The Stopping Estimator scores each group
  (topic entity, direction, relation) -> entities kept after PoG's entity pruning, with the path from the question's
  entity as a query. The best group with a score of at least τ (`ESTIMATOR_TAU`, else the server's threshold) is
  returned: PoG stops and answers with that group's entity names. Otherwise reflection, the memory update and the
  search continue; at its maximum depth (4) PoG's LLM writes its native answer. A failed estimator request counts
  as a rejection.
- `gold`: the same stopping points and candidates, accepted by the reference-answer rule (coverage and precision at
  least 0.5 over the first 50 items); the best F1 among the accepted is returned.

Every stop decision, accepted or not, is written to the question's verdicts file.

**Test questions in the authors' prompts** (kept as released). Their subobjective prompt contains a CWQ test
question ("Which of the countries in the Caribbean has the smallest country calling code?",
WebQTrn-1812_bcc6abd18c7c4099763754c183edb542). The Caribbean exemplar (`prompt_list.py` lines 215 and 241) lists
candidate countries of the Caribbean that are the gold answers of 28 CWQ test questions on the Caribbean, among them
WebQTrn-1812_058fa348010b0e8e04f335a440658804 (Barbados) and WebQTrn-1812_688d69d484eeaffda36f06fbba9fe9fb
(Anguilla). Exemplars shared with ToG show the gold answers of WebQSP test WebQTest-1836 (De Smet), WebQTest-447 and
WebQTest-1923 (and its CWQ variants).
