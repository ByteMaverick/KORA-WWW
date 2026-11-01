# Chain-of-Relations (CoR)

**Upstream.** <https://github.com/Chenhui7au/Chain-of-Relations>, commit `f92557913271ad4b8eeb5046a206a8c1b4ffb2ff`
("Chain-of-Relations: Faithful and Efficient LLM Reasoning over Knowledge Graphs via Relation-Centric Exploration",
Liu, Zhou and Wang, Findings of ACL 2026). The repository has no licence file.

**Vendored** in `chain_of_relations/`: their package, with every import rewritten from `chain_of_relations.` to
`baselines.cor.chain_of_relations.`. Left out: their runner (`run.py`; `baselines/cor/run.py` passes the same
arguments), `utils.py`, `graph.py`, `eval/`, the Wikidata backend and the other methods (`methods/{tog,pog,io_prompt,
cot_prompt}`).

Files unchanged but for the import paths (the checksum is of the file with its imports turned back to
`chain_of_relations.`, which then equals theirs byte for byte):

| file | sha256 |
|---|---|
| __init__.py | `d395eb03d737b2b348c73d92da670256c4efd33efa853127c5f115199f415630` |
| kg_backend/base.py | `955d038b7df6e0647ea5af1ded5109830a6fd9b41eab1bfab295068d8d8999c4` |
| kg_backend/freebase/__init__.py | `1f0d87cad2175ab909d55832290cb9baa71ffb636e007ec38b68dd3073ec91cd` |
| kg_backend/freebase/query_template.yml | `23cc33697744fb50f7a40ff7b619b8032d625e78aa066a04d9fc4bdf30bed725` |
| kg_backend/freebase_backend.py | `968c39a7991afd0dd15082f331447dde0372e04437297e1ad4a0b5eba1a4cd96` |
| llm_api.py | `af783d191d0f3842bf4b78e035477a4043522307a2ec025c93a1a58562e89f68` |
| methods/__init__.py | `cdbaa92bc4127b6c5e0be72a689b56621951707f7baff3ff5614d65a3affd89b` |
| methods/cor/__init__.py | `ffb8a8b885a1ec4fd1252ca4103e38fcf6f730bf31497ad8ae8056b8fb1cfe8a` |
| methods/cor/dfs_stack.py | `e45ad2594c82bebfad1ec72d7dccfdf19c28ed2d5f2da2c94b822d6eefd04c51` |
| methods/cor/prompt.yml | `81bd5991daaf18a9dc8d5e59df0bf3053f9ade7a4f04cb2b027fb0cbbc23d226` |
| methods/cor/tools/__init__.py | `1255810cc0f73bd824d63d47f531265194ce1652b81c089198a3486ea8076a40` |
| methods/cor/tools/filter.py | `8e27ec673958494ea76e2e17932d6d85f9ad7c20cec3aa49682151f930903ebc` |
| schema.py | `ded1fe4f53286a354dbd86558606d2fbb1ed6f1a0a2e1804991e0c22d8a32413` |
| tools/__init__.py | `e0c4f5ddb9f8e7a2c7ea68c8914608d8338e9c41dd018899b908de8ba8b582cd` |
| tools/entity_prune.py | `5dde96aabed9c028a4a44ac9170f021312ab4100841c61259b0de0dbd0f391d7` |
| tools/entity_search.py | `bbc77425e62bc30cbec58765880d1d32dbe1b0b04d733b7734a37bb5da0f9de7` |
| tools/generate_directly.py | `343724b916b211610efb2a1288fdd8b42088e9976debee7cc48bd31cc7dd16a9` |
| tools/reasoning.py | `04a0c8641fa5b2cbb9978870ac140b8749cc44aebe8e6ee32a637aaa88034597` |
| tools/relation_prune.py | `6589817f5e46f74f1485a2bd9389b761ed3e8f7f28cfc1abc64dd09e244f8c2d` |
| tools/relation_search.py | `7d10cb290d27077cbce72a628a9718086d8389efbd9e0e627dc653c9de255387` |

Files we edited (each edit marked `harness:`):

| file | change |
|---|---|
| methods/cor/agent.py | the estimator and gold stop (below), after CoR's reasoning call at each node with an executed result. With neither passed (`cor_native`) the block never runs. |
| kg_backend/freebase/db_func.py | their two SPARQL helpers retry three times on `except BaseException`, which also caught the harness's per-question timeout; each handler now starts with `if not isinstance(e, Exception): raise`. Without those two inserts the file is theirs byte for byte (sha256 `c49b5a32ad9dc5506a64fd876023b7c229e7d3a46ad73b0314b5c6ec367eaad8`). |
| kg_backend/__init__.py | rewritten: theirs builds a Freebase and a Wikidata backend when imported; the harness always passes its Freebase backend in, so `get_default_backend` raises instead. |

**Settings** (`harness/settings.py`): temperature 0.01 for every step (paper §5.3; their runner uses 0.3 / 0.1); depth
4 on CWQ and 3 elsewhere (paper §5.3; their runner's 3 where the paper names no set); relation width 3; 1,000
relation candidates into relation pruning (their runner's); max tokens 512 (their code), the model's allowance
(4,096) on gpt-oss; the first topic entity only (their behaviour). top_p 1.0 and no top_k are sent explicitly.

**Harness integration** (`run.py`): model calls through the harness's model client, each counted under the CoR step
that made it; their Freebase backend on the harness's endpoint (`FREEBASE_SPARQL_ENDPOINT`, set from
`SPARQL_ENDPOINT`). CoR is not run on MetaQA: its tools write SPARQL themselves and MetaQA is served from Neo4j.

**Stop modes** (`--stop`, framework ids `cor_native`, `cor_estimator`, `cor_gold`):

- `native`: CoR as above.
- `estimator`: CoR's LLM still proposes its action. At every node whose executed relation chain returns entities, the
  Stopping Estimator scores the full result (every returned id and literal, with the chain from the topic entity as a
  query). If the score is at least τ (`ESTIMATOR_TAU`, else the server's threshold) the executed result is returned
  unchanged. Otherwise CoR's Stop or Constraint becomes Forward (Backtrack at the maximum depth); Forward and
  Backtrack are unchanged. If the stack empties, CoR's LLM answers directly (`generate_directly`). A failed estimator
  request counts as a rejection.
- `gold`: the same stopping points, the result accepted by the reference-answer rule (coverage and precision at least
  0.5 over the first 50 items).

Every stop decision, with CoR's own action at that point, is written to the question's verdicts file.
