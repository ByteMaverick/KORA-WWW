# KB-BINDER

**Upstream.** <https://github.com/TIGER-AI-Lab/KB-BINDER> (formerly ltl3A87/KB-BINDER), `main` at `a98ac53` (abbreviated)
("Few-shot In-context Learning for Knowledge Base Question Answering", Li et al., ACL 2023). Licence: MIT
(`upstream/LICENSE`, the repository's). KB-BINDER is few-shot: there is no checkpoint; the LLM is the harness's
backbone (their Codex `code-davinci-002` is discontinued, and their README says to use another engine).

**Vendored unchanged** in `upstream/`: every file of the repository.

| file | sha256 |
|---|---|
| LICENSE | `ed50c4f58a9076d169b97d64e70113a16f05753b6f174d66b69460f8f345c129` |
| README.md | `c1102d2ccdbfaa09936ac064ce5748c38fc788c751f5e05dc8acb9a9e17498cf` |
| contriever_fb_relation/freebase_contriever_index/docid | `4eb9a16b644e750d2d39f15f8bdc0b487278d67992e45e1798c7d0cf0954b7b6` |
| contriever_fb_relation/index_relation_fb/_0.fdm | `ebf270248a53b115a678811629764a527a76f43088ef6901327c58b127a24acc` |
| contriever_fb_relation/index_relation_fb/_0.fdt | `759ca1d18e79290a5846a9676f5443cbc5c1e441ce34d678b5c2c1b33ad04a9c` |
| contriever_fb_relation/index_relation_fb/_0.fdx | `66690e77992affd1573432c504340db3262b76ad933158b34d6e9ef202b347f0` |
| contriever_fb_relation/index_relation_fb/_0.fnm | `8c4128ad7d34fc26129c401c6baaa109ded81a24d6fa42a8d833e50c2548ef52` |
| contriever_fb_relation/index_relation_fb/_0.nvd | `90bf9426c7ff6ef6b075cf3aa065f4ca5fb9d860a6081fd26426e360b0956580` |
| contriever_fb_relation/index_relation_fb/_0.nvm | `92ac5d078427461c23acdb1c7d6b3e9f63e63cb7827b6b60c3d5cafff1b27b39` |
| contriever_fb_relation/index_relation_fb/_0.si | `d44e3552b06683f955ded6b3bfcbbe3eb3c6e4d366556dd61948f78f59cfa086` |
| contriever_fb_relation/index_relation_fb/_0.tvd | `cdce35d8d106e0265ff1321b3279219a72b49bef592fd2c072b1cef4666e29be` |
| contriever_fb_relation/index_relation_fb/_0.tvm | `e31bc715f63a191ce2060298d13a43203d33b4f38c5ec60078ccfadec0c38dea` |
| contriever_fb_relation/index_relation_fb/_0.tvx | `cb0a73363df4d02e0e5a94c8627386f52ce31e26b102ce87b2ffbad61b722a4e` |
| contriever_fb_relation/index_relation_fb/_0_Lucene90_0.doc | `56d86519c00b952bf460b28adaab0993a763f54633b37bd3c7453519c37c2bec` |
| contriever_fb_relation/index_relation_fb/_0_Lucene90_0.dvd | `8e1f17d735019cd9e5059d11f1cf7354aa5e873b2729531747d622ec3c857180` |
| contriever_fb_relation/index_relation_fb/_0_Lucene90_0.dvm | `de322067d76ae031cc197026e2fc703e4e36a4a654b4901edb2ffff2fcd46e8b` |
| contriever_fb_relation/index_relation_fb/_0_Lucene90_0.pos | `d1a864948db4923833cda7990b4d122c6eaec35d750b932b31533d081318902c` |
| contriever_fb_relation/index_relation_fb/_0_Lucene90_0.tim | `6018349b2567e4d6e6da00d964dcdf9bdba7b5871e444fb14fb0ef886f327cf4` |
| contriever_fb_relation/index_relation_fb/_0_Lucene90_0.tip | `68acab96e45dec9a1620b06db9439450af7cde27a89414556340a900bc344f9c` |
| contriever_fb_relation/index_relation_fb/_0_Lucene90_0.tmd | `699ef29fd805daa9349e275c3f1b6660c617f69a6a79aef92536af241b727042` |
| contriever_fb_relation/index_relation_fb/segments_1 | `03d5607d6d135602d938e6ba47df632908aa1a38a62ca85be480ee48eb9ae53f` |
| contriever_fb_relation/index_relation_fb/write.lock | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| contriever_fb_relation/relation retrieval contriever.ipynb | `05910ac3d307d257c19a37ddd9489d9dd28edd530b820649e8e158b3246215ff` |
| few_shot_kbqa.py | `d11b2969af8324d9afddb7fd7029a40905544f5fba5430af400a980eb1eb3a78` |
| metaqa_src/bm25_trial.py | `af5c4b4891431bf801aabd3f5382efd8f1f70fb65519df39fd717fbc62f38d80` |
| metaqa_src/metaqa_1hop.py | `fa930501f1f6760248a1681481a2f295164117424a16fff683e208bfaf81d729` |
| metaqa_src/metaqa_1hop_retrieval.py | `2a4a6ccea5e9f70b464cad42d79a8c246635295077ab7f908fff7902d2c7198d` |
| metaqa_src/metaqa_2hop.py | `a444922d9502cd103902bf1911e812427fa87641cc24f0561c16a597055e8c88` |
| metaqa_src/metaqa_2hop_retrieval.py | `5cefd216a726a350125412daa32a3accd7661b9a1d5427a4e6340d439cd5b6ca` |
| metaqa_src/metaqa_3hop.py | `2517f0e729ead4fd8ab3b518c70a3e7025c0d84ace63f33f7a7c188e542a0194` |
| metaqa_src/metaqa_3hop_retrieval.py | `09e4f78c8b66f29b72bc258aba8feb7333b755ab4658c1f2b33eea4893d1e380` |
| requirements.txt | `0601a197ef177e89903041f1ccac6ab987bb5f5112d3831baf0c97bbc8db92eb` |
| sparql_exe.py | `40b7674022b1cd4f5748a594726eea1669a326687818500a3bf2864f0454e29a` |
| utils.py | `2bf1d83025bb2648b0064d52fb7bbe8981750c852a72d64d9acf39177bc894ac` |


`run.py` creates `upstream/data`, a link to `$KBB_DATA` (the layout their README asks for: `sparql_exe.py` reads
`data/fb_roles` at import); it is not part of the repository.

## How it runs

The paper's -R variant (exemplars retrieved by BM25 from the training set; their §3.2), from their files, at the
paper's values where it states them and the code's where it does not:

- **Freebase (GrailQA, WebQSP): KB-BINDER(6)-R** (their tables 2 and 3), `few_shot_kbqa.py`. Their `main()` setup is
  executed from their file with the arguments of a `--retrieval` run, and their `all_combiner_evaluation` for one
  question: 7 drafts at temperature 0.3 (the code's; the paper states none), the first 6 bound and executed (the
  paper's K = 6); BM25 + Contriever hybrid relation search kept to the 2-hop relations of the bound entities, trying
  the paper's m = 10 relation candidates (§4.3; their code keeps 7); the majority vote over executed answers. Shots,
  the paper's (§4.3): WebQSP 100, GrailQA 40, passed as their `--shot_num`.
- **MetaQA: KB-BINDER(1)-R** (their table 5: one draft, the type, at temperature 0; the paper's m = 1 is their type
  binder's top 1), `metaqa_src/metaqa_{1,2,3}hop_retrieval.py`, the script for the question's hop count. Their
  main-block setup and their test loop are executed from their file for one question, with the paper's 5
  BM25-retrieved exemplars for every hop (§4.3; their 1-hop script retrieves 20, their 2- and 3-hop scripts 5).
- **Coverage.** GrailQA, WebQSP and MetaQA, and the Unseen WebQSP sets (a set's source benchmark decides its
  configuration). Their code has no CWQ (no S-expression training data): those cells are "not applicable".

**Settings** (`harness/settings.py`, read by `run.py`): `SYSTEMS["kbbinder_official"] = {"shots": {"webqsp": 100,
"grailqa": 40, "metaqa": 5}, "relation_candidates": 10}`; `DECODING["kbbinder_official"] = {"temperature_drafts": 0.3,
"temperature_type": 0.0, "max_tokens": 256, "top_p": 1.0, "top_k": -1}` (every `Completion.create` of theirs sends 256
and top_p 1); on gpt-oss the cap is the model's allowance, 4,096, because its reasoning counts against it.

**Entry point.** `run_kbbinder_official(sample, args)` answers one question (`sample.qid`, `.question`,
`.topic_entity`; `args.engine`, `.kg_name`, `.dataset_benchmark`, `.question_timeout_seconds`) and returns the answer
items and the row's metrics. The paper's harness called it once per question as the framework `kbbinder_official`;
this release's harness does too, as `--system kbbinder` (stop `native`).

## What the harness adds (`run.py`, each marked `harness:`)

1. **The model.** Their `openai.Completion.create` is answered by the harness's model as a text completion of their
   prompt (the server's completion endpoint), their `n` samples in one request, cut at their stop sequence
   (`Question: `). Sent as a chat message instead, the models replied to the prompt rather than continuing it, and
   their parser skipped those drafts. On a reasoning model (gpt-oss) the completion holds its reasoning, then its
   answer: the stop is not sent (the server would match it in the reasoning too), the draft is the text after the last
   `assistantfinal` cut at their stop, and a sample with no answer is an empty draft, counted on the row as
   `kbb_samples_without_answer`. gpt-oss's reasoning effort is not sent: a completion carries no system message. On
   a Vertex engine (local smoke tests only), which has no completion endpoint, one chat call per sample.
2. **The graph.** Freebase: their `sparql_exe.sparql` points at the harness's endpoint (`SPARQL_ENDPOINT`). MetaQA:
   their `enti_to_fact_dict` (`kb.txt` read into `{entity: ["s|r|o", ...]}`) becomes a lookup on the harness's Neo4j
   graph returning the same lines, each entity read once per run; their setup's `kb.txt` statement is skipped.
3. **Entity linking: the gold topic entities**, which every system in the harness gets. The paper's entity binder looks
   a draft's names up in FACC1's surface map (2.5 GB), falling back to BM25 over every name in it. Their binder code
   runs unchanged on a name -> MID table that holds the question's topic entities, with BM25 over those names; a draft
   entity that is not a topic entity (for example a WebQSP constraint like "male") cannot be bound. Their `main()`
   still reads a surface map: it gets a one-line placeholder, because their BM25 needs at least one name.
4. **MetaQA input.** The paper's MetaQA questions mark the entity, `[Kill the Messenger]`; the harness's do not, so the
   topic entity gets those brackets.
5. **Exemplar pools.** Their training files minus every question the harness holds out, written once to
   `$KBB_DATA/cache/`: the val and test splits (`data/questions/<set>/{val,test,test_full}.json`) and both halves of
   the Unseen pairs (`data/unseen/<set>/{test,validation}.json`). The harness's WebQSP and MetaQA val questions, and
   the real halves of the WebQSP-Unseen validation pairs, come from those training files and would otherwise retrieve
   themselves.
6. **Exemplar MIDs their file leaves unnamed.** Their prompt builder names each exemplar's MIDs from their table of the
   question's entity nodes and raised a `KeyError`, outside any `try`, on two WebQSP training questions that name a
   MID the table lacks: WebQTrn-615 (`m.05zppz`, male) and WebQTrn-3496 (`m.0690_r`, no entity of Freebase; the topic
   is `m.0b90_r`, Mexico). A test question whose 100 exemplars include either failed with no model call (about 10% of
   WebQSP rows). Following the paper, which shows every exemplar's MIDs as names (§3.1), the adapter names a MID as
   their table names it for another question (`m.05zppz`: "male", from 19 questions' entity nodes); an exemplar with a
   MID their file names nowhere gets an empty logical form, and their prompt builder leaves it out, as it leaves out
   the 46 pool exemplars that have none. GrailQA's pool has none.
7. **The clock.** Their code catches every exception with a bare `except:`, the harness's 200 s timeout included:
   their retry loops (`while not got_result: try: ... except: sleep(3)`, forever on 2- and 3-hop MetaQA when a draft has
   no "operations:" line) and their binder, which executes every bound combination. Once the question's time is up,
   their `sleep`, model call and graph query raise the harness's timeout, so the loops end at once, and an answer that
   arrives after the time is a timeout. Their one-off setup (spaCy over the training questions, Contriever, pyserini)
   runs once per process (once per MetaQA hop) with the alarm paused and is not that question's time: the row records
   its seconds as `setup_seconds`, which the harness takes off its wall time.
8. **Graph errors.** On Freebase their `execute_query` sits inside their `try: ... except: continue`, so an
   unreachable graph gave an empty answer with no error. A query the graph rejects as malformed (HTTP 400) still finds
   nothing and is counted (`graph_rejected`); any other error is kept, every later graph query, model call and `sleep`
   of theirs raises it again, and it is raised after their code returns, so the row is an error and the question is
   retried.
9. **The paper's values where their code hard-codes others.** Neither is an argument of their functions, and their
   files stay unchanged: the adapter compiles the one function from their file with one constant replaced and puts it
   in their module. `bound_to_existed` (`few_shot_kbqa.py`): its two 7s become the paper's m = 10 (`_paper_m`); the
   MetaQA question-type call: the `n` of its one `bm25_train_full.get_top_n(..., n=...)` becomes the paper's 5
   (`_paper_shots`).
10. **The answer.** The executed result, Freebase MIDs named by their own `get_friendly_name`; their
    `all_combiner_evaluation` gets one statement appended, `return` the answer it computed (it only logs it).

## Edits to our files for this release

`run.py` is the paper harness's file (`4eebf1b`, abbreviated); comments and docstrings shortened, and these edits (the
file is AST-identical to the source with exactly these applied): its imports rewired to this release's modules (table
below); the held-out questions read from `data/questions/` and `data/unseen/` (item 5: GrailQA's full test is
`test_full.json` here, and the paper's Unseen split folders are the pair files); the setup's log line about unnamed exemplar MIDs
lost an internal label. `fetch.sh`: its header comment only.

| imported in the paper's harness | here |
|---|---|
| `baselines.official_llm.chat` | `baselines.official_llm.chat` |
| `baselines.official_llm.complete` | `baselines.official_llm.complete` (moved unchanged; its trace call is `harness.llm._trace_call`) |
| `baselines.official_loader.load` | `baselines.official_loader.load` |
| `common.graphs.errors.rejected` | `harness.neo4j_client.rejected` (the same function) |
| `common.metrics.QueryMetrics` | `harness.runner.QueryMetrics` (the same class) |
| `common.settings` (`system`, `decoding`, `MODELS`, `model_of`) | `harness.settings` (its `kbbinder_official` rows hold the values above) |
| `common.main.QuestionTimeoutError` | `harness.settings.QuestionTimeoutError` |
| `common.timeouts.untimed` | `harness.settings.untimed` |
| `common.neo4j_client` (`NEO4J_URI`, `NEO4J_DATABASE`, `run_cypher`) | `harness.neo4j_client` |
| `data/kora/splits/<set>/{val,test}.json` (item 5) | `data/questions/<set>/{val,test}.json` |

Held-out question texts, this release vs the paper's run (`_held_out`'s own logic on both trees): GrailQA 5,879 vs
5,879, the same set. WebQSP 3,299 vs 4,499: the 1,200 not held out here are an earlier build of the Unseen twins
(`webqsp-unseen-v0`), not shipped; they are renamed questions, and none is in WebQSP's training split (checked). MetaQA
817 vs 1,621: the 804 not held out here are the paper's counterfactual and anonymised copies of MetaQA's val and test
(`metaqa-cf`, `metaqa-anon`), not shipped; their entity names are rewritten, and none is in the shipped MetaQA training
sample (2,000 questions; checked). The two MetaQA questions that also appear in the benchmark's training split are held
out here, as in the paper (`test_full.json`).

The local Vertex path of item 1 calls `chat` with a `vertex:` engine; this release's `chat` has no Vertex client.

## Steps

1. Data, with internet: `KBB_DATA=<folder> bash baselines/kbbinder/fetch.sh` (gdown; about 250 MB: `fb_roles`,
   `grailqa_v1.0_train.json`, `webqsp_0107.train.json`, `metaQA/*`, the Contriever Faiss index `contriever/index`;
   not the 2.5 GB surface map).
2. Dependencies: pyserini 0.22.1 (`--no-deps`; current pyserini reads index metadata over the network at import, which
   fails offline; it needs a Java 11+ JVM, the paper's runs used Java 21), faiss, spaCy with `en_core_web_sm`,
   rank_bm25, SPARQLWrapper, and `facebook/contriever` in the Hugging Face cache.
3. Run with `KBB_DATA` set, through the harness, which calls `run_kbbinder_official` (above) once per question and
   passes the graph settings on (`FREEBASE_SPARQL_URL` or `--sparql-endpoint`; `NEO4J_METAQA_URI` for MetaQA):
   `python -m harness run --system kbbinder --dataset webqsp --out runs/kbbinder-webqsp` (also `grailqa`, `metaqa`,
   `webqsp-unseen`, `webqsp-unseen-real`), then `python -m harness score runs/kbbinder-webqsp`.

## Disclosed

- **Codex replaced.** Their engine is discontinued; the harness's backbone answers their prompts as text
  completions (item 1). On gpt-oss-120B the runs did not finish (`results/`: WebQSP incomplete, GrailQA and MetaQA not
  run).
- **Gold topic entities** for the FACC1 surface map (item 3): every system in the harness gets them; KB-BINDER's
  binder cannot bind a draft entity that is not one.
- **Exemplar MIDs** named from their own file, or the exemplar left out (item 6); the setup log lists both exemplars,
  and a question that retrieved WebQTrn-3496 shows one exemplar fewer.
