# Search-on-Graph (SoG)

**Upstream.** The supplementary material of "Search-on-Graph: Iterative Informed Navigation for Large Language Model
Reasoning on Knowledge Graphs" (Sun et al., arXiv:2510.08825), <https://openreview.net/forum?id=aAyONRMcdB>, which holds
the experiment code (`main.py`). There is no public repository or commit and no licence file; the files are identified
by their checksums.

**Vendored unchanged** in `upstream/` (their `data/` and `efficiency-statistic/` folders are left out):

| file | sha256 |
|---|---|
| README.md | `8b24dadbb476e2d1f091daf6ff3ea363e5525ec2a0e7da77fc9f816a74b4b301` |
| few_shot_llm_tools_unified_helper.py | `d018e7cdb47c5b9a96e66138ac84d937d082dcf74c03a19836a8feefa741416c` |
| main.py | `7d61544c7acca4b772bc8c43cd72b92187f3f63a1c7db4e3fdd0c5fc94240fb5` |
| prompt.py | `83740bb4405f9520704e105fd7bbb40c3f90ff8a0ad03cb5e248c7a498f688af` |
| requirements.txt | `c704dce3e88d020d639526cfc5d8c51ab4ce1c31a07e3c9f42287ff1d280e480` |
| run.sh | `289bb6856c8e2bf91fa135fba2e304958b51c280af6f1f53939265ea1c0219d0` |

**How it runs** (`run.py`). One question is their `process_single_question` (`main.py`): their system prompt and
exemplars, their one tool (`get_adjacent_relations_and_entities`), their loop of up to 50 model turns, their stop (a
reply with no tool call is the answer). SoG runs with its own stopping only (`--stop native`, framework id
`sog_native`).

**The setting**: the paper's where it states one, the code's where it is silent. The paper's: five exemplars per
dataset, high-degree threshold k = 50 (properties only above it), at most p = 1,000 rows per tool result (§5.1); their
code keeps 2,000 rows (`top_k = 2000`). The code's: up to 50 model turns (`main.py:312`); their Qwen3 Instruct
sampling (`main.py:319-353`: temperature 0.7, top_p 0.8, top_k 20, presence penalty 1.5, 20,480 output tokens); on
gpt-oss, where their code sets none, the OpenAI defaults (temperature 1.0, top_p 1.0, no top_k) are sent explicitly.
Their prompt lets the model answer from its own knowledge when the graph does not give enough; that is kept.

**Harness integration** (`run.py`, `graphs.py`, `extra.py`; their files are not edited):

1. `NUM_EXAMPLES = 10`: their `get_system_prompt` samples `num_examples - 5` exemplars, so 10 gives the paper's five.
   Their order is seeded per question.
2. The model is the harness's server through the same OpenAI client call; every turn sends the decoding of
   `harness/settings.py` (`DECODING["sog"]`). When prompt + 20,480 output tokens exceed the server's context, the
   request is resent with the room that is left.
3. Freebase at the harness's `SPARQL_ENDPOINT` (theirs: `localhost:8890`).
4. MetaQA, which the authors did not run: the tool over the harness's Neo4j helper with the authors'
   post-processing (`graphs.py`); the tool spec with the graph's name (`extra.py`, as they did for Wikidata); five
   exemplars built from MetaQA train questions along their gold paths with the real tool (`build_examples.py` ->
   `examples_extra.json`), using only train questions whose text names every relation on their gold path; their CWQ
   instructions (no graph named in them).
5. The gold never reaches their code (their readers get a placeholder for their exact-match printout); scoring is the
   harness's.
6. The answer is the curly-brace entities of their final reply (their own extraction); the reply if it has none.
7. The clock: their tool calls run in worker threads inside `with ThreadPoolExecutor`, whose exit waits for every
   pending call; the executor their `main.py` uses is a subclass that does not wait when the harness's timeout leaves
   the block, so the question ends at its time.
8. Graph errors: a query the graph rejects as malformed finds nothing and is counted (`graph_rejected`), as their
   Freebase tool returns `[]` after its three attempts. A graph that cannot answer is raised (on Freebase after their
   three attempts); since their loop hands a tool exception to the model as a message, the adapter sends no further
   model turn after a failure and raises it, so the question is an error.
9. The paper's p = 1,000 rows: every tool result reaches the model through their `convert_tool_result_to_table`, and
   the adapter hands it at most the first p rows (`_table_of_p_rows`).
10. A tool call written as text: their exemplars write each tool call as text (`Tool Call: [{...}]`, then an imagined
    result), and their loop ends on any reply without a native tool call. A reply that writes a call that way has its
    first written call executed as the native call it stands for and the rest of the reply dropped; nothing is added
    to the prompt. A reply with neither is their final answer. Each row counts these (`sog_text_tool_calls`).

**Rebuilding the MetaQA exemplars.** `build_examples.py --train FILE` needs MetaQA train questions with their gold
relation paths, which this release does not ship; the shipped `examples_extra.json` holds the exemplars the paper's
runs used.

**Test questions in the authors' prompts** (kept as released; all five WebQSP exemplars are in every WebQSP prompt):

* "what years have the dallas cowboys won the superbowl" (`prompt.py:676-689`) answers {Super Bowl XXX}, the gold of
  WebQSP test WebQTest-994 ("when was the last time the dallas cowboys won the superbowl").
* "who voices Darth Vader in the clone wars" (`prompt.py:630-671`) walks from Darth Vader (m.0f2y0) to Matt Lanter,
  the gold of WebQTest-560 ("who voiced darth vader") and one gold of WebQTest-794, WebQTest-1065, WebQTest-1171 and
  CWQ WebQTest-1171_05a60a5e9a6d52b6f23993c4868c111f.
* Others show the gold answers of WebQTest-745 (Politician, Roman emperor; and two CWQ variants), WebQTest-282
  (Azerbaijan, Estonia, Norway) and, beside Japan in a film's table, Japanese Language (WebQTest-638).
