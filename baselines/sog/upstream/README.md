# ToG Publish - README (short)

This folder contains the main experiment runner for knowledge-graph QA experiments that
orchestrates an LLM and optional graph-querying tools (Freebase/Wikidata).

Quick start:
- Configure environment variables (e.g., OPENAI_API_KEY, DATASET, MODEL_NAME, NUM_EXAMPLES).
- For local vLLM runs, see run.sh (serves Qwen models and runs experiments).
- Run: `python main.py`

Primary environment variables:
- DATASET: CWQ, WebQSP, SimpleQuestions, GrailQA, QALD-9, QALD-10
- MODEL_NAME: model identifier (e.g., gpt-4o or Qwen/Qwen3-...)
- OPENAI_API_KEY / OPENAI_API_BASE_URL (for vLLM)
- NUM_EXAMPLES, USE_TOOLS, IO_MODE, SAMPLE, DEBUG, START_FROM_INDEX

Results are saved to results/<DATASET>/ with timestamps.
