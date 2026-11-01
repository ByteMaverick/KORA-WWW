"""The one settings table every system runs with, on every model (README: the settings table)."""
from __future__ import annotations

import os

TIMEOUT_SECONDS = 200


class QuestionTimeoutError(BaseException):
    """A question over its time limit; a BaseException, so a baseline's `except Exception` cannot swallow it."""


#: Seconds the alarm was paused this question (a system's one-off setup), taken off the row's wall time.
untimed = {"seconds": 0.0}


MODELS: dict[str, dict] = {
    "gpt-oss-120b": {"reasoning_effort": "low", "max_tokens": 4096},
    "gpt-oss-20b": {"reasoning_effort": "low", "max_tokens": 4096},
    "qwen3-30b-a3b": {"reasoning_effort": None, "max_tokens": 1024},
    "qwen3-4b": {"reasoning_effort": None, "max_tokens": 1024},
}


SYSTEMS: dict[str, dict] = {
    "tog": {"head_order": "paper"},
    # CoR: paper §5.3, depth 3 on WebQSP / QALD10 and 4 on CWQ; their runner's 3 where the paper is silent;
    # relation width 3 in both.
    "cor": {"relation_width": 3, "depth": 3, "depth_by_set": {"cwq": 4}},
    # SoG (paper §5.1: "we set the high-degree threshold k = 50 and the maximum result size p = 1000"): p and k
    # the paper's; 50 turns the code's (main.py:312), where the paper states none.
    "sog": {"max_rows": 1000, "high_degree": 50, "max_turns": 50},
    # KB-BINDER (paper §4.3: N = 100 exemplars for WebQSP, 40 for GrailQA, "For MetaQA, we only sample 5 questions";
    # "the top 10 (i.e., m = 10) relation candidates ... for GrailQA, WebQSP and GraphQA, and the top 1 (i.e.,
    # m = 1) for MetaQA"). The code: MetaQA 1-hop 20 exemplars, m = 7 on Freebase; the paper's values run.
    "kbbinder_official": {"shots": {"webqsp": 100, "grailqa": 40, "metaqa": 5}, "relation_candidates": 10},
    # KG-R1 (paper v5 §4.1 and Table 1: H = 5 turns; App. E.4: deterministic evaluation; §4.5: flat relations): its
    # authors' checkpoint and the run/run.sh turn-5 script's settings (their evaluation_k_beam.yaml where the paper
    # is silent: 3,500-token prompt window, 512-token observations, the "detailed_flat" guideline). One rollout per
    # question: at temperature 0 their four (pass@k) are one answer four times.
    "kg_r1": {"checkpoint": "SCL2025/KG-R1-CQW@fa505b2e330843dbeebef7dd695141ebfc1af764", "max_turns": 5,
              "rollouts": 1, "relation_format": "flat", "guideline": "detailed_flat", "max_prompt_length": 3500,
              "max_obs_length": 512},
}


#: The OpenAI API's defaults (top_p 1, no top_k): what every authors' code that calls that API without them ran at,
#: and what run_llm sends when its caller passes none (harness/llm.py).
OPENAI_SAMPLING = {"top_p": 1.0, "top_k": -1}
_OURS = {"temperature": 0.0, "max_tokens": 1024, **OPENAI_SAMPLING, "source": "the harness's own (no authors' code)"}
DECODING: dict[str, dict] = {
    "direct_llm": _OURS,
    "tog": {"temperature_exploration": 0.4, "temperature_reasoning": 0.0, "max_tokens": 256,
            **OPENAI_SAMPLING,
            "source": "ToG main_freebase.py defaults; their openai.ChatCompletion sends no top_p"},
    "pog": {"temperature_exploration": 0.3, "temperature_reasoning": 0.3, "max_tokens": 1024,
            **OPENAI_SAMPLING,
            "source": "PoG paper App. E (their script defaults to 4,096); their OpenAI call sends no top_p"},
    "sog": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5, "max_tokens": 20480,
            "model": {"gpt-oss": {"temperature": 1.0, **OPENAI_SAMPLING, "presence_penalty": None}},
            "source": "SoG main.py:319-353 (Qwen3 sampling); gpt-oss: none sent, the OpenAI defaults "
                      "(T 1.0, top_p 1.0, no top_k)"},
    "cor": {"temperature_exploration": 0.01, "temperature_reasoning": 0.01, "max_tokens": 512, **OPENAI_SAMPLING,
            "source": "CoR paper §5 (T 0.01); max_token 512 from their code; their LLMAPI sends no top_p"},
    "kbbinder_official": {"temperature_drafts": 0.3, "temperature_type": 0.0, "max_tokens": 256, **OPENAI_SAMPLING,
                          "source": "KB-BINDER few_shot_kbqa.py / metaqa_src (every Completion.create: 256, "
                                    "top_p=1)"},
    "gcr": {"temperature": 0.0, "max_tokens": 1024, **OPENAI_SAMPLING,
            "source": "GCR step 2 (the harness's model): the authors' T 0.0; max_tokens 1,024 is the harness's cap "
                      "(their client sends none); their client relied on the OpenAI API's top_p / top_k "
                      "defaults, which the patched client sends (baselines/gcr/patch_upstream.py)"},
    # KG-R1 answers with its own 3B (baselines/kg_r1): the paper's evaluation decoding (App. E.4: "temperature 0.0 and
    # sampling disabled"); per turn the response window of their config (data.max_response_length 256). Their turn-5
    # script itself samples at the rollout's T 1.0 (it sets no validate flag): the paper's value runs.
    "kg_r1": {"temperature": 0.0, "max_tokens": 256, "top_p": 1.0, "top_k": -1,
              "source": "KG-R1 paper v5 App. E.4 (T 0.0, do_sample false); 256 tokens per turn from their "
                        "evaluation_k_beam.yaml; sent as Hydra overrides by baselines/kg_r1/eval_env.py"},
    "kora": {"temperature": 0.0, "max_tokens": 3000, **OPENAI_SAMPLING,
             "source": "KORA's own default; run_llm's sampling defaults"},
}


def family_of(framework: str) -> str:
    """The system a framework id belongs to: `tog_gold` is ToG, `direct_llm` is itself."""
    return framework if framework in DECODING else framework.split("_", 1)[0]


def decoding(framework: str, engine: str) -> dict:
    """The decoding a system runs with on this engine (DECODING; every stop rule of a system reads its row)."""
    row = DECODING.get(family_of(framework))
    if row is None:
        raise KeyError(f"no decoding row for {framework!r} in harness/settings.py DECODING")
    out = {k: v for k, v in row.items() if k not in ("model", "source")}
    model = model_of(engine)
    for family, over in (row.get("model") or {}).items():
        if model and model.startswith(family):
            out.update(over)
    if model and MODELS[model]["reasoning_effort"]:
        out["max_tokens"] = max(out["max_tokens"], MODELS[model]["max_tokens"])
    return out


def system(framework: str) -> dict:
    return dict(SYSTEMS.get(family_of(framework), {}))


def model_of(engine: str) -> str | None:
    """The MODELS row an engine name belongs to, whatever its provider prefix; the longest name wins."""
    e = (engine or "").lower().rsplit("/", 1)[-1].split(":", 1)[-1]
    found = [m for m in MODELS if e == m or e.startswith(m + "-") or e.startswith(m + "_")]
    return max(found, key=len) if found else None


def reasoning_effort(engine: str) -> str | None:
    """The effort to send with this engine's calls, or None for a model that takes none."""
    override = os.getenv("LLM_REASONING_EFFORT", "").strip()
    m = model_of(engine)
    if m is None:
        return None
    effort = MODELS[m]["reasoning_effort"]
    return (override or effort) if effort else None
