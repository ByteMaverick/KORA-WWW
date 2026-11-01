"""KG-R1's evaluation settings from the settings table: shell exports (--env), Hydra overrides (--overrides)."""
from __future__ import annotations

import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from baselines.kg_r1.upstream import ENGINE  # noqa: E402
from harness import settings  # noqa: E402

SCRIPT_MAX_TURNS = 5   # eval_scripts/kg_r1_eval_main/eval_qwen_3b_turn5_hf.sh: MAX_TURNS=5


def env() -> dict[str, str]:
    s = settings.system("kg_r1")
    if s["max_turns"] != SCRIPT_MAX_TURNS:
        raise SystemExit(f"settings say {s['max_turns']} turns; the turn-5 script runs {SCRIPT_MAX_TURNS}")
    return {"KGR1_K_VALUES": " ".join(str(k) for k in range(1, s["rollouts"] + 1)),
            "KGR1_RELATION_FORMAT": s["relation_format"], "KGR1_MAX_TURNS": str(s["max_turns"])}


def overrides(port: int) -> list[str]:
    s, d = settings.system("kg_r1"), settings.decoding("kg_r1", ENGINE)
    url = f"http://127.0.0.1:{port}/retrieve"
    return [
        f"actor_rollout_ref.rollout.temperature={d['temperature']}",
        f"actor_rollout_ref.rollout.top_p={d['top_p']}",
        f"actor_rollout_ref.rollout.top_k={d['top_k']}",
        f"actor_rollout_ref.rollout.response_length={d['max_tokens']}",
        f"data.max_response_length={d['max_tokens']}",
        f"data.max_prompt_length={s['max_prompt_length']}",
        f"actor_rollout_ref.rollout.prompt_length={s['max_prompt_length']}",
        f"data.max_obs_length={s['max_obs_length']}",
        f"data.prompt_augmentation.guideline_level={s['guideline']}",
        f"actor_rollout_ref.rollout.search.search_url={url}",
        # their run/run.sh's flags for this checkpoint; they change only their own scoring, which is not used
        "reward_model.reward_kwargs.use_legacy_entity_f1=false",
        "reward_model.reward_kwargs.use_exact_match_binary_for_passk=true",
    ]


if __name__ == "__main__":
    if sys.argv[1:2] == ["--env"]:
        print("export " + " ".join(f"{k}={shlex.quote(v)}" for k, v in env().items()))
    elif sys.argv[1:2] == ["--overrides"] and len(sys.argv) == 3:
        print("\n".join(overrides(int(sys.argv[2]))))
    else:
        raise SystemExit("usage: eval_env.py --env | --overrides <port>")
