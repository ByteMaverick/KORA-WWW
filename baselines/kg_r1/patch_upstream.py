"""The listed edits to the authors' KG-R1 clone (UPSTREAM.md); run once on a clone, safe to run again."""
from __future__ import annotations

import sys
from pathlib import Path

MARK = "# patched by baselines/kg_r1/patch_upstream.py v1"
SCRIPT = "eval_scripts/kg_r1_eval_main/eval_qwen_3b_turn5_hf.sh"
EVALUATOR = "verl/trainer/ppo/ray_evaluator_kg.py"
#: MARK lines per file once patched.
EDITS = {SCRIPT: 6, EVALUATOR: 7}
#: The authors' files at upstream.COMMIT that this patch is written against.
UPSTREAM_SHA256 = {
    SCRIPT: "e414bee5cc12dc55dc0b52059bb87c1fb1058416bbeca0c9b913361806335a99",
    EVALUATOR: "a16742eac4512701886330366415dea405fa9488150c614aea40e13d2c4a9a24",
}

_SCRIPT = [
    ('K_VALUES="1 2 3 4"\n',
     f'K_VALUES="${{KGR1_K_VALUES:-1 2 3 4}}"  {MARK}: edit 1\n'),
    ("VAL_BATCH_SIZE=128\n",
     f'VAL_BATCH_SIZE="${{KGR1_VAL_BATCH_SIZE:-128}}"  {MARK}: edit 8\n'),
    ("export CUDA_VISIBLE_DEVICES=4\n",
     f'export CUDA_VISIBLE_DEVICES="${{CUDA_VISIBLE_DEVICES:-4}}"  {MARK}: edit 2\n'),
    ("\n# Check if data file exists\n",
     f'\nif [ -n "${{KGR1_TEST_FILE:-}}" ]; then TEST_FILE="$KGR1_TEST_FILE"; fi  {MARK}: edit 3\n'
     "\n# Check if data file exists\n"),
    ('mkdir -p "eval_results/eval_kg-r1/$EXPERIMENT_NAME"\n',
     f'OUT_DIR="${{KGR1_OUT_DIR:-eval_results/eval_kg-r1/$EXPERIMENT_NAME}}"  {MARK}: edit 4\n'
     'mkdir -p "$OUT_DIR"\n'),
    ('    trainer.default_local_dir="eval_results/eval_kg-r1/$EXPERIMENT_NAME" \\\n',
     '    trainer.default_local_dir="$OUT_DIR" \\\n'),
    ('    2>&1 | tee "eval_results/eval_kg-r1/${EXPERIMENT_NAME}/evaluation.log"\n',
     f'    2>&1 | tee "$OUT_DIR/evaluation.log"  {MARK}: edit 4\n'),
]

_EVALUATOR = [
    ("import json\nimport os\nimport uuid\n",
     f"import json\nimport os\nimport time  {MARK}: edit 5\nimport uuid\n"),
    ("""            # Process batch and generate responses
            batch_metrics = self._evaluate_batch(test_data)
""",
     f"""            # Process batch and generate responses
            harness_ids = [str(s) for s in test_data["sample_id"]]  {MARK}: edit 6
            harness_start = time.time()  {MARK}: edit 6
            batch_metrics = self._evaluate_batch(test_data)
            batch_metrics["harness_batch_seconds"] = time.time() - harness_start  {MARK}: edit 6
            batch_metrics["harness_sample_ids"] = harness_ids  {MARK}: edit 6
"""),
    ("        reward_extra_infos = batch_metrics.get('reward_extra_infos', [])\n",
     "        reward_extra_infos = batch_metrics.get('reward_extra_infos', [])\n"
     f'        harness_ids = batch_metrics.get("harness_sample_ids") or []  {MARK}: edit 7\n'),
    ("""                "metadata": {
                    "response_length": len(response) if response else 0,
                    "prompt_length": len(prompt) if prompt else 0
                }
            }
""",
     f"""                "metadata": {{
                    "response_length": len(response) if response else 0,
                    "prompt_length": len(prompt) if prompt else 0
                }}
            }}
            if harness_ids:  {MARK}: edit 7 (records are interleaved: each question's rollouts in a row)
                record["harness_sample_id"] = harness_ids[i // max(1, max_len // len(harness_ids))]
                record["harness_batch_seconds"] = batch_metrics.get("harness_batch_seconds")
"""),
]


def _apply(path: Path, edits: list, count: int) -> None:
    s = path.read_text(encoding="utf-8")
    if MARK in s:
        return
    if "patched by baselines/kg_r1/patch_upstream.py" in s:
        raise SystemExit(f"{path} carries another version of this patch; restore it first (git checkout -- <file>)")
    for old, new in edits:
        if s.count(old) != 1:
            raise SystemExit(f"{path}: upstream changed ({old.strip().splitlines()[0]!r} found {s.count(old)} times); "
                             "patch by hand")
        s = s.replace(old, new, 1)
    if s.count(MARK) != count:
        raise SystemExit(f"{path}: {s.count(MARK)} marked lines, expected {count}")
    path.write_text(s, encoding="utf-8")


def patch(repo: Path) -> None:
    repo = Path(repo)
    _apply(repo / SCRIPT, _SCRIPT, EDITS[SCRIPT])
    _apply(repo / EVALUATOR, _EVALUATOR, EDITS[EVALUATOR])


def is_patched(repo: Path) -> bool:
    return all(MARK in (Path(repo) / f).read_text(encoding="utf-8") for f in EDITS)


if __name__ == "__main__":
    patch(Path(sys.argv[1]))
    print("patched", sys.argv[1])
