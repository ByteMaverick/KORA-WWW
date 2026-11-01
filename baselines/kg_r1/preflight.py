"""Check that KG-R1's graph server answers its four operations on an exported set (no model needed)."""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

OPERATIONS = ("get_relations_out", "get_relations_in", "get_entities_out", "get_entities_in")


def _post(url: str, payload, timeout: float = 30.0):
    req = urllib.request.Request(url.rstrip("/") + "/retrieve", data=json.dumps(payload).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def _get(url: str, path: str, timeout: float = 30.0):
    with urllib.request.urlopen(url.rstrip("/") + path, timeout=timeout) as r:
        return json.load(r)


def probe(export: Path, set_name: str) -> dict:
    """A question with a non-empty subgraph and one of its triples, by name, from the exported files."""
    from baselines.kg_r1.export import dataset_name, kg_name
    folder = Path(export) / "kg" / kg_name(set_name)
    names = [l.strip() for l in (folder / "entities_text.txt").open(encoding="utf-8") if l.strip()]
    relations = [l.strip() for l in (folder / "relations.txt").open(encoding="utf-8") if l.strip()]
    for line in (folder / "test_simple.json").open(encoding="utf-8"):
        r = json.loads(line)
        for h, p, t in (r.get("subgraph") or {}).get("tuples") or []:
            if h != t:
                return {"sample_id": str(r["id"]), "dataset_name": dataset_name(set_name), "head": names[h],
                        "relation": relations[p], "tail": names[t]}
    raise SystemExit(f"{folder}: no question with a non-empty subgraph")


def check(url: str, export: Path, set_name: str, post=_post, get=_get) -> list[str]:
    """The problems found (empty: the server answers the four operations on this export)."""
    p = probe(export, set_name)
    problems = []
    try:
        if get(url, "/health").get("status") != "healthy":
            problems.append("/health is not healthy")
        listed = set(get(url, "/actions").get("actions") or [])
        if not set(OPERATIONS) <= listed:
            problems.append(f"/actions lists {sorted(listed)}, not the four operations")
    except Exception as e:
        return [f"the server at {url} does not answer: {type(e).__name__}: {e}"]
    base = {"sample_id": p["sample_id"], "dataset_name": p["dataset_name"]}
    asks = [
        ({**base, "action_type": "get_relations_out", "entity_id": p["head"]}, p["relation"]),
        ({**base, "action_type": "get_relations_in", "entity_id": p["tail"]}, p["relation"]),
        ({**base, "action_type": "get_entities_out", "entity_id": p["head"], "relation": p["relation"]}, p["tail"]),
        ({**base, "action_type": "get_entities_in", "entity_id": p["tail"], "relation": p["relation"]}, p["head"]),
    ]
    try:
        answers = post(url, [a for a, _ in asks])
    except Exception as e:
        return problems + [f"POST /retrieve failed: {type(e).__name__}: {e}"]
    if not isinstance(answers, list) or len(answers) != len(asks):
        return problems + [f"POST /retrieve returned {str(answers)[:200]}"]
    for (ask, want), got in zip(asks, answers):
        content = ((got.get("choices") or [{}])[0].get("message") or {}).get("content", "")
        if not got.get("success") or want not in content:
            problems.append(f"{ask['action_type']}({ask['entity_id']!r}{', ' + repr(ask['relation']) if 'relation' in ask else ''})"
                            f" on {p['sample_id']}: {content[:200]!r} does not hold {want!r}")
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="the graph server, http://<host>:<port>")
    ap.add_argument("--export", type=Path, required=True, help="the export folder (holds kg/<K>/)")
    ap.add_argument("--set", required=True)
    a = ap.parse_args(argv)
    problems = check(a.url, a.export, a.set)
    print("\n".join(problems) or f"kg_r1 preflight ok: {a.url} answers the four operations on {a.set}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
