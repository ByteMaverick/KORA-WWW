"""The harness's topic entities as a subgraph keyed by entity name knows them (baselines/gcr/UPSTREAM.md)."""
from __future__ import annotations

import ast


def topic_nodes(topic_entity, sub: dict) -> tuple[list[str], list[str], dict]:
    """(nodes, absent, mapped) for a question's topic entities ({id: name}, or its repr) in the subgraph `sub`."""
    topic = topic_entity or {}
    topic = ast.literal_eval(topic) if isinstance(topic, str) else topic
    names = [str(name or eid) for eid, name in topic.items()]
    nodes = {n for h, _, t in sub.get("graph") or [] for n in (h, t)}
    theirs = list(sub.get("q_entity") or [])
    lined_up = len(theirs) == len(names) and all(t == n for n, t in zip(names, theirs) if n in nodes)
    found, absent, mapped = [], [], {}
    for i, name in enumerate(names):
        if name in nodes:
            found.append(name)
        elif lined_up and theirs[i] in nodes:
            found.append(theirs[i])
            mapped[name] = theirs[i]
        else:
            absent.append(name)
    return list(dict.fromkeys(found)), list(dict.fromkeys(absent)), mapped
