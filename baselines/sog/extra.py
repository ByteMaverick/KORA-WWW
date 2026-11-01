"""
sog/extra.py
-------------
What SoG needs for the graph the authors did not run (MetaQA), and nothing else:

  * the tool spec: the authors' TOOLS_FREEBASE copied, with the graph's name and an id of its own in the
    descriptions (as the authors did for Wikidata, TOOLS_WIKIDATA);
  * the five exemplars, built from TRAIN questions by build_examples.py (examples_extra.json);
  * the per-dataset instructions: the authors' CWQ text, which names no graph.
"""
from __future__ import annotations

import copy
import json
import os

from baselines.sog import upstream_path  # noqa: F401
import prompt as sog_prompt

HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS = {"metaqa": "MetaQA"}

_GRAPH = {   # dataset -> (graph name in the tool description, an entity id, two properties)
    "MetaQA": ("the MetaQA movie knowledge graph", "Inception", ["DIRECTED_BY", "STARRED_ACTORS"]),
}


def tools(dataset: str) -> list:
    graph, eid, props = _GRAPH[dataset]
    spec = copy.deepcopy(sog_prompt.TOOLS_FREEBASE)
    fn = spec[0]["function"]
    fn["description"] = fn["description"].replace("on Freebase", f"on {graph}")
    params = fn["parameters"]["properties"]
    params["entity"]["description"] = params["entity"]["description"].replace("'m.04yd0fh'", f"'{eid}'")
    params["properties_to_filter_for"]["description"] = params["properties_to_filter_for"]["description"].replace(
        "['people.person.place_of_birth', 'people.person.nationality']", repr(props))
    return spec


def examples(dataset: str) -> list[str]:
    key = {v: k for k, v in DATASETS.items()}[dataset]
    found = json.load(open(os.path.join(HERE, "examples_extra.json")))[key]
    return [e["text"] for e in found]


def instructions(dataset: str) -> str:
    return sog_prompt.prompt["additional"]["CWQ"]
