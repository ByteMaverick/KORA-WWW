# Synthesis

A finished run's execution provenance becomes the answer: the entities the root stands
behind, projected as the plan says, named, with the derivation that produced
them and the constraints nothing established. It runs once, after the
controller has finished. It never queries, plans, judges or repairs, and it
writes nothing to the state.

Two files. `synthesis.py` is the draft and the code that builds it, with no
model call; `phrase.py` is the one optional model call, which the harness never
makes.

## What is built

| rule | what `synthesise` does |
|---|---|
| entities | `state.produced_by("T0")`: already composed, already narrowed when the recipe closed with an operate |
| projection | `plan.answer.projection`, `identity` without a plan. `count` is one `value` entity holding the size, unless the set is already the one `COUNT` row; `argmax`, `argmin` and `identity` leave the set as it is |
| boolean | `plan.answer.type == "boolean"`: when the root holds the single `yes` or `no` a `verify` operate produced (`reduction.YES`, `NO`), that is `text`: the graph's values decide, not a model. Otherwise `text` is `no` if a root claim was judged NOT, `yes` if every root claim is ESTABLISHED and the set is non-empty when the plan declares a variable, otherwise `unknown`; `entities` is empty. The root's claims are the last judgement recorded on the root |
| kind, count | copied from `plan.answer` |
| text | the names joined, the number for a count, `yes` / `no` / `unknown` for a boolean |
| derivation | every task that produced or settled, in creation order (a task is created after the tasks it waits on); an operate task is a step whose goal is its op |
| established, unmet | the contract's requirement ids, resolved and not |

`phrase(draft, llm)` asks `synthesis.phrase` for one sentence and the names it
used. A reply that uses a name the draft does not hold, or has no sentence, or
is not JSON, is dropped and the draft comes back as it was; a kept reply
replaces `text` and nothing else. It is given the draft, never the state.

Callers: the run path uses `apply` (the state's projection of the root's set);
`synthesise` and `phrase` are the library entry points. The oracle judges read
whether a gold is a count from the task view's `projection`, which copies
`plan.answer.projection`.

## API

```python
synthesise(state) -> AnswerDraft
phrase(draft: AnswerDraft, llm) -> AnswerDraft
apply(projection: str, entities: list[EntityRef]) -> list[EntityRef]

AnswerDraft: entities, names, text, projection, kind, count,
             established: list[str], unmet: list[str],
             derivation: list[Step], evidence_ids: list[str]
Step:        task_id, goal, anchors, produced, settled, after
```

```python
from kora.modules.synthesis import phrase, synthesise

draft = synthesise(state)          # after the controller has finished
draft.text                         # "Denmark" | "9" | "yes"
draft.entities                     # [EntityRef(id="m.0k9p4", label="Denmark")]
draft.unmet                        # ["c2"]: shown as "not established"
[s.task_id for s in draft.derivation]   # ["T0", "T0.1", "T0.2"]

sentence = phrase(draft, llm).text # the harness never calls it
```
