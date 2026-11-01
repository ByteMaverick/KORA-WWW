"""Print the paper's tables and figure data from the CSVs in this folder.

    python results/make_tables.py              every table and figure
    python results/make_tables.py table1 fig7  only those

Standard library only. Percentages are printed with one decimal, token counts in thousands, as in the paper.
"""
from __future__ import annotations

import csv
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DASH = "–"
B30, B4, B120 = "Qwen3-30B-A3B-Instruct-2507", "Qwen3-4B", "gpt-oss-120B"
SHORT = {B30: "Qwen3-30B", B4: "Qwen3-4B", B120: "gpt-oss-120B"}
STD = [("webqsp", "WebQSP"), ("cwq", "CWQ"), ("grailqa", "GrailQA"), ("metaqa", "MetaQA")]
UNSEEN = [("webqsp", "WebQSP-Unseen"), ("cwq", "CWQ-Unseen")]
METHODS = ["Direct LLM", "KB-BINDER", "ToG", "PoG", "SoG", "CoR", "GCR", "KG-R1", "KORA"]


def load(name: str) -> list[dict]:
    with open(os.path.join(HERE, name), newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def index(rows: list[dict]) -> dict:
    return {(r.get("backbone", ""), r["dataset"], r["system"]): r for r in rows}


def pct(row: dict | None, key: str) -> str:
    if row is None or row.get("status", "ok") != "ok" or row.get(key, "") == "":
        return DASH
    return f"{float(row[key]):.1f}"


def ktok(value: float | None) -> str:
    if value is None:
        return DASH
    return "<0.1K" if value < 50 else f"{value / 1000:.1f}K"


def label(system: str) -> str:
    return "KORA (ours)" if system == "KORA" else system


def pooled(rows: list[dict], key: str) -> float | None:
    """The per-question mean over every question of the sets a system ran on (weighted by each set's n)."""
    have = [r for r in rows if r.get("status") == "ok" and r.get(key, "") != ""]
    if not have:
        return None
    return sum(int(r["n"]) * float(r[key]) for r in have) / sum(int(r["n"]) for r in have)


# ---------------------------------------------------------------- tables
def table1() -> dict:
    rows = load("table1_standard.csv")
    ix = index(rows)
    header = ["Method"] + [f"{t} {m}" for _, t in STD + [("mean", "Mean")] for m in ("Hit", "F1")] + \
             ["In", "Out", "Calls"]
    out = []
    for s in METHODS:
        cells = [label(s)]
        for ds, _ in STD + [("mean", "Mean")]:
            r = ix.get((B30, ds, s))
            cells += [pct(r, "hit"), pct(r, "f1")]
        mine = [r for r in rows if r["system"] == s and r["dataset"] != "mean"]
        calls = pooled(mine, "llm_calls")
        cells += [ktok(pooled(mine, "input_tokens")), ktok(pooled(mine, "output_tokens")),
                  DASH if calls is None else f"{calls:.1f}"]
        out.append(cells)
    return {"key": "table1", "title": "Table 1: Results on standard KGQA benchmarks (Qwen3-30B).",
            "header": header, "blocks": [(None, out)]}


STOP_LABEL = {"native": "Native", "estimator": "Estimator", "gold": "Gold", "llm": "LLM"}


def table2() -> dict:
    rows = load("table2_transfer.csv")
    ix = index(rows)
    header = ["Agent", "Stop"] + [f"{t} {c}" for _, t in UNSEEN for c in ("Source", "Unseen", "Supp.")]
    out = []
    systems = []
    for r in rows:
        if r["system"] not in systems:
            systems.append(r["system"])
    for s in systems:
        stop = next(r["stop"] for r in rows if r["system"] == s)
        cells = [s.split(" (")[0], STOP_LABEL[stop]]
        for ds, _ in UNSEEN:
            src, uns = ix.get((B30, f"{ds}-unseen-real", s)), ix.get((B30, f"{ds}-unseen", s))
            cells += [pct(src, "hit"), pct(uns, "hit"), pct(uns, "support")]
        out.append(cells)
    return {"key": "table2", "title": "Table 2: Learned stopping within KORA and transferred to other agents "
            "(Qwen3-30B, 500 pairs per set; Supp.: support on the unseen half).", "header": header,
            "blocks": [(None, out)]}


def unseen_cells(ix: dict, backbone: str, system: str) -> list[str]:
    cells = []
    for ds, _ in UNSEEN:
        src, uns = ix.get((backbone, f"{ds}-unseen-real", system)), ix.get((backbone, f"{ds}-unseen", system))
        cells += [pct(src, "hit"), pct(uns, "hit"), pct(uns, "support")]
    return cells


def table3() -> dict:
    ix = index(load("table3_unseen.csv"))
    header = ["Method"] + [f"{t} {c}" for _, t in UNSEEN for c in ("Source", "Unseen", "Supp.")]
    out = [[s] + unseen_cells(ix, B30, s) for s in METHODS if s != "GCR"]
    return {"key": "table3", "title": "Table 3: Performance on newly added knowledge (Qwen3-30B, 1,000 pairs per "
            "set; Supp.: support on the unseen half).", "header": header, "blocks": [(None, out)]}


def table4() -> dict:
    ix = index(load("table4_backbones.csv"))
    header = ["Method"] + [f"{t} {m}" for _, t in STD + [("mean", "Mean")] for m in ("Hit", "F1")] + \
             [f"{t} {c}" for _, t in UNSEEN for c in ("Src", "Uns", "Supp.")]
    blocks = []
    for b in (B4, B120):
        out = []
        for s in METHODS:
            cells = [label(s)]
            for ds, _ in STD + [("mean", "Mean")]:
                r = ix.get((b, ds, s))
                cells += [pct(r, "hit"), pct(r, "f1")]
            out.append(cells + unseen_cells(ix, b, s))
        blocks.append((SHORT[b], out))
    return {"key": "table4", "title": "Table 4: Results with Qwen3-4B and gpt-oss-120B on the standard benchmarks "
            "and on newly added knowledge.", "header": header, "blocks": blocks}


def table5() -> dict:
    ix = index(load("table5_stop_rules.csv"))
    header = ["Stop"] + [t for _, t in STD] + ["Mean Hit", "Mean F1"] + [f"Unseen {t}" for _, t in STD[:2]]
    blocks = []
    for b in (B30, B4):
        out = []
        for s, name in (("KORA (LLM stop)", "LLM stop"), ("KORA", "Estimator"), ("KORA (gold stop)", "Gold")):
            cells = [name] + [pct(ix.get((b, ds, s)), "hit") for ds, _ in STD]
            m = ix.get((b, "mean", s))
            cells += [pct(m, "hit"), pct(m, "f1")]
            cells += [pct(ix.get((b, f"{ds}-unseen", s)), "hit") for ds, _ in UNSEEN]
            out.append(cells)
        blocks.append((SHORT[b], out))
    return {"key": "table5", "title": "Table 5: KORA with each stopping rule (Hits@Any, %).", "header": header,
            "blocks": blocks}


def table7() -> dict:
    out = [[r["check"], r["webqsp"], r["cwq"]] for r in load("table7_unseen_validation.csv")]
    return {"key": "table7", "title": "Table 7: Validation of WebQSP-Unseen and CWQ-Unseen (released pairs).",
            "header": ["Check", "WebQSP", "CWQ"], "blocks": [(None, out)]}


# ---------------------------------------------------------------- figures
ROUTES = ["Direct query", "Neighborhood search", "Decomposed, accepted", "Decomposed, not accepted",
          "Not decomposed, not accepted"]


def fig5() -> dict:
    rows = load("fig5_routes.csv")
    cell = {(r["dataset"], r["route"]): r for r in rows}
    shares, tokens, derived = [], [], []
    for ds, title in STD:
        shares.append([title] + [f"{float(cell[(ds, x)]['pct']):.1f}" for x in ROUTES])
        tokens.append([title] + [ktok(float(cell[(ds, x)]["tokens"])) for x in ROUTES])
        n = int(cell[(ds, ROUTES[0])]["n"])
        dec = sum(int(cell[(ds, x)]["count"]) for x in ROUTES[2:4])
        direct = float(cell[(ds, "Direct query")]["tokens"])
        derived.append([title, f"{100 * dec / n:.1f}",
                        f"{float(cell[(ds, 'Neighborhood search')]['tokens']) / direct:.1f}×",
                        f"{float(cell[(ds, 'Decomposed, accepted')]['tokens']) / direct:.1f}×"])
    return {"key": "fig5", "title": "Figure 5: How KORA settled each question on the full test sets (Qwen3-30B).",
            "header": None, "blocks": [
                ("(a) Questions (%)", [["Dataset"] + ROUTES] + shares),
                ("(b) LLM tokens per question", [["Dataset"] + ROUTES] + tokens),
                ("Derived", [["Dataset", "Decomposed (%)", "Neighborhood / direct tokens",
                              "Decomposed, accepted / direct tokens"]] + derived)]}


def fig6() -> dict:
    rows = load("fig6_latency.csv")
    out = []
    order = []
    for r in rows:
        if r["system"] not in order:
            order.append(r["system"])
    sogs = [float(r["tokens"]) for r in rows if r["system"] == "SoG"]
    for s in order:
        mine = [r for r in rows if r["system"] == s]

        def ms(key, digits=1, scale=1.0):
            v = [float(r[key]) / scale for r in mine]
            return f"{statistics.mean(v):.{digits}f} ± {statistics.stdev(v):.{digits}f}"
        tok = statistics.mean(float(r["tokens"]) for r in mine)
        out.append([s, str(len(mine)), ms("hit"), ms("tokens", 1, 1000.0) + "K",
                    f"{statistics.mean(float(r['llm_calls']) for r in mine):.1f}", ms("latency_median_s"),
                    f"{statistics.mean(float(r['latency_p90_s']) for r in mine):.1f}",
                    f"{100 * (1 - tok / statistics.mean(sogs)):.0f}%"])
    header = ["System", "Runs", "Hits@Any (mean ± s.d.)", "LLM tokens/question (mean ± s.d.)",
              "LLM calls/question", "Median latency s (mean ± s.d.)", "p90 latency s (mean)",
              "Tokens below SoG"]
    return {"key": "fig6", "title": "Figure 6: Cost and latency under controlled conditions (Qwen3-30B, 200 unseen "
            "questions, three runs).", "header": header, "blocks": [(None, out)]}


F7_PARTS = ["correct_supported", "correct_not_fully_supported", "wrong_supported", "wrong_not_fully_supported",
            "no_answer"]


def fig7() -> dict:
    rows = load("fig7_support.csv")
    header = ["System", "Correct, supported", "Correct, not fully supported", "Wrong, supported",
              "Wrong, not fully supported", "No answer / error / timeout", "support %"]
    blocks = []
    for ds, title in UNSEEN:
        out = []
        for s in ["KORA", "SoG", "KB-BINDER", "CoR", "PoG", "ToG"]:
            r = next((x for x in rows if x["dataset"] == f"{ds}-unseen" and x["system"] == s), None)
            if r is None:
                out.append([s] + ["n/a"] + [""] * 5)
                continue
            n = int(r["n"])
            out.append([s] + [f"{100 * int(r[p]) / n:.1f}" for p in F7_PARTS] + [pct(r, "support")])
        blocks.append((title, out))
    return {"key": "fig7", "title": "Figure 7: Correctness and graph-trace support on the unseen halves (Qwen3-30B); "
            "% of questions.", "header": header, "blocks": blocks}


def paired() -> dict:
    out = [[SHORT[r["backbone"]], r["dataset"], r["metric"], r["comparison"], r["difference"],
            f"[{r['ci_lo']}, {r['ci_hi']}]", r["n"]] for r in load("paired_comparisons.csv")]
    return {"key": "paired", "title": "Paired differences quoted in Section 5 (95% paired bootstrap, 10,000 "
            "resamples).", "header": ["Backbone", "Set", "Metric", "Comparison", "Difference", "95% CI", "n"],
            "blocks": [(None, out)]}


BUILDERS = [table1, table2, table3, table4, table5, table7, fig5, fig6, fig7, paired]


def tables(keys: list[str] | None = None) -> list[dict]:
    names = {b.__name__: b for b in BUILDERS}
    unknown = [k for k in keys or [] if k not in names]
    if unknown:
        raise SystemExit(f"unknown table {', '.join(unknown)}: one of {', '.join(names)}")
    return [b() for b in BUILDERS if not keys or b.__name__ in keys]


def render(t: dict) -> str:
    lines = [t["title"], ""]
    for name, rows in t["blocks"]:
        grid = ([t["header"]] if t["header"] else []) + rows
        widths = [max(len(r[i]) for r in grid if i < len(r)) for i in range(max(len(r) for r in grid))]
        if name:
            lines.append(f"[{name}]")
        for k, r in enumerate(grid):
            lines.append("  ".join(c.ljust(widths[i]) if i == 0 else c.rjust(widths[i]) for i, c in enumerate(r)))
            if k == 0:
                lines.append("  ".join("-" * w for w in widths))
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    for t in tables(argv):
        print(render(t))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
