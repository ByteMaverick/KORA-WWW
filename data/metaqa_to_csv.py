"""MetaQA's kb.txt as CSV for `neo4j-admin database import`.

    python data/metaqa_to_csv.py KB_TXT OUT_DIR

kb.txt is one `subject|relation|object` triple per line. MetaQA has no identifiers, so every distinct string
becomes one node whose `id` and `name` are that string, and every line one relationship whose type is the relation
upper-cased (`directed_by` -> `DIRECTED_BY`). Writes OUT_DIR/entity_nodes.csv (sorted) and OUT_DIR/kb_rels.csv
(in file order). Lines that are not three non-empty fields are skipped.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path


def convert(kb: Path, out: Path) -> tuple[int, int]:
    entities: set[str] = set()
    triples: list[tuple[str, str, str]] = []
    with kb.open(encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("|")
            if len(parts) != 3 or not all(parts):
                continue
            subject, relation, obj = parts
            entities.update((subject, obj))
            triples.append((subject, relation.upper(), obj))

    out.mkdir(parents=True, exist_ok=True)
    with (out / "entity_nodes.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id:ID", "name"])
        writer.writerows([e, e] for e in sorted(entities))
    with (out / "kb_rels.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([":START_ID", ":TYPE", ":END_ID"])
        writer.writerows(triples)
    return len(entities), len(triples)


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] in ("-h", "--help"):
        sys.exit(print("usage: python data/metaqa_to_csv.py KB_TXT OUT_DIR"))
    if len(sys.argv) != 3:
        sys.exit("usage: python data/metaqa_to_csv.py KB_TXT OUT_DIR")
    nodes, rels = convert(Path(sys.argv[1]), Path(sys.argv[2]))
    print(f"{nodes} nodes, {rels} relationships")
