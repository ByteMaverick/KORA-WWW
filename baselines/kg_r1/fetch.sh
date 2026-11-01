#!/bin/bash
# KG-R1's code, checkpoint and Freebase data, one step at a time (KGR1 defaults to $DATA_ROOT/kg_r1):
#   baselines/kg_r1/fetch.sh code         the authors' repository at the pinned commit -> $KGR1/KG-R1
#   baselines/kg_r1/fetch.sh checkpoint   SCL2025/KG-R1-CQW at the pinned revision, the bf16 files only, one file at a
#                                         time with curl (resumable), each checked against the hub's sha256
#   baselines/kg_r1/fetch.sh freebase     the authors' CWQ and WebQSP subgraphs (their scripts/download_kg.py, Google
#                                         Drive) and the Freebase RDF dump their entity names come from
# The pins are baselines/kg_r1/upstream.py's.
set -u
[ -n "${KGR1:-}${DATA_ROOT:-}" ] || { echo "set DATA_ROOT (or KGR1)" >&2; exit 2; }
KGR1="${KGR1:-$DATA_ROOT/kg_r1}"
REPO=https://github.com/junhongmit/KG-R1
COMMIT=a3c02facbb3f4a905c9cf24707867d1df0746595
CHECKPOINT=SCL2025/KG-R1-CQW
REVISION=fa505b2e330843dbeebef7dd695141ebfc1af764
FILES="config.json generation_config.json model.safetensors.index.json model-00001-of-00002.safetensors
model-00002-of-00002.safetensors tokenizer.json tokenizer_config.json vocab.json merges.txt added_tokens.json
special_tokens_map.json"

code() {
  local dest="$KGR1/KG-R1"
  [ -d "$dest/.git" ] || git clone -q "$REPO" "$dest" || return 1
  git -C "$dest" fetch -q origin && git -C "$dest" checkout -q "$COMMIT" || return 1
  [ "$(git -C "$dest" rev-parse HEAD)" = "$COMMIT" ] && echo "ok code $dest @ $COMMIT"
}

checkpoint() {
  local dest="$KGR1/hf/KG-R1-CQW" api="https://huggingface.co/api/models/$CHECKPOINT/revision/$REVISION?blobs=true"
  mkdir -p "$dest"
  curl -s "$api" > "$dest/.api.json" || return 1
  for f in $FILES; do
    curl -sL -C - --retry 5 -o "$dest/$f" "https://huggingface.co/$CHECKPOINT/resolve/$REVISION/$f" || { echo "FAIL $f"; continue; }
    python3 - "$dest/.api.json" "$dest/$f" "$f" <<'EOF'
import hashlib, json, os, sys
api, path, name = sys.argv[1:]
blob = next(s for s in json.load(open(api))["siblings"] if s["rfilename"] == name)
want = (blob.get("lfs") or {}).get("sha256")
if want:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    ok = h.hexdigest() == want
else:                                   # a small file kept in git: its size is the check
    ok = os.path.getsize(path) == blob.get("size")
print(("ok " if ok else "MISMATCH ") + name)
sys.exit(0 if ok else 1)
EOF
  done
  echo "$CHECKPOINT@$REVISION" > "$dest/REVISION"
}

freebase() {
  # Needs `fetch.sh code` first and the kg_server environment (requests). The dump is ~30 GB; its names are extracted
  # in a separate step (UPSTREAM.md, step 4).
  local clone="$KGR1/KG-R1"
  (cd "$clone" && python scripts/download_kg.py --save_path data_kg --datasets CWQ webqsp) || return 1
  [ -f "$KGR1/freebase-rdf-latest.gz" ] || wget -c -O "$KGR1/freebase-rdf-latest.gz" \
    http://commondatastorage.googleapis.com/freebase-public/rdf/freebase-rdf-latest.gz
}

case "${1:-}" in
  code) code ;;
  checkpoint) checkpoint ;;
  freebase) freebase ;;
  *) echo "usage: $0 code|checkpoint|freebase" >&2; exit 2 ;;
esac
