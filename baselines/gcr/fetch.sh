#!/bin/bash
# GCR's weights and RoG's CWQ / WebQSP data from Hugging Face into $GCR/hf/, one file at a time with curl
# (resumable). GCR defaults to $DATA_ROOT/gcr.
set -u
[ -n "${GCR:-}${DATA_ROOT:-}" ] || { echo "set DATA_ROOT (or GCR)" >&2; exit 2; }
GCR="${GCR:-$DATA_ROOT/gcr}"
fetch() {  # fetch <model|dataset> <repo>
  local kind=$1 repo=$2 api base dest
  if [ "$kind" = model ]; then api="https://huggingface.co/api/models/$repo"; base="https://huggingface.co/$repo/resolve/main"
  else api="https://huggingface.co/api/datasets/$repo"; base="https://huggingface.co/datasets/$repo/resolve/main"; fi
  dest="$GCR/hf/${repo#*/}"
  for f in $(curl -s "$api" | python3 -c "import sys,json; print(\"\n\".join(s[\"rfilename\"] for s in json.load(sys.stdin)[\"siblings\"]))"); do
    mkdir -p "$dest/$(dirname "$f")"
    curl -sL -C - --retry 5 -o "$dest/$f" "$base/$f" && echo "ok $repo/$f $(du -h "$dest/$f" | cut -f1)" || echo "FAIL $repo/$f"
  done
}
fetch model rmanluo/GCR-Meta-Llama-3.1-8B-Instruct
fetch dataset rmanluo/RoG-webqsp
fetch dataset rmanluo/RoG-cwq
echo ALL-DONE
