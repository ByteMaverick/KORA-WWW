#!/usr/bin/env bash
# KB-BINDER's data from its authors' Google Drive (the README's links), into $KBB_DATA. Needs gdown (pip install
# gdown). The 2.5 GB surface map is not fetched: the harness links entities with the gold topic entities (run.py).
set -euo pipefail
OUT="${KBB_DATA:-$(dirname "$0")/data}"
mkdir -p "$OUT/metaQA" "$OUT/contriever"
get() { [ -s "$OUT/$2" ] || gdown -q "https://drive.google.com/uc?id=$1" -O "$OUT/$2"; }
get 1cVHkU2h451fVV6reUZjUvewi0CmxxiZc fb_roles
get 1_4Iaa93JY-EObaucok4Nu7piBk4C2CEh grailqa_v1.0_train.json
get 1hnyW-_k0YaAUZDTdYzhbKDTnFuLEW-W2 contriever/index          # the README's "index file" (Contriever, Faiss)
get 1GqCljSvAVbqakvLnDzI-K--th5ukPHnl metaQA/kb.txt
get 15WfAiMgTsaOjHBln0xeV2w8dyhsbdPpW metaQA/kb_entity_dict.txt
get 16ZhYJRWRYUVmSqkMijD6TwKM41mt3hrC metaQA/qa_test_1hop.txt
get 1HbbPjA6NsqmiO19sz0W91lYvHWcOCGW1 metaQA/qa_test_2hop.txt
get 1Qt0yp2bCu4C0QMJc7wM2Xz20lMdiAYuG metaQA/qa_test_3hop.txt
get 1fFjl7Y-kerPjmULNWG_qwc-dtPcyPg8F metaQA/qa_test_qtype_1hop.txt
get 1K4y6jojAz2ni3j2ZguWJiy8L5z6wULRE metaQA/qa_test_qtype_2hop.txt
get 10QJsQkaOWQyR6UWCcbRPLvG9c3-FNGQb metaQA/qa_test_qtype_3hop.txt
get 15ilE8igbCJjJsKfSRsbB3gibSqLagdst metaQA/qa_train_1hop.txt
get 1wNUB49_3YPWH41J_XcOHpBSUMaopUSee metaQA/qa_train_2hop.txt
get 16DGiiOoMSfX03JViYcgNKQcdE5Ks8AZJ metaQA/qa_train_3hop.txt
get 1SL-q6jck5an-WZ4NAUEGEi1xtTJZtVe6 metaQA/qa_train_qtype_1hop.txt
get 1KzOZr26zdS_GDveq0cFokjwbQ5gkrqLL metaQA/qa_train_qtype_2hop.txt
get 1jgQ1jlXJZFSqdK_ZK3Anr6-7rBxkuOYS metaQA/qa_train_qtype_3hop.txt
get 1Zb_O5ylEnq95pzxcfgUVZkgzkOVMT39I webqsp_0107.train.json
