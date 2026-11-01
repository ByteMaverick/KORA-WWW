# Gaps between the result CSVs and the submitted paper

Every cell `make_tables.py` prints for Tables 1–5 and 7 was compared with the paper, together with the labels the
figures print (Figures 5–7) and the numbers quoted in the text of Sections 5.2–5.5 and Appendices A.2 and A.4. No CSV
value was changed to match the paper; every difference is listed below.

| Compared | Cells matching / compared |
|---|---|
| Table 1 | 117 / 117 |
| Table 2 | 60 / 72 |
| Table 3 | 44 / 48 |
| Table 4 | 278 / 288 |
| Table 5 | 48 / 48 |
| Table 7 | 14 / 14 |
| Figure 5 | 22 / 22 |
| Figure 6 | 6 / 6 |
| Figure 7 | 54 / 66 |
| Text | 40 / 43 |

All Hits@Any, F1 and LLM-cost cells match. Every difference is a graph-trace support value (KORA's and CoR's on the
Unseen pairs, and ToG's with gpt-oss-120B) or a Figure 7 segment that follows from a support value.

What was compared. Figure 5: the share labels printed on the bars (segments of about 10% and more; an unlabelled
segment is compared as unlabelled) and the two token ratios printed on CWQ. Figure 6: the LLM-calls label of each
system; its Hits@Any, token and latency values quoted in Appendix A.4 are under Text. Figure 7: the segment labels
(segments of 7% and more) and the support printed right of each bar. Positions of points, bar lengths and error bars
were not measured.

## Differences

| Table | Row | Column | CSV | Paper | Likely reason |
|---|---|---|---|---|---|
| Table 2 | CoR / Native | WebQSP-Unseen Supp. | 73.7 | 82.0 | Paper value from a later re-scoring of support (note B) |
| Table 2 | CoR / Native | CWQ-Unseen Supp. | 79.8 | 89.5 | Paper value from a later re-scoring of support (note B) |
| Table 2 | CoR / Estimator | WebQSP-Unseen Supp. | 94.0 | 96.0 | Paper value from a later re-scoring of support (note B) |
| Table 2 | CoR / Estimator | CWQ-Unseen Supp. | 95.9 | 97.6 | Paper value from a later re-scoring of support (note B) |
| Table 2 | CoR / Gold | WebQSP-Unseen Supp. | 90.0 | 93.7 | Paper value from a later re-scoring of support (note B) |
| Table 2 | CoR / Gold | CWQ-Unseen Supp. | 88.2 | 95.1 | Paper value from a later re-scoring of support (note B) |
| Table 2 | KORA / LLM | WebQSP-Unseen Supp. | 98.5 | 100.0 | Paper value from a later re-scoring of support (note A) |
| Table 2 | KORA / LLM | CWQ-Unseen Supp. | 98.0 | 100.0 | Paper value from a later re-scoring of support (note A) |
| Table 2 | KORA / Estimator | WebQSP-Unseen Supp. | 98.5 | 100.0 | Paper value from a later re-scoring of support (note A) |
| Table 2 | KORA / Estimator | CWQ-Unseen Supp. | 99.0 | 100.0 | Paper value from a later re-scoring of support (note A) |
| Table 2 | KORA / Gold | WebQSP-Unseen Supp. | 98.8 | 100.0 | Paper value from a later re-scoring of support (note A) |
| Table 2 | KORA / Gold | CWQ-Unseen Supp. | 98.4 | 100.0 | Paper value from a later re-scoring of support (note A) |
| Table 3 | CoR | WebQSP-Unseen Supp. | 74.6 | 81.4 | Paper value from a later re-scoring of support (note B) |
| Table 3 | CoR | CWQ-Unseen Supp. | 78.2 | 87.9 | Paper value from a later re-scoring of support (note B) |
| Table 3 | KORA | WebQSP-Unseen Supp. | 98.6 | 100.0 | Paper value from a later re-scoring of support (note A) |
| Table 3 | KORA | CWQ-Unseen Supp. | 98.5 | 100.0 | Paper value from a later re-scoring of support (note A) |
| Table 4 | Qwen3-4B / CoR | WebQSP-Unseen Supp. | 81.2 | 88.2 | Paper value from a later re-scoring of support (note B) |
| Table 4 | Qwen3-4B / CoR | CWQ-Unseen Supp. | 80.8 | 92.5 | Paper value from a later re-scoring of support (note B) |
| Table 4 | Qwen3-4B / KORA (ours) | WebQSP-Unseen Supp. | 99.4 | 100.0 | Paper value from a later re-scoring of support (note A) |
| Table 4 | Qwen3-4B / KORA (ours) | CWQ-Unseen Supp. | 98.7 | 100.0 | Paper value from a later re-scoring of support (note A) |
| Table 4 | gpt-oss-120B / ToG | WebQSP-Unseen Supp. | 0.0 | –‡ | The paper omits it: most ToG runs with gpt-oss-120B logged no graph calls; the CSV keeps the computed value (note D) |
| Table 4 | gpt-oss-120B / ToG | CWQ-Unseen Supp. | 0.1 | –‡ | The paper omits it: most ToG runs with gpt-oss-120B logged no graph calls; the CSV keeps the computed value (note D) |
| Table 4 | gpt-oss-120B / CoR | WebQSP-Unseen Supp. | 87.5 | 91.1 | Paper value from a later re-scoring of support (note B) |
| Table 4 | gpt-oss-120B / CoR | CWQ-Unseen Supp. | 80.5 | 83.3 | Paper value from a later re-scoring of support (note B) |
| Table 4 | gpt-oss-120B / KORA (ours) | WebQSP-Unseen Supp. | 98.3 | 100.0 | Paper value from a later re-scoring of support (note A) |
| Table 4 | gpt-oss-120B / KORA (ours) | CWQ-Unseen Supp. | 98.5 | 100.0 | Paper value from a later re-scoring of support (note A) |
| Figure 7 | WebQSP-Unseen / KORA | Correct, supported | 79 | 80 | Follows from the support split (note C): KORA's correct total is 79.6 in both |
| Figure 7 | WebQSP-Unseen / KORA | support % | 98.6 | 100.0 | Paper value from a later re-scoring of support (note A) |
| Figure 7 | WebQSP-Unseen / CoR | Correct, supported | 36 | 40 | Follows from CoR's support split (note C); its correct and wrong totals agree |
| Figure 7 | WebQSP-Unseen / CoR | Wrong, supported | 20 | 22 | Follows from CoR's support split (note C); its correct and wrong totals agree |
| Figure 7 | WebQSP-Unseen / CoR | Wrong, not fully supported | 17 | 15 | Follows from CoR's support split (note C); its correct and wrong totals agree |
| Figure 7 | WebQSP-Unseen / CoR | support % | 74.6 | 81.4 | Paper value from a later re-scoring of support (note B) |
| Figure 7 | CWQ-Unseen / KORA | Correct, supported | 64 | 65 | Follows from the support split (note C); with every KORA item supported the segment would be 64.3, so the printed 65 is not explained |
| Figure 7 | CWQ-Unseen / KORA | support % | 98.5 | 100.0 | Paper value from a later re-scoring of support (note A) |
| Figure 7 | CWQ-Unseen / CoR | Correct, supported | 26 | 30 | Follows from CoR's support split (note C); its correct and wrong totals agree |
| Figure 7 | CWQ-Unseen / CoR | Wrong, supported | 32 | 37 | Follows from CoR's support split (note C); its correct and wrong totals agree |
| Figure 7 | CWQ-Unseen / CoR | Wrong, not fully supported | 14 | 9 | Follows from CoR's support split (note C); its correct and wrong totals agree |
| Figure 7 | CWQ-Unseen / CoR | support % | 78.2 | 87.9 | Paper value from a later re-scoring of support (note B) |
| Text | Section 5.2 | KORA graph-trace support (%) | 98.8 | 100 | No CSV has support on the standard sets; 98.8 is the mean of KORA's source-half support (98.6, 99.0) in table3_unseen.csv, where KB-BINDER's 96.2 and SoG's 85.4 reproduce (note E) |
| Text | Section 5.4 | KORA support on webqsp-unseen (%) | 98.6 | 100 | Paper value from a later re-scoring of support (note A) |
| Text | Section 5.4 | KORA support on cwq-unseen (%) | 98.5 | 100 | Paper value from a later re-scoring of support (note A) |

## Notes

**A. KORA's support.** The paper reports 100.0 for KORA, from a later re-scoring of the runs' graph-call logs under the
rule of Appendix B.3. The CSVs hold the earlier scoring of the same runs, which left 1–2% of KORA's answer items
untraced: items reached through a second question entity that KORA looked up by name (the earlier scoring rooted a
chain only at the dataset's topic entities, while B.3 also roots it at a question entity looked up by the system),
items printed without a graph identifier, and items missing from the kept graph-call log. The released scorer
(`harness/metrics/groundedness/grounding.py`) applies the B.3 rule, including the name-looked-up roots; it has not been rerun on
these logs, so neither value is confirmed by it.

**B. CoR's support.** The paper's CoR support values also come from a later re-scoring of the graph-call logs; the
CSVs hold the earlier scoring. CoR's Hits@Any agrees in every cell, so only the support values differ.

**C. Figure 7 segments.** The figure splits each system's correct and wrong answers by support, so the KORA and CoR
segments move with notes A and B. Their correct and wrong totals agree with the CSV.

**D. ToG with gpt-oss-120B.** Table 4 prints no support for ToG with gpt-oss-120B, because most of those runs logged
no graph calls (Table 4's note). The CSV keeps the value the rule gives (0.0 and 0.1).

**E. Support in Section 5.2.** The sentence "100% graph-trace supported answers, compared with 96.2% for KB-BINDER
and 85.4% for SoG" has no matching column in `table1_standard.csv`. The two baseline values reproduce as support on
the source halves of the Unseen pairs (`table3_unseen.csv`: KB-BINDER 96.2 on WebQSP-Unseen; SoG 92.7 and 78.1,
mean 85.4); on the same basis KORA's is 98.6 and 99.0 (mean 98.8), which note A covers.

## Not covered by these CSVs

- Appendix C (the Stopping Estimator's threshold, its validation rates and AUROC, the renaming false-negative rates and
  the Unseen Hits@Any of the estimator trained without renamed copies) and the memory result of Appendix A.2 have no
  CSV in this release.
- Table 2 and Figure 6 are on subsets of the Unseen test pairs (500 pairs per set; 200 twins, 100 per set). The id
  lists of those subsets are not in `data/`.
- MetaQA: the paper's appendix states 517 questions; the evaluated set is 515 (two questions that also appear in the
  benchmark's training or validation split are excluded), and `data/questions/metaqa/test.json` holds those 515.
- Figure 5's route split and Figure 7's support-by-correctness split are not produced by `python -m harness score`,
  which reports a run's support as one number.
