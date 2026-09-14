# Analysis results obtained so far

Snapshot date: 2026-09-14

## Final requested DirectProbe and GED results

The requested DirectProbe and paper-compatible legacy GED computations are
complete. The publishable result package deliberately contains the compact raw
solver outputs, per-program GED arrays, summaries, configurations, and
provenance manifests. Multi-gigabyte embedding matrices, extracted attention
pickle files, temporary GED shards, progress logs, and interrupted-run
archives are reproducible intermediates and are not published in Git.

DirectProbe covers CodeBERT, GraphCodeBERT, and CodeT5 for Java, Go, and
JavaScript at hidden states 5, 9, and 12. All five tasks (`siblings`,
`siblings_id`, `dfg`, `distance`, and `distance_id`) are complete: 135/135
configurations. The canonical files are under
`DirectProbe/final_3000/results/<language>/<task>/<model>/<layer>/`; each
configuration contains `clusters.txt`, `prediction.txt`, `dis.txt`, and
`log.txt`. The final queue and watchdog manifests are stored beside that
directory, and every solver log records Gurobi use.

GED covers the same three models and added languages at every transformer
layer: 9/9 cohorts and 108/108 layer jobs. Every cohort evaluated all 3,000
programs and passed its final validation with no error. Canonical CodeBERT GED
results are under
`analysis_results/attention_final_3000/<language>/similarity_legacy/codebert/`.
GraphCodeBERT and CodeT5 results are under
`analysis_results/final_3000_multimodel/<language>/<model>/attention/similarity_legacy/<model>/`.
Each canonical directory contains all 12 layer JSON files, the merged
`program_metrics_threshold_0.05.npz`, the similarity manifest, and evaluation
protocol. The corresponding `section_3_2_summary*` files include the GED
tables and plots. Completion records are under
`analysis_results/ged_final_3000/validation/`, with the top-level status in
`analysis_results/ged_final_3000/ged_queue_manifest.json`.

The committed GED outputs reproduce the historical first candidate from
NetworkX `optimize_graph_edit_distance` and record the installed NetworkX
version (3.2.1) alongside the paper-declared version (3.0). They should be
described as paper-compatible legacy GED estimates, not globally minimal exact
GED values.

## 2026-08-26 active extension

A resource-bounded background schedule now targets two additions:

1. full 3,000-program Python AST/DFG overlap and representative t-SNE for the
   same six models as the added-language comparison, with GED and probing
   excluded and every overlap curve checked against the stored paper output;
2. DirectProbe for CodeBERT, GraphCodeBERT and CodeT5 in Java, Go and
   JavaScript, using all five tasks at hidden states 5, 9 and 12.

The authoritative live manifests are
[`analysis_results/background_runs/python_analysis_queue.json`](analysis_results/background_runs/python_analysis_queue.json)
and
[`DirectProbe/final_3000/requested_probing_parallel_queue_manifest.json`](DirectProbe/final_3000/requested_probing_parallel_queue_manifest.json).
Do not count a scheduled item as a result until its nested validation/solver
manifest says `complete`.

The five probing tasks now run as concurrent lanes, with four workers per lane
and model/language/layer configurations sequential inside each lane. This caps
the queue at five configurations and 20 DirectProbe workers concurrently. The
current planning horizon is 7--14 days rather than the earlier fully serial
5--10-week estimate.

Adapter dataset preparation keeps the requested balanced label counts fixed.
When a model exposes too few aligned candidates within the paper's initial
program cap, only the search cap is expanded and the actual cap is recorded.
The first validated case was GraphCodeBERT/Java: distance required 240 instead
of 160 programs and identifier-distance required 675 instead of 450; both
still contain exactly 1,300 examples per label.

This document inventories the results currently present in the repository. A
result is called **complete** only when its manifest and every expected output
passed the pipeline's coverage checks. Pilot and interrupted results are kept
separate from the completed 3,000-program analyses.

The paper used as the reference is
[`analysis_results/python/paper.pdf`](analysis_results/python/paper.pdf).

## What is complete

| Scope | Model(s) | Language(s) | Completed analysis | Main result location |
|---|---|---|---|---|
| Python reference sanity check, 100 programs | CodeBERT | Python | AST/DFG attention overlap and legacy GED; comparison with the stored 3,000-program Python curves | [`analysis_results/attention_pilot_100/python`](analysis_results/attention_pilot_100/python) |
| Final cohort, 3,000 programs per language | CodeBERT | Java, Go, JavaScript | AST/DFG attention overlap for every attention layer; representative hidden-state t-SNE analysis | [`analysis_results/attention_final_3000`](analysis_results/attention_final_3000), [`analysis_results/java/tsne/final_3000`](analysis_results/java/tsne/final_3000), [`analysis_results/go/tsne/final_3000`](analysis_results/go/tsne/final_3000), [`analysis_results/javascript/tsne/final_3000`](analysis_results/javascript/tsne/final_3000) |
| Final cohort, 3,000 programs per model/language | GraphCodeBERT, UniXcoder, CodeT5, PLBART, CodeT5+220M | Java, Go, JavaScript | AST/DFG attention overlap for every attention layer; representative hidden-state t-SNE analysis | [`analysis_results/final_3000_multimodel`](analysis_results/final_3000_multimodel) |
| DirectProbe verification | CodeBERT | Python | Five tasks at four selected layers, 20/20 Gurobi runs | [`DirectProbe/pilot_100_gurobi`](DirectProbe/pilot_100_gurobi) |
| DirectProbe adapter verification | GraphCodeBERT, UniXcoder, CodeT5, PLBART, CodeT5+220M | Python | Siblings task at the final model layer, 5/5 Gurobi runs | [`DirectProbe/model_adapter_verification_gurobi`](DirectProbe/model_adapter_verification_gurobi) |

The final non-CodeGen matrix contains 15 successful adapter runs. Its aggregate
manifest reports `status=complete`, `num_complete=15`, no failed run, and
478,288,937,558 bytes of validated raw tensors purged after their permanent
outputs were saved. See
[`analysis_results/final_3000_multimodel/run_manifest.json`](analysis_results/final_3000_multimodel/run_manifest.json).

At the original snapshot, the retained adapter outputs occupied about 205 MB
and the CodeBERT intermediates occupied roughly 100 GB. On 2026-08-26, after
all 15 CodeBERT DirectProbe datasets and permanent attention/t-SNE outputs
were validated, 18,000 manifest-listed graph/embedding tensors totaling
105,839,423,206 bytes were removed. The cleanup record is
[`DirectProbe/final_3000/codebert_source_cleanup_manifest.json`](DirectProbe/final_3000/codebert_source_cleanup_manifest.json).
The generated DirectProbe datasets remain; future CodeBERT GED or new probing
pair selection would require re-extraction.

Together with the three CodeBERT runs, the repository therefore contains 18
complete 3,000-program model/language combinations. Every saved AST layer file
reports 3,000 evaluated programs. Every corresponding DFG layer file reports
3,000 aligned programs and an end-to-end alignment rate of 1.0. This establishes
coverage and alignment; it does not by itself prove that every parser edge is
semantically perfect.

## Main 3,000-program measurements

The table below gives a compact index into the full result files. `AST F` and
`DFG F` are the highest macro-mean F-scores at the paper's primary attention
threshold of 0.05, selected independently across all heads and layers. Layer
and head numbers in this table are one-based, matching the paper's presentation.

The final two columns are additional reproducibility diagnostics produced by
our t-SNE runner at hidden-state index 5: cosine silhouette for token-type
labels and Spearman correlation between AST and hidden-representation distance
matrices. The paper treats t-SNE qualitatively, so these two diagnostics are
not paper-reported scores and should not be interpreted as substitutes for the
saved plots.

| Language | Model | Attention layers | Best AST F (layer/head) | Best DFG F (layer/head) | Token-type silhouette | AST/hidden distance Spearman r |
|---|---|---:|---:|---:|---:|---:|
| Java | CodeBERT | 12 | 0.3684 (6/9) | 0.1570 (2/10) | 0.1253 | 0.1419 |
| Java | GraphCodeBERT | 12 | 0.3601 (4/10) | 0.2145 (11/7) | 0.1100 | 0.1365 |
| Java | UniXcoder | 12 | 0.3553 (7/12) | 0.1361 (6/11) | 0.0881 | 0.0526 |
| Java | CodeT5 | 12 | 0.3783 (10/8) | 0.1356 (3/1) | 0.1839 | -0.0088 |
| Java | PLBART | 6 | 0.3905 (2/4) | 0.1409 (1/6) | 0.2250 | 0.0110 |
| Java | CodeT5+220M | 12 | 0.3607 (1/8) | 0.1611 (5/5) | 0.2803 | -0.0542 |
| Go | CodeBERT | 12 | 0.3655 (8/12) | 0.1583 (2/10) | 0.1368 | 0.1858 |
| Go | GraphCodeBERT | 12 | 0.3568 (6/4) | 0.1881 (11/11) | 0.1132 | 0.2056 |
| Go | UniXcoder | 12 | 0.3423 (4/12) | 0.1596 (6/11) | 0.1038 | 0.2620 |
| Go | CodeT5 | 12 | 0.3572 (4/8) | 0.1416 (3/1) | 0.1843 | 0.1181 |
| Go | PLBART | 6 | 0.3748 (2/4) | 0.1295 (2/1) | 0.2434 | 0.1093 |
| Go | CodeT5+220M | 12 | 0.3724 (8/8) | 0.1863 (6/1) | 0.2812 | 0.0274 |
| JavaScript | CodeBERT | 12 | 0.3442 (6/9) | 0.1482 (6/7) | 0.1423 | 0.2450 |
| JavaScript | GraphCodeBERT | 12 | 0.3378 (6/4) | 0.1803 (11/7) | 0.1155 | 0.2412 |
| JavaScript | UniXcoder | 12 | 0.3471 (1/7) | 0.1434 (4/10) | 0.1091 | 0.2153 |
| JavaScript | CodeT5 | 12 | 0.3522 (4/8) | 0.1726 (3/1) | 0.1706 | 0.1532 |
| JavaScript | PLBART | 6 | 0.4043 (2/4) | 0.1510 (1/6) | 0.2385 | 0.2259 |
| JavaScript | CodeT5+220M | 12 | 0.3520 (2/8) | 0.1997 (5/5) | 0.2730 | 0.1096 |

The detailed per-layer, per-head, per-threshold precision, recall, F-score and
confidence-interval data remain in each model/language directory. For example:

- CodeBERT/Java:
  [`analysis_results/attention_final_3000/java/section_3_2_summary.json`](analysis_results/attention_final_3000/java/section_3_2_summary.json)
- GraphCodeBERT/Java:
  [`analysis_results/final_3000_multimodel/java/graphcodebert/attention/section_3_2_summary.json`](analysis_results/final_3000_multimodel/java/graphcodebert/attention/section_3_2_summary.json)
- CodeT5+220M/JavaScript:
  [`analysis_results/final_3000_multimodel/javascript/codet5p_220/attention/section_3_2_summary.json`](analysis_results/final_3000_multimodel/javascript/codet5p_220/attention/section_3_2_summary.json)

## Python reference check

The Python 100-program CodeBERT rerun was compared descriptively with the
stored 3,000-program Python results used for the paper. At threshold 0.05:

| Curve | Pearson correlation across layers | Mean absolute difference |
|---|---:|---:|
| AST F-score | 0.9980 | 0.00265 |
| DFG F-score | 0.9987 | 0.00559 |

This is strong evidence that the extended overlap implementation reproduces
the shape and magnitude of the established Python pipeline despite the much
smaller pilot sample. It is a sampling sanity check, not an equivalence proof.

The legacy GED pilot was less uniform: DFG GED had correlation 0.9887 and AST
without identifiers 0.9079, while full-AST GED correlation was only 0.2306.
The legacy calculation intentionally takes the first candidate returned by
NetworkX's GED optimizer and is sensitive to sample size and NetworkX behavior.
Consequently, legacy GED is retained for paper comparability but should be
reported as an estimate, not as exact graph edit distance. The complete
comparison is in
[`python_reference_comparison.json`](analysis_results/attention_pilot_100/python/python_reference_comparison.json).

## Partial DirectProbe results

The full DirectProbe matrix was deliberately stopped before completion. Four
individual 3,000-cohort CodeBERT sibling runs did finish with Gurobi:

| Language | Hidden layer | Clusters | Test accuracy | Result directory |
|---|---:|---:|---:|---|
| Java | 5 | 4 | 0.9117 | [`DirectProbe/final_3000/results/java/siblings/codebert/5`](DirectProbe/final_3000/results/java/siblings/codebert/5) |
| Go | 5 | 4 | 0.9067 | [`DirectProbe/final_3000/results/go/siblings/codebert/5`](DirectProbe/final_3000/results/go/siblings/codebert/5) |
| Go | 9 | 4 | 0.9183 | [`DirectProbe/final_3000/results/go/siblings/codebert/9`](DirectProbe/final_3000/results/go/siblings/codebert/9) |
| JavaScript | 5 | 5 | 0.8600 | [`DirectProbe/final_3000/results/javascript/siblings/codebert/5`](DirectProbe/final_3000/results/javascript/siblings/codebert/5) |

These values are usable as individual results, but they do **not** make the
full hidden-representation probing analysis complete. The enclosing manifests
remain `in_progress`. As a paper reference, Table 6 reports four clusters for
Python CodeBERT's layer-5 Keyword-All sibling task and per-label accuracies of
0.87/0.94. That is qualitatively consistent with the Java and Go cluster count,
but is not a direct multilingual baseline.

## Paper mapping

| Repository analysis | Paper method | Paper result reference |
|---|---|---|
| Attention threshold sweep and AST/DFG precision, recall and F-score | Section 3.2.1 and 3.2.2 | Figures 2-4 and Appendix Figures 7-8 |
| Legacy NetworkX GED per node | Section 3.2.2 | Figure 5 and Appendix Figure 9 |
| Token-type and AST-vs-hidden-distance t-SNE | Section 3.3.1 and Appendix G | Section 4.2, Figures 10-11 |
| DirectProbe distance, sibling and DFG tasks | Section 3.3.2 and Appendix H | Tables 1-3 and Appendix Tables 5-10 |
| Dataset preparation: CodeSearchNet test split, comment/docstring removal and fewer than 500 model subtokens | Appendix D | Dataset protocol rather than a result figure |

The multilingual Java/Go/JavaScript results extend the paper's Python analysis;
they are not values reported by the paper itself.

## Not complete yet

- **Full GED:** not run for any final 3,000-program Java, Go or JavaScript
  model/language pair. Every final summary explicitly records GED as skipped.
- **Full DirectProbe:** only the four individual CodeBERT sibling runs listed
  above are complete. Distance, identifier-restricted distance, remaining
  sibling layers, identifier-restricted siblings and DFG probing remain.
- **CodeGen:** parser/tokenizer preflights were completed for 100 programs in
  all four languages, and a one-program Python forward/extraction/AST/DFG test
  completed. No 3,000-program CodeGen analysis has been run.
- **Full Python rerun:** the new verification is a 100-program pilot. The
  existing 3,000-program Python files are the stored paper reference, not a new
  end-to-end rerun under the multilingual pipeline.

For the adapter runs, the large staged `.pkl` graph and hidden-state tensors
were purged only after validation. Their manifests and all permanent overlap
and t-SNE outputs remain. GED or DirectProbe for those models therefore
requires re-extraction. CodeBERT's final raw graph and hidden-state artifacts
are still retained under [`graph_info/final_3000`](graph_info/final_3000) and
[`structural_probe`](structural_probe).
