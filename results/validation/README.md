# Validation audit of the four-language results

Completed 2026-09-15. This audit tests three comparability risks without rerunning
the expensive DirectProbe/Gurobi or GED matrices. The scripts and machine-readable
results in this folder are intended to make each conclusion reproducible.

## 1. Program-disjoint probing control

`probe_split_audit.py` reassembles the retained final-3000 layer-12 feature
matrices using each `dataset_manifest.json` pair's artifact and split. It fits
the **same** standardized Ridge linear classifier on (a) the project's original
pair-stratified train/test split and (b) each of five stratified, **program-disjoint**
folds. All three new languages, three probed models, and five tasks are covered:
45 configurations, each with five valid folds. The program group is the source
artifact, and assertions verify zero shared artifacts in each disjoint fold.

- The median fraction of original test programs also present in training is
  **98.96%** across the 45 configurations.
- Five-fold disjoint accuracy is below the original pair-split linear-probe
  accuracy in **35/45** configurations, with a median difference of **−1.76
  percentage points**. Balanced accuracy falls by a similar median **−1.75
  points**.
- Every DFG configuration falls; its median drop is **−4.37 points** (range
  −1.97 to −10.91 points). Keyword–identifier distance also falls in all nine
  configurations (median **−2.63 points**). Sibling tasks are mixed.

This is evidence of **split sensitivity**, not a corrected estimate of the
paper's DirectProbe/Gurobi scores: Ridge and DirectProbe are different probes,
and the two splitting protocols test different pair sets. Do not replace the
manuscript's tables with these linear-control numbers. For publication-grade
DirectProbe comparison, regenerate each task's pairs with source-program-grouped
train/test partitioning before balancing, then run the *same DirectProbe solver*
on those pairs. Do not place pairs from one program in both partitions.

Raw per-configuration outputs: `probe_split_audit.json` and
`probe_split_audit.csv`.

## 2. PLBART Python-reference reproduction

The existing Python reference comparison reported a PLBART AST-recall curve
mean absolute difference of **0.03755**, larger than the other five models'
near-matches (maximum curve MAD at most 0.000415). The retained PLBART program
metrics verify **3000/3000** evaluated Python programs and `source_index`
0–2999 with `file_name` matching the original `exp_0.jsonl` rows. Thus a gross
candidate-cohort mismatch or failed extraction is not the explanation.

`plbart_reference_audit.py` compared all 12 heads' F-score, recall, and precision
curves over all nine thresholds, not just the selected best-head curve. It found:

- AST direct-head mean absolute difference grows from **0.0021** at layer 1 to
  **0.0258** at layer 6; DFG head-curve differences are smaller (0.0011–0.0072).
- Candidate and reference choose the same best head in **11/12** layer×graph
  cases. Minimum-cost head remapping leaves the AST differences unchanged, so
  simple head reordering does not explain the mismatch.
- The original PLBART attention extraction and token-merge functions are
  unchanged from the repository's original commit. The candidate evaluator
  records complete coverage, but the legacy reference JSONs do **not** record
  evaluated program counts, checkpoint revision, tokenizer revision, or
  environment details.

An empirical dropout control (`plbart_dropout_check.py`) rules out the tempting
but incorrect explanation that an omitted explicit `model.eval()` caused this
recomputation to use dropout. In the installed Transformers 4.57.6,
`from_pretrained` already returns an evaluation-mode model; repeated default
and evaluation-mode attention is bitwise stable. Forced training mode is
stochastic, but it is not the observed extraction mode. A fixed-input,
fixed-weight control run under Transformers **4.29.2 and 4.57.6** differs in
encoder attention by only about **3.5×10⁻⁸** mean absolute value, so version
drift between these two releases is also not a demonstrated cause. The original
requirements specify Transformers 4.25.1, which was not available for an exact
test. The cause therefore remains **unresolved**; checkpoint/tokenizer history
or unavailable legacy per-program coverage may matter. Keep PLBART's
cross-language curves exploratory rather than treating their absolute level as
paper-validated.

Raw outputs: `plbart_reference_audit.json`, `plbart_dropout_check.json`, and
the two fixed-input `plbart_attention_transformers_*.npz` arrays.

## 3. DFG semantics and graph density

The repository's DFG semantic test suite passes **13 tests with one expected
failure**: Java block-shadowing can lose the outer definition because the
GraphCodeBERT-style extractor tracks names rather than full lexical scope.

`dfg_manual_audit.py` checked equivalent `y=x; z=y+x; return z` functions in
each language. All four produce the same **seven binary dependency edges**, and
the three expected assignment/computation edges are present. Their **native
signed labels differ**, however: Python/Go label those edges −1 while
Java/JavaScript label them +1. This confirms that the project's binary DFG
overlap can be compared cautiously, but signed DFG labels are **not a shared
cross-language semantic ontology**; per-label DirectProbe results should not
be interpreted as if `Comes from` and `Computed from` mean identical source
constructs in every language.

The same script inspected 24 fixed-seed real programs per language using exact
dataset-token alignment; **96/96** extracted and aligned. Median edges per
token are Python **0.280**, Java **0.151**, Go **0.183**, JavaScript **0.251**.
This small sample agrees in broad direction with the existing complete
GraphCodeBERT 3000-program DFG evaluation:

| Language | Mean DFG edges | Mean directed-edge density | Empty DFGs / 3000 |
| --- | ---: | ---: | ---: |
| Python | 31.90 | 0.005254 | 0 |
| Java | 18.76 | 0.002501 | 21 |
| Go | 24.35 | 0.003585 | 5 |
| JavaScript | 29.22 | 0.002896 | 20 |

The full evaluation's density is the mean of per-program directed edge
densities, not the density of a pooled graph. Java's sparser ground-truth DFG
is a plausible contributor to high attention recall, but graph density and
precision–recall trade-offs do not establish causality or extractor accuracy.
The 24-program manual sample is deliberately too small to certify arbitrary
language constructs. Scope/shadowing remains a known open semantic issue.

Raw output: `dfg_manual_audit.json`.

## Reproduction

From the repository root, use the existing attention environment and a single
CPU thread for the probe audit:

```bash
PYTHONPATH=attention MPLCONFIGDIR=/tmp /home/abhinav/miniconda3/envs/attention/bin/python results/validation/dfg_manual_audit.py
/home/abhinav/miniconda3/envs/attention/bin/python results/validation/plbart_reference_audit.py
```

`probe_split_audit.py` requires the layer-12 probe embedding matrices. Those
large reproducible matrices are not archived in the final repository; recreate
them with `ANALYSIS_RUNBOOK.md` Section 6.1 before rerunning this diagnostic.
The final JSON and CSV outputs are retained here.

No Gurobi license or server-wide configuration is needed for these controls.
