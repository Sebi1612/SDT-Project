# Analysis runbook

This runbook describes how to reproduce the paper-compatible AST/DFG attention
analysis (Section 3.2), qualitative hidden-state analysis (Section 3.3.1), and
DirectProbe analysis (Section 3.3.2) for Python, Java, Go and JavaScript.

Read [`ANALYSIS_RESULTS_SUMMARY.md`](ANALYSIS_RESULTS_SUMMARY.md) first to see
which runs already exist. The commands below do not need to be rerun for a
combination whose validation manifest is already complete.

## 1. Environment and conventions

Run commands from the repository root and use the `attention` environment:

```bash
REPO_ROOT=/home/abhinav/sdt_project/dev-sebastian
PYTHON_BIN=/home/abhinav/miniconda3/envs/attention/bin/python
MPLCONFIGDIR=/tmp/sdt-matplotlib
export MPLCONFIGDIR
cd "$REPO_ROOT"
```

Use `--device cuda:0` only when `torch.cuda.is_available()` is true and the GPU
has enough memory. Otherwise use `--device cpu`. The completed adapter matrix
was run on CPU with four OpenMP/MKL threads.

The adapter runner uses locally cached model files and sets Hugging Face and
Transformers to offline mode. The required checkpoints are listed in
[`attention/MODEL_ADAPTERS.md`](attention/MODEL_ADAPTERS.md).

DirectProbe must receive the project-local Gurobi licence on every solver
invocation; do not change a global Gurobi setting:

```bash
GRB_LICENSE_FILE=/home/abhinav/sdt_project/.config/gurobi/dev-sebastian/gurobi.lic \
  "$PYTHON_BIN" DirectProbe/run_multilingual_pilot.py --help
```

The model names accepted by the new adapter path are `graphcodebert`,
`unixcoder`, `codet5`, `plbart`, `codet5p_220`, and `codegen`. CodeBERT uses its
established extraction path and is intentionally not routed through the
adapter.

## 2. Prepare the paper-compatible 3,000-program cohorts

For Java, Go and JavaScript, the cohort builder performs all required steps in
one pass: select only CodeSearchNet's test partition, remove comments, preserve
and validate AST token alignment, require DFG alignment, require exact model
subtoken merging, enforce fewer than 500 subtokens for the relevant models,
and sample deterministically with seed 0.

Run once for each language, replacing `LANG`:

```bash
LANG=java
"$PYTHON_BIN" attention/prepare_attention_cohort.py \
  --input "attention/exp_data/${LANG}_full.jsonl" \
  --output "attention/exp_data/final_3000/${LANG}.jsonl" \
  --manifest "attention/exp_data/final_3000/${LANG}_manifest.json" \
  --lang "$LANG" \
  --sample_size 3000 \
  --seed 0
```

Repeat with `LANG=go` and `LANG=javascript`. Do not overwrite the full input
file. Accept the cohort only when the manifest says `status=complete`,
`sample_size=3000`, `output_partitions={"test": 3000}`, and
`duplicate_cleaned_code_count=0`.

The existing validated files are:

- [`attention/exp_data/final_3000/java.jsonl`](attention/exp_data/final_3000/java.jsonl)
- [`attention/exp_data/final_3000/go.jsonl`](attention/exp_data/final_3000/go.jsonl)
- [`attention/exp_data/final_3000/javascript.jsonl`](attention/exp_data/final_3000/javascript.jsonl)

Python uses the established comment/docstring-free test-split data from the
paper. The multilingual pilot inputs live under
[`attention/exp_data/pilot_100`](attention/exp_data/pilot_100).

## 3. Run the verification ladder before a new full matrix

### 3.1 CodeBERT Python reference gate

This reruns the 100-program Section 3.2 pilot and compares Python with the
stored paper curves. The command intentionally includes the slow,
paper-compatible legacy GED check. For an overlap-only run, add both
`--skip_ged` and `--skip_python_reference_comparison`, because the combined
Python reference report requires GED.

```bash
"$PYTHON_BIN" attention/run_section_3_2.py \
  --languages python \
  --dataset_dir attention/exp_data/pilot_100 \
  --graph_root graph_info/pilot_100 \
  --results_root analysis_results/attention_pilot_100 \
  --expected_count 100 \
  --device cpu \
  --ged_mode legacy
```

The important reference report is
[`analysis_results/attention_pilot_100/python/python_reference_comparison.json`](analysis_results/attention_pilot_100/python/python_reference_comparison.json).

### 3.2 Non-CodeBERT adapter preflight

Run this for every intended model/language before committing storage to a full
extraction. This validates the selected cohort, parser alignment, tokenizer
protocol and model limits without retaining large tensors:

```bash
MODEL=graphcodebert
LANG=java
"$PYTHON_BIN" attention/verify_model_adapter.py \
  --model "$MODEL" \
  --code_file "attention/exp_data/pilot_100/${LANG}.jsonl" \
  --lang "$LANG" \
  --num_codes 100 \
  --local_files_only \
  --report "analysis_results/model_adapter_verification/preflight_100/${LANG}/${MODEL}.json"
```

Add `--forward --device cpu` (or a valid CUDA device) for a real checkpoint
forward pass. For CodeGen, use `--num_codes 1` for the initial forward gate
because the 3.7B checkpoint is much larger.

## 4. Recommended storage-bounded non-GED/non-DirectProbe matrix

The following command performs the already-tested sequence for five models and
three languages:

1. paired attention/hidden-state extraction;
2. strict 3,000-artifact validation;
3. AST overlap on every attention layer;
4. DFG overlap and strict end-to-end alignment on every attention layer;
5. paper-style Section 3.2 summary at threshold 0.05, with GED marked skipped;
6. representative layer-5 t-SNE (100 shortest programs with at least 100 code
   tokens, token perplexity 50, distance perplexities 5/10, 50,000 iterations);
7. permanent-output validation; and
8. deletion of only the manifest-listed raw `.pkl` tensors after validation.

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  "$PYTHON_BIN" attention/run_staged_model_analysis.py \
  --models graphcodebert unixcoder codet5 plbart codet5p_220 \
  --languages java go javascript \
  --dataset_root attention/exp_data/final_3000 \
  --graph_stage_root graph_info/staged_final_3000 \
  --embedding_stage_root structural_probe/staged_final_3000 \
  --results_root analysis_results/final_3000_multimodel \
  --run_manifest analysis_results/final_3000_multimodel/run_manifest.json \
  --expected_programs 3000 \
  --tsne_layer 5 \
  --device cpu \
  --minimum_free_gb 35 \
  --omp_threads 4
```

The runner is resumable: combinations already marked complete in the aggregate
manifest are skipped. For a genuinely fresh rerun, use new graph, embedding,
results and manifest roots instead of overwriting the validated run.

This staged command is appropriate only when GED and DirectProbe are deferred.
Those analyses need the raw tensors, which the command purges after validation.

## 5. CodeBERT attention and t-SNE analysis

### 5.1 Attention overlap without GED

The Section 3.2 runner creates CodeBERT attention/AST artifacts, validates the
manifest, computes all AST/DFG overlap layers and writes summaries:

```bash
"$PYTHON_BIN" attention/run_section_3_2.py \
  --languages java go javascript \
  --dataset_dir attention/exp_data/final_3000 \
  --graph_root graph_info/final_3000 \
  --results_root analysis_results/attention_final_3000 \
  --expected_count 3000 \
  --device cpu \
  --bootstrap_samples 1000 \
  --skip_ged \
  --skip_python_reference_comparison
```

If the validated graph artifacts already exist and only the downstream metrics
must be regenerated, add `--skip_graph_generation`.

### 5.2 Hidden states and representative t-SNE

Run once for each language. `save_word_embedding.py` uses the graph manifest to
guarantee that hidden states and attention results refer to the same programs.

```bash
LANG=java
"$PYTHON_BIN" attention/save_word_embedding.py \
  --model codebert \
  --code_file "attention/exp_data/final_3000/${LANG}.jsonl" \
  --graph_loc "graph_info/final_3000/${LANG}/codebert" \
  --save_dir structural_probe \
  --exp_name final_3000 \
  --lang "$LANG" \
  --device cpu \
  --seed 0

"$PYTHON_BIN" attention/validate_representation_run.py \
  --graph_dir "graph_info/final_3000/${LANG}/codebert" \
  --embedding_dir "structural_probe/${LANG}/final_3000/codebert" \
  --expected_model codebert \
  --expected_language "$LANG" \
  --expected_count 3000

"$PYTHON_BIN" attention/hidden_tsne.py \
  --embedding_dir "structural_probe/${LANG}/final_3000/codebert" \
  --save_dir "analysis_results/${LANG}/tsne/final_3000" \
  --lang "$LANG" \
  --layers 5 \
  --token_perplexities 50 \
  --distance_perplexities 5 10 \
  --min_tokens 100 \
  --max_programs 100 \
  --program_selection shortest \
  --iterations 50000 \
  --seed 0
```

Repeat with `LANG=go` and `LANG=javascript`.

## 6. Manual retained workflow for a complete model/language analysis

Use this workflow when GED and/or DirectProbe will follow. It keeps the raw
graph and hidden-state tensors. The example uses GraphCodeBERT/Java; change the
two variables for another adapter model or language.

```bash
MODEL=graphcodebert
LANG=java
GRAPH_DIR="graph_info/retained_final_3000/${LANG}/${MODEL}"
EMBED_DIR="structural_probe/retained_final_3000/${LANG}/${MODEL}"
RESULT_DIR="analysis_results/retained_final_3000/${LANG}/${MODEL}"

"$PYTHON_BIN" attention/extract_model_representations.py \
  --model "$MODEL" \
  --code_file "attention/exp_data/final_3000/${LANG}.jsonl" \
  --graph_output_dir "$GRAPH_DIR" \
  --embedding_output_dir "$EMBED_DIR" \
  --lang "$LANG" \
  --device cpu \
  --local_files_only

"$PYTHON_BIN" attention/validate_representation_run.py \
  --graph_dir "$GRAPH_DIR" \
  --embedding_dir "$EMBED_DIR" \
  --expected_model "$MODEL" \
  --expected_language "$LANG" \
  --expected_count 3000

"$PYTHON_BIN" attention/graph_comp.py \
  --graph_loc "$GRAPH_DIR" \
  --save_dir "$RESULT_DIR/attention" \
  --all_layers \
  --bootstrap_samples 1000

"$PYTHON_BIN" attention/dfg_comp.py \
  --graph_loc "$GRAPH_DIR" \
  --code_file "attention/exp_data/final_3000/${LANG}.jsonl" \
  --save_dir "$RESULT_DIR/attention" \
  --lang "$LANG" \
  --all_layers \
  --bootstrap_samples 1000

"$PYTHON_BIN" attention/hidden_tsne.py \
  --embedding_dir "$EMBED_DIR" \
  --save_dir "$RESULT_DIR/hidden_tsne" \
  --lang "$LANG" \
  --layers 5 \
  --token_perplexities 50 \
  --distance_perplexities 5 10 \
  --min_tokens 100 \
  --max_programs 100 \
  --program_selection shortest \
  --iterations 50000 \
  --seed 0
```

Do not use `extract_model_representations.py` for CodeBERT; use Section 5.

## 7. GED attention analysis

GED is still part of Section 3.2. For comparison with the paper, use
`--distance_mode legacy`. It reproduces the original first-candidate NetworkX
estimate and can take roughly ten hours per model layer according to the
project README. Run it only while the corresponding graph artifacts are still
present.

For the retained adapter example from Section 6:

```bash
"$PYTHON_BIN" attention/similarity.py \
  --graphs_dir "$GRAPH_DIR" \
  --code_file "attention/exp_data/final_3000/${LANG}.jsonl" \
  --save_dir "$RESULT_DIR/attention" \
  --lang "$LANG" \
  --all_layers \
  --threshold 0.05 \
  --bootstrap_samples 1000 \
  --distance_mode legacy

"$PYTHON_BIN" attention/summarize_attention_analysis.py \
  --results_dir "$RESULT_DIR/attention" \
  --output "$RESULT_DIR/attention/section_3_2_summary.json" \
  --model "$MODEL" \
  --lang "$LANG" \
  --threshold 0.05 \
  --expected_programs 3000 \
  --ged_mode legacy
```

For CodeBERT, either run the same two commands against
`graph_info/final_3000/$LANG/codebert`, or invoke `run_section_3_2.py` without
`--skip_ged` and with `--skip_graph_generation`.

The exact fixed-node edge distance is also available with
`--distance_mode fixed`, but save it separately and do not compare it directly
with the paper's legacy GED numbers.

## 8. DirectProbe hidden-representation analysis

DirectProbe needs both retained graph and embedding `.pkl` files. Build one
task dataset at a time with `create_multilingual_dp.py`. To follow Appendix H,
use these sampling settings:

| Task | `--target_per_label` | `--max_programs` |
|---|---:|---:|
| `distance` | 1300 | 160 |
| `distance_id` | 1300 | 450 |
| `siblings` | 1500 | 100 |
| `siblings_id` | 1500 | 300 |
| `dfg` | 1500 | 130 |

The paper-compatible hidden layers are model-specific:

| Model | Layers used for the paper-style middle/deep comparison |
|---|---|
| CodeBERT, GraphCodeBERT, UniXcoder, CodeT5 | `5 9 12` |
| PLBART | `3 6` |
| CodeT5+220M | `5 12` |
| CodeGen | `8 16` |

Example: create GraphCodeBERT/Java sibling data at layers 5, 9 and 12:

```bash
DP_ROOT=DirectProbe/retained_final_3000
"$PYTHON_BIN" attention/create_multilingual_dp.py \
  --task siblings \
  --lang "$LANG" \
  --model "$MODEL" \
  --embedding_dir "$EMBED_DIR" \
  --graph_dir "$GRAPH_DIR" \
  --output_root "$DP_ROOT" \
  --layers 5 9 12 \
  --target_per_label 1500 \
  --max_programs 100 \
  --require_target \
  --seed 0
```

Repeat with the appropriate values from the two tables for all five tasks. The
same seed, cohort and sampling parameters must be used for every model so the
selected token pairs stay comparable.

Then run the generated configurations with Gurobi. A long timeout is necessary;
start conservatively with one worker so the server is not monopolized:

```bash
GRB_LICENSE_FILE=/home/abhinav/sdt_project/.config/gurobi/dev-sebastian/gurobi.lic \
  "$PYTHON_BIN" DirectProbe/run_multilingual_pilot.py \
  --dp_dir DirectProbe \
  --pilot_root "$DP_ROOT" \
  --languages "$LANG" \
  --models "$MODEL" \
  --tasks distance distance_id siblings siblings_id dfg \
  --layers 5 9 12 \
  --workers 1 \
  --timeout 172800 \
  --required_solver gurobi \
  --manifest_name "directprobe_run_manifest_${LANG}_${MODEL}.json"
```

Adjust `--layers` per model. A run is accepted only if the manifest marks it
complete, all four DirectProbe result files exist, and `solver` is `gurobi`.

## 9. CodeGen

CodeGen is intentionally excluded from the completed full matrix. Before a
3,000-program run, retain the one-program forward gate:

```bash
"$PYTHON_BIN" attention/verify_model_adapter.py \
  --model codegen \
  --code_file attention/exp_data/pilot_100/python.jsonl \
  --lang python \
  --num_codes 1 \
  --forward \
  --device cpu \
  --local_files_only \
  --report analysis_results/model_adapter_verification/forward_1/python/codegen.json
```

After checking free disk and memory, use the retained workflow from Section 6
with `MODEL=codegen`. CodeGen has 16 attention layers and 17 hidden-state
outputs including the embedding state. Its full extraction should be scheduled
separately from the five smaller models.

## 10. Acceptance checklist

Do not use a result in aggregate analysis until all applicable checks pass:

- Cohort manifest is complete, contains exactly 3,000 test records, and records
  comment removal plus AST, DFG and model-token alignment validation.
- Graph and embedding manifests identify the expected model, language, dataset
  hash and artifact cohort, with zero extraction failures.
- `validate_representation_run.py` succeeds for retained paired tensors.
- Every AST layer JSON has `num_evaluated=3000`.
- Every DFG layer JSON has `num_aligned=3000` and
  `end_to_end_alignment_rate=1.0`.
- Section 3.2 summary has `status=complete`, the correct model/language and
  `expected_programs=3000`.
- GED mode is explicitly recorded as `legacy` for paper comparison or `fixed`
  for the newer metric; never combine the two in one curve.
- t-SNE manifests record the intended layer, perplexities, 50,000 iterations,
  seed and selected program rule.
- DirectProbe manifests record `solver=gurobi`; incomplete result directories
  are not counted.
- Keep pilot, verification, final, legacy-GED and fixed-GED outputs in distinct
  directories.

The JSON manifests are the authoritative provenance record. PNG files alone
are not sufficient evidence that a run completed correctly.
