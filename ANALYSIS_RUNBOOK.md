# Analysis runbook

This is the final reproduction guide for the four-language structural analysis.
Run commands from the repository root. Existing final outputs are complete; use
new staging/output directories for exploratory reruns so validated results are
not overwritten accidentally.

## 1. Scope and authoritative inputs

The retained 3,000-program inputs are:

```text
attention/exp_data/exp_0.jsonl                 Python paper cohort
attention/exp_data/final_3000/java.jsonl       Java extension cohort
attention/exp_data/final_3000/javascript.jsonl JavaScript extension cohort
attention/exp_data/final_3000/go.jsonl         Go extension cohort
```

Each added-language file has a neighboring `_manifest.json` recording its
test-partition filter, seed, sample size, duplicate checks, alignment checks,
and SHA-256. The exact DirectProbe pairs and splits are recorded in each
`DirectProbe/final_3000/data/<language>/<task>/<model>/dataset_manifest.json`;
the corresponding `entities/` and `labels/` files are retained.

The final experiment matrix is:

- attention overlap: six models × four languages;
- GED: CodeBERT, GraphCodeBERT, and CodeT5 × Java, JavaScript, and Go, with
  original Python outputs as the reference;
- t-SNE: CodeBERT × four languages in the main comparison;
- DirectProbe: three models × four languages × five tasks × layers 5, 9, 12,
  using original Python outputs as the reference.

## 2. Environment

The completed project used Python 3.9 and the following interpreter:

```bash
REPO_ROOT=/home/abhinav/sdt_project/dev-sebastian
PYTHON_BIN=/home/abhinav/miniconda3/envs/attention/bin/python
MPLCONFIGDIR=/tmp/sdt-matplotlib
export MPLCONFIGDIR
cd "$REPO_ROOT"
```

To create a fresh environment:

```bash
conda create -n attention python=3.9.16
conda activate attention
pip install -r attention/requirements.txt
pip install -r DirectProbe/requirements.txt
pip install -e DirectProbe
```

Build the pinned ABI-14 Tree-sitter libraries once. The explicit builder avoids
a known setuptools linker problem with Python's C++ scanner:

```bash
"$PYTHON_BIN" attention/build_parsers.py --output-dir build
```

Analysis scripts then load `build/my-languages-<language>.so`. Model extraction
needs the checkpoints listed in
`attention/MODEL_ADAPTERS.md`. Remove `--local_files_only` only when a checkpoint
must be downloaded and network access is available.

For bounded CPU runs, use:

```bash
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
```

Use a CUDA device only after checking its free memory. The completed queues use
disk and memory guards and are resumable from their JSON manifests.

## 3. Validate retained data and results

Check cohort sizes:

```bash
wc -l \
  attention/exp_data/exp_0.jsonl \
  attention/exp_data/final_3000/java.jsonl \
  attention/exp_data/final_3000/javascript.jsonl \
  attention/exp_data/final_3000/go.jsonl
```

Rebuild the result inventory and package checksums. This also verifies that
every indexed primary output and validation path exists:

```bash
"$PYTHON_BIN" results/build_inventory.py
"$PYTHON_BIN" results/validate_archive.py
(cd results && sha256sum -c checksums.sha256)
```

The authoritative completion overview is `ANALYSIS_RESULTS_SUMMARY.md`, and the
file-level index is `results/RESULT_INVENTORY.csv`.

## 4. Attention overlap and t-SNE

### 4.1 CodeBERT for Java, JavaScript, and Go

`run_section_3_2.py` extracts attention/AST artifacts, validates all 3,000
programs, computes AST/DFG precision/recall/F-score for every layer, and writes
the paper-style threshold-0.05 summary. GED is deliberately separate.

```bash
"$PYTHON_BIN" attention/run_section_3_2.py \
  --languages java javascript go \
  --dataset_dir attention/exp_data/final_3000 \
  --graph_root graph_info/reproduction_codebert \
  --results_root analysis_results/reproduction_codebert \
  --expected_count 3000 \
  --device cpu \
  --bootstrap_samples 1000 \
  --skip_ged \
  --skip_python_reference_comparison
```

To regenerate CodeBERT hidden states and t-SNE for one language:

```bash
LANG=java
GRAPH_DIR="graph_info/reproduction_codebert/${LANG}/codebert"
EMBED_DIR="structural_probe/reproduction/${LANG}/final_3000/codebert"

"$PYTHON_BIN" attention/save_word_embedding.py \
  --model codebert \
  --code_file "attention/exp_data/final_3000/${LANG}.jsonl" \
  --graph_loc "$GRAPH_DIR" \
  --save_dir structural_probe/reproduction \
  --exp_name final_3000 \
  --lang "$LANG" \
  --device cpu \
  --seed 0

"$PYTHON_BIN" attention/hidden_tsne.py \
  --embedding_dir "$EMBED_DIR" \
  --save_dir "analysis_results/reproduction_tsne/${LANG}" \
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

Repeat with `LANG=javascript` and `LANG=go`.

### 4.2 GraphCodeBERT, UniXcoder, PLBART, CodeT5, and CodeT5+220M

The storage-bounded runner performs paired attention/hidden-state extraction,
validation, AST/DFG overlap, summary generation, layer-5 t-SNE, final
validation, and manifest-scoped deletion of raw tensors:

```bash
"$PYTHON_BIN" attention/run_staged_model_analysis.py \
  --models graphcodebert unixcoder plbart codet5 codet5p_220 \
  --languages java javascript go \
  --dataset_root attention/exp_data/final_3000 \
  --graph_stage_root graph_info/reproduction_adapters \
  --embedding_stage_root structural_probe/reproduction_adapters \
  --results_root analysis_results/reproduction_multimodel \
  --run_manifest analysis_results/reproduction_multimodel/run_manifest.json \
  --expected_programs 3000 \
  --tsne_layer 5 \
  --device cpu \
  --minimum_free_gb 35 \
  --omp_threads 4
```

For Python, use the same runner with the paper cohort and reference comparison:

```bash
"$PYTHON_BIN" attention/run_staged_model_analysis.py \
  --models graphcodebert unixcoder plbart codet5 codet5p_220 \
  --languages python \
  --dataset_root attention/exp_data \
  --dataset_pattern exp_0.jsonl \
  --graph_stage_root graph_info/reproduction_python_adapters \
  --embedding_stage_root structural_probe/reproduction_python_adapters \
  --results_root analysis_results/reproduction_python_multimodel \
  --run_manifest analysis_results/reproduction_python_multimodel/run_manifest.json \
  --expected_programs 3000 \
  --compare_python_reference \
  --device cpu \
  --minimum_free_gb 35 \
  --omp_threads 4
```

Python CodeBERT has its dedicated, paper-reference-aware runner:

```bash
"$PYTHON_BIN" attention/run_staged_codebert_analysis.py \
  --dataset_root attention/exp_data \
  --dataset_pattern exp_0.jsonl \
  --graph_stage_root graph_info/reproduction_python_codebert \
  --embedding_stage_root structural_probe/reproduction_python_codebert \
  --attention_results_root analysis_results/reproduction_python_codebert \
  --hidden_results_root analysis_results/reproduction_python_tsne \
  --run_manifest analysis_results/reproduction_python_codebert/run_manifest.json \
  --python_reference_root attention/graph_comparision \
  --expected_programs 3000 \
  --device cpu
```

## 5. Graph edit distance

The paper-compatible analysis uses `distance_mode=legacy`, which reproduces the
first candidate yielded by NetworkX `optimize_graph_edit_distance`; it is not a
guaranteed global minimum. The final queue covers 3 models × 3 added languages
× 12 layers. For each model/language cohort it extracts representations, runs
up to 12 isolated layer jobs, merges and validates them, and then purges only
manifest-listed temporary tensors.

Dry-run the schedule first:

```bash
"$PYTHON_BIN" attention/run_requested_ged_queue.py \
  --models codebert graphcodebert codet5 \
  --languages java javascript go \
  --workers 12 \
  --device cpu \
  --dry_run
```

Start a durable background run:

```bash
tmux new-session -d -s sdt-ged \
  "cd $REPO_ROOT && exec $PYTHON_BIN attention/run_requested_ged_watchdog.py \
   --python $PYTHON_BIN --workers 12"
```

Monitor with:

```bash
tmux capture-pane -pt sdt-ged
sed -n '1,240p' analysis_results/ged_final_3000/ged_queue_manifest.json
```

Final values are merged into the associated attention result directory under
`similarity_legacy/`; `analysis_results/ged_final_3000/validation/` records the
coverage check. GED does not use Gurobi.

## 6. DirectProbe

The five tasks are `distance`, `distance_id`, `siblings`, `siblings_id`, and
`dfg`. The selected models are CodeBERT, GraphCodeBERT, and CodeT5, and the
hidden states are 5, 9, and 12.

### 6.1 Recreate adapter probe matrices

This staged command recreates GraphCodeBERT and CodeT5 representations, builds
all balanced task datasets, verifies the generated configs/matrices, and
deletes the large source tensors after the probe matrices are safe:

```bash
"$PYTHON_BIN" attention/run_staged_directprobe_preparation.py \
  --models graphcodebert codet5 \
  --languages java javascript go \
  --dataset_root attention/exp_data/final_3000 \
  --graph_stage_root graph_info/reproduction_directprobe \
  --embedding_stage_root structural_probe/reproduction_directprobe \
  --output_root DirectProbe/reproduction_final_3000 \
  --run_manifest DirectProbe/reproduction_final_3000/preparation_manifest.json \
  --expected_programs 3000 \
  --device cpu \
  --minimum_free_gb 55 \
  --omp_threads 4
```

For CodeBERT, retain a paired graph/embedding cohort using
`save_graph_info.py` and `save_word_embedding.py`, then run
`attention/create_multilingual_dp.py` once per task. Use the retained final
dataset manifest's `program_limit_used`, `selected_per_label`, layers, and seed
for an exact reconstruction. Example:

```bash
LANG=java
GRAPH_BASE="graph_info/reproduction_dp_codebert/${LANG}"
GRAPH_DIR="${GRAPH_BASE}/final_3000/codebert"
EMBED_DIR="structural_probe/reproduction_dp_codebert/${LANG}/final_3000/codebert"

"$PYTHON_BIN" attention/save_graph_info.py \
  --model codebert \
  --code_file "attention/exp_data/final_3000/${LANG}.jsonl" \
  --save_dir "$GRAPH_BASE" \
  --exp_name final_3000 \
  --lang "$LANG" \
  --device cpu \
  --seed 0

"$PYTHON_BIN" attention/save_word_embedding.py \
  --model codebert \
  --code_file "attention/exp_data/final_3000/${LANG}.jsonl" \
  --graph_loc "$GRAPH_DIR" \
  --save_dir structural_probe/reproduction_dp_codebert \
  --exp_name final_3000 \
  --lang "$LANG" \
  --device cpu \
  --seed 0

"$PYTHON_BIN" attention/create_multilingual_dp.py \
  --task distance \
  --lang "$LANG" \
  --model codebert \
  --embedding_dir "$EMBED_DIR" \
  --graph_dir "$GRAPH_DIR" \
  --output_root DirectProbe/reproduction_final_3000 \
  --layers 5 9 12 \
  --target_per_label 1300 \
  --max_programs 160 \
  --require_target \
  --seed 0
```

The paper-style starting caps are 160 (`distance`), 450 (`distance_id`), 100
(`siblings`), 300 (`siblings_id`), and 130 (`dfg`). Some final multilingual
datasets expanded the cap to reach the same balanced per-label target; the
retained manifest is authoritative.

### 6.2 Run the solver

Never change the server's global Gurobi configuration. Scope the dedicated
licence to each invocation:

```bash
GRB_LICENSE_FILE=/home/abhinav/sdt_project/.config/gurobi/dev-sebastian/gurobi.lic \
  "$PYTHON_BIN" DirectProbe/run_multilingual_pilot.py \
  --dp_dir DirectProbe \
  --pilot_root DirectProbe/reproduction_final_3000 \
  --languages java javascript go \
  --models codebert graphcodebert codet5 \
  --tasks distance distance_id siblings siblings_id dfg \
  --layers 5 9 12 \
  --workers 4 \
  --timeout 172800 \
  --required_solver gurobi \
  --quarantine_incomplete \
  --manifest_name directprobe_reproduction_manifest.json
```

The historical production scheduler is
`DirectProbe/run_requested_probe_queue.py`; it runs the five tasks as separate
lanes and injects the same local licence into every child process. Use the
generic command above for a clean output root.

## 7. Rebuild the final comparison report

This step reads stored results only; it does not perform model inference, GED,
or probing:

```bash
MPLCONFIGDIR=/tmp/sdt-final-report \
  "$PYTHON_BIN" results/build_results.py

cd results/report
latexmk -pdf -interaction=nonstopmode four_language_results.tex
cd "$REPO_ROOT"
"$PYTHON_BIN" results/build_inventory.py
(cd results && sha256sum -c checksums.sha256)
```

If `latexmk` is unavailable, upload the complete `results/report/` directory to
Overleaf or use another LaTeX engine that provides `graphicx`, `booktabs`, and
`geometry`.

## 8. Re-run validation audits

The DFG and PLBART audits run directly from retained artifacts:

```bash
PYTHONPATH=attention MPLCONFIGDIR=/tmp \
  "$PYTHON_BIN" results/validation/dfg_manual_audit.py

"$PYTHON_BIN" results/validation/plbart_reference_audit.py
```

The stored program-disjoint audit output is final and remains in
`results/validation/probe_split_audit.{json,csv}`. Re-running its source needs
the layer-12 probe embedding matrices. Recreate them with Section 6.1, place
them under the data root selected by the script, and then run:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  "$PYTHON_BIN" results/validation/probe_split_audit.py
```

Read `results/validation/README.md` before interpreting the diagnostics. In
particular, the program-disjoint probe audit is a limitation analysis; the main
paper-compatible probe results retain the original pair-level split protocol.

## 9. Acceptance criteria

Do not use a run in the final comparison unless all applicable checks pass:

- cohort manifest: complete, 3,000 test records, no duplicate cleaned code;
- graph/embedding manifests: correct model, language, dataset hash, and zero
  extraction failures;
- every AST layer: `num_evaluated=3000`;
- every DFG layer: `num_aligned=3000` and alignment rate 1.0;
- attention summary: complete with threshold 0.05;
- GED: explicitly `legacy` for paper comparison and all 12 layers present;
- t-SNE manifest: layer 5, recorded perplexity, 50,000 iterations, seed 0;
- DirectProbe manifest: all four output files present and `solver=gurobi`;
- final inventory rebuild succeeds and all package checksums verify.

Raw `.pkl` tensors, queue logs, GED shards, and probe embedding text matrices
are staging artifacts. Delete them only after these validations pass and only
within the manifest-recorded scope.
