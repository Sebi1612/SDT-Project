# Result-package reproduction

Run all commands from the repository root with the established Python 3.9
environment.

## Regenerate figures, tables, and derived values

```bash
PYTHON_BIN=/home/abhinav/miniconda3/envs/attention/bin/python
MPLCONFIGDIR=/tmp/sdt-final-report "$PYTHON_BIN" results/build_results.py
```

This reads the stored result files and creates six comparative figures, three
DirectProbe tables, and `results/report/derived_summary.json`. It performs no
model inference and no solver work.

## Compile the PDF

```bash
cd results/report
latexmk -pdf -interaction=nonstopmode four_language_results.tex
```

The entire `report/` directory must be present because the TeX file uses
relative figure and table paths.

## Rebuild the inventory and checksums

```bash
/home/abhinav/miniconda3/envs/attention/bin/python results/build_inventory.py
/home/abhinav/miniconda3/envs/attention/bin/python results/validate_archive.py
(cd results && sha256sum -c checksums.sha256)
```

The inventory builder fails if any indexed primary output or validation record
is missing.

## Re-run validation controls

The DFG and PLBART controls run from retained artifacts:

```bash
PYTHONPATH=attention MPLCONFIGDIR=/tmp \
  /home/abhinav/miniconda3/envs/attention/bin/python \
  results/validation/dfg_manual_audit.py

/home/abhinav/miniconda3/envs/attention/bin/python \
  results/validation/plbart_reference_audit.py
```

Re-running `probe_split_audit.py` additionally requires the reproducible
layer-12 probe embedding matrices. Recreate those first with Section 6.1 of
the root runbook. The published JSON/CSV audit outputs are retained.

Full model, GED, and DirectProbe reruns are documented in the root
`ANALYSIS_RUNBOOK.md`. Every Gurobi process must receive the project-local
licence via `GRB_LICENSE_FILE`; do not alter the global configuration.
