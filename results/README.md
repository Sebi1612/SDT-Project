# Final results package

This directory is the review entry point for the multilingual extension of
Anand et al., *A Critical Study of What Code-LLMs (Do Not) Learn*.

## Contents

- [`report/four_language_results.pdf`](report/four_language_results.pdf):
  compiled four-language results section.
- `report/four_language_results.tex`: editable LaTeX source.
- `report/figures/` and `report/tables/`: all report inputs.
- `report/derived_summary.json`: machine-readable values derived for the report.
- [`RESULT_INVENTORY.csv`](RESULT_INVENTORY.csv): 153 indexed result groups with
  primary path, provenance, validation, production code, and caveat.
- [`METHODOLOGY.md`](METHODOLOGY.md): experimental matrix and retained protocol.
- [`PROVENANCE.md`](PROVENANCE.md): separation of original Python references,
  Python recomputations, and new multilingual outputs.
- [`LIMITATIONS.md`](LIMITATIONS.md): interpretation boundaries.
- `validation/`: diagnostic source code and outputs for the program-disjoint
  probe check, PLBART reproduction investigation, and manual DFG/density audit.
- [`REPRODUCTION.md`](REPRODUCTION.md): compact package regeneration commands.

The large authoritative result trees are not duplicated here. Each inventory
row points to its canonical repository path. Exact input programs and probe
pair definitions remain in the repository; only reproducible model tensors and
runtime staging artifacts are omitted.

## Integrity

From the repository root:

```bash
/home/abhinav/miniconda3/envs/attention/bin/python results/build_inventory.py
/home/abhinav/miniconda3/envs/attention/bin/python results/validate_archive.py
(cd results && sha256sum -c checksums.sha256)
```

For complete extraction, GED, t-SNE, DirectProbe, validation, and report-build
commands, see the root [`ANALYSIS_RUNBOOK.md`](../ANALYSIS_RUNBOOK.md).
