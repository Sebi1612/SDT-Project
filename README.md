# Multilingual structural evaluation of code language models

This repository is the final artifact for a multilingual extension of Anand et
al., *A Critical Study of What Code-LLMs (Do Not) Learn*. The original work
studied Python; this project applies the same analysis families to Java,
JavaScript, and Go and compares all four languages.

The repository contains the exact sampled program cohorts, the original Python
reference outputs, the new multilingual outputs, the code that produced them,
validation audits, and a compiled four-language result report. Large model
representations that can be regenerated from the retained inputs are not kept.

## Start here

1. Read the final report: [`results/report/four_language_results.pdf`](results/report/four_language_results.pdf).
2. Browse every result group and its provenance in
   [`results/RESULT_INVENTORY.csv`](results/RESULT_INVENTORY.csv).
3. Read [`results/METHODOLOGY.md`](results/METHODOLOGY.md),
   [`results/PROVENANCE.md`](results/PROVENANCE.md), and
   [`results/LIMITATIONS.md`](results/LIMITATIONS.md).
4. Use [`ANALYSIS_RUNBOOK.md`](ANALYSIS_RUNBOOK.md) for exact reproduction and
   report-generation commands.

## Experimental scope

| Analysis | Models | Languages | Final evidence |
| --- | --- | --- | --- |
| AST/DFG attention overlap | CodeBERT, GraphCodeBERT, UniXcoder, PLBART, CodeT5, CodeT5+220M | Python, Java, JavaScript, Go | per-layer JSON/CSV summaries and comparative figures |
| Graph edit distance (GED) | CodeBERT, GraphCodeBERT, CodeT5 | Python reference plus new Java, JavaScript, Go runs | 12 layer files per model/language and validation manifests |
| t-SNE hidden-state visualizations | CodeBERT in the main comparison; all six models archived | Python, Java, JavaScript, Go | token-type and AST-distance plots with manifests |
| DirectProbe | CodeBERT, GraphCodeBERT, CodeT5 | Python reference plus new Java, JavaScript, Go runs | five tasks at layers 5, 9, and 12 |

The main cohorts contain 3,000 programs from the CodeSearchNet test partition.
Python uses the paper repository's `exp_0.jsonl`; the added-language cohorts and
their selection manifests are in `attention/exp_data/final_3000/`.

## Repository layout

```text
.
├── README.md                     project entry point
├── ANALYSIS_RUNBOOK.md           commands for every analysis and validation
├── ANALYSIS_RESULTS_SUMMARY.md   completion matrix and canonical locations
├── results/                      final human- and machine-readable result package
│   ├── report/                   final PDF, LaTeX, figures, tables, summary JSON
│   ├── validation/               split, PLBART, and DFG audit code/results
│   └── RESULT_INVENTORY.csv      index of 153 result groups
├── attention/                    extraction and structural-analysis source code
│   ├── exp_data/                 exact final program cohorts
│   └── graph_comparision/        original Python paper outputs
├── analysis_results/             authoritative multilingual analysis outputs
├── DirectProbe/                  probe implementation, data definitions, configs,
│   ├── results/                  original Python paper outputs
│   └── final_3000/               multilingual pairs, configs, manifests, results
├── parser/                       DFG extraction implementation
└── tree-sitter-{python,java,go,javascript}/
                                  language grammar sources
```

## Data and storage policy

The following data needed to inspect or reproduce the study are versioned:

- the exact 3,000-program input cohort for every language;
- cohort hashes, filtering statistics, seeds, and source-partition manifests;
- the exact DirectProbe train/test entity pairs, labels, configurations, and
  dataset manifests;
- all final metric outputs, figures, tables, validation records, and original
  Python reference results.

Downloaded full CodeSearchNet exports, model checkpoints, attention tensors,
hidden-state tensors, and text copies of probe embedding matrices are
reproducible intermediates and are intentionally excluded. Keeping them would
add tens of gigabytes without changing any reported value. The runbook explains
how to regenerate them, and the final manifests record the inputs and settings.

## Environment

The project was run on Ubuntu with Python 3.9. The established environment in
this workspace is `/home/abhinav/miniconda3/envs/attention/bin/python`.
Dependencies are listed in `attention/requirements.txt` and
`DirectProbe/requirements.txt`. Model checkpoints must either be present in the
Hugging Face cache or downloaded before an offline run.

DirectProbe must use the project-local Gurobi licence on every invocation:

```bash
GRB_LICENSE_FILE=/home/abhinav/sdt_project/.config/gurobi/dev-sebastian/gurobi.lic \
  /home/abhinav/miniconda3/envs/attention/bin/python DirectProbe/main.py --help
```

This does not modify the global Gurobi configuration.

## Upstream sources

- Original paper repository: <https://github.com/stg-tud/code-LLM-critical-evaluation>
- DirectProbe: <https://github.com/utahnlp/DirectProbe>

See the root `LICENSE` and `DirectProbe/LICENSE` for licensing information.
