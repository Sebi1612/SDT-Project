# Final analysis status and locations

All analyses selected for the multilingual study are complete. The canonical
index is [`results/RESULT_INVENTORY.csv`](results/RESULT_INVENTORY.csv); it gives
the primary output, validation record, production script, provenance, and
caveat for each of 153 result groups.

## Completion matrix

| Analysis | Python | Java | JavaScript | Go |
| --- | --- | --- | --- | --- |
| Attention overlap, six models | complete recomputation and paper reference | complete | complete | complete |
| GED, three selected models | original paper reference | complete | complete | complete |
| t-SNE, six models (CodeBERT in main figure) | complete | complete | complete | complete |
| DirectProbe, three models/five tasks/three layers | original paper reference | complete | complete | complete |

## Canonical locations

- Final report, figures, tables, and derived summary: `results/report/`
- Validation audits: `results/validation/`
- Original Python attention/GED outputs: `attention/graph_comparision/`
- Original Python DirectProbe outputs: `DirectProbe/results/`
- CodeBERT final attention outputs: `analysis_results/attention_final_3000/`
- Other-model attention and hidden-state outputs:
  `analysis_results/final_3000_multimodel/`
- Main CodeBERT t-SNE results: `analysis_results/<language>/tsne/final_3000/`
- Multilingual GED validation: `analysis_results/ged_final_3000/validation/`
- Multilingual DirectProbe results: `DirectProbe/final_3000/results/`
- Exact DirectProbe pairs, labels, configs, and manifests:
  `DirectProbe/final_3000/data/` and `DirectProbe/final_3000/config_files/`

## Provenance boundary

The final comparative report uses original paper-repository values for Python
GED and DirectProbe. It uses the final-3000 Python recomputation for attention
overlap and t-SNE. Java, JavaScript, and Go results were produced by this
project. See `results/PROVENANCE.md` for details and
`results/LIMITATIONS.md` before interpreting cross-language differences.

Large raw attention/hidden-state tensors and probe embedding copies were
removed after validation because they are reproducible intermediates. This does
not remove the exact input programs, probe pair definitions, configurations, or
any final output used in the report.
