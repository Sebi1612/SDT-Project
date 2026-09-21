# Result locations

`<language>` = `java`, `go`, or `javascript`. These are the retained final results. The runbook writes reruns under `analysis_results/reproduction/` and `DirectProbe/reproduction_final_3000/` instead.

## 1. Attention analysis

- CodeBERT: `analysis_results/attention_final_3000/<language>/`
- Other models: `analysis_results/final_3000_multimodel/<language>/<model>/attention/`
- Models: `graphcodebert`, `unixcoder`, `plbart`, `codet5`, `codet5p_220`.
- Layer results: `ast/`, `dfg/`; summaries: `section_3_2_summary.json`, `section_3_2_summary_overlap.{csv,png}`.
- Original Python: `attention/graph_comparision/{ast,dfg}/exp_0/`.
- Comparison figures in `results/report/figures/`: `figure4_recall_four_languages.pdf`, `figure7_precision_four_languages.pdf`.

## 2. GED

- CodeBERT: `analysis_results/attention_final_3000/<language>/similarity_legacy/codebert/`
- GraphCodeBERT/CodeT5: `analysis_results/final_3000_multimodel/<language>/<model>/attention/similarity_legacy/<model>/`
- Layer results: `layer_<layer>_threshold_0.05.json`; per-program metrics: `program_metrics_threshold_0.05.npz`.
- Summaries: `section_3_2_summary_ged.{csv,png}` in the corresponding attention directory.
- Validation: `analysis_results/ged_final_3000/validation/<language>/<model>/ged_validation.json`.
- Original Python: `attention/graph_comparision/similarity/exp_0/<model>/`.
- Comparison figure: `results/report/figures/figure5_ged_four_languages.pdf`.

## 3. t-SNE

- CodeBERT: `analysis_results/<language>/tsne/final_3000/`
- Other models: `analysis_results/final_3000_multimodel/<language>/<model>/hidden_tsne/`
- Newly generated Python: the same paths with `<language>` = `python`.
- Models: `graphcodebert`, `unixcoder`, `plbart`, `codet5`, `codet5p_220`.
- Both layouts contain `token_types/` and `distances/`: PNG plots, CSV coordinates, and JSON manifests.
- Original Python PDFs in `attention/results/`: `tsne_hidden_cb_5.pdf`, `tsne_hidden_cb_12.pdf`, `tsne_dist_cb_5.pdf`.
- Comparison figures in `results/report/figures/`: `figure10_token_tsne_four_languages.pdf`, `figure11_distance_tsne_four_languages.pdf`; currently use newly generated Python CodeBERT panels.

## 4. DirectProbe

- Results: `DirectProbe/final_3000/results/<language>/<task>/<model>/<layer>/`
- Files: `clusters.txt`, `prediction.txt`, `dis.txt`, `log.txt`.
- Datasets/manifests: `DirectProbe/final_3000/data/<language>/<task>/<model>/`; large embedding matrices must be regenerated.
- Configurations: `DirectProbe/final_3000/config_files/`.
- Models: `codebert`, `graphcodebert`, `codet5`; layers: `5`, `9`, `12`.
- Tasks: `distance`, `distance_id`, `siblings`, `siblings_id`, `dfg`.
- Original Python: `DirectProbe/results/<task>/<model>/<layer>/`.
- Comparison figure: `results/report/figures/appendix_probe_layers_four_languages.pdf`.
