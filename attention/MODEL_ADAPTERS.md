# Model adapter protocol

This additive pipeline covers GraphCodeBERT, UniXcoder, CodeT5, PLBART,
CodeT5+220M, and CodeGen2-3.7B. CodeBERT is deliberately excluded: its
verified `save_graph_info.py` and `save_word_embedding.py` paths remain
unchanged.

## Paper and implementation mapping

| Adapter | Checkpoint used by the repository | Analyzed representation | Input protocol |
|---|---|---|---|
| `graphcodebert` | `microsoft/graphcodebert-base` | RoBERTa encoder | Plain code subtokens |
| `unixcoder` | `microsoft/unixcoder-base` | Bidirectional encoder-only mode | `<s> <encoder-only> </s> code </s>` |
| `codet5` | `Salesforce/codet5-base` | Encoder | `<s> code </s>` |
| `plbart` | `uclanlp/plbart-base` | Encoder | `<s> code </s>` |
| `codet5p_220` | `Salesforce/codet5p-220m` | Encoder | `<s> code </s>` |
| `codegen` | `Salesforce/codegen2-3_7B` | Causal decoder | Code subtokens without added special tokens |

The paper describes DFG nodes as part of GraphCodeBERT's architecture and
pretraining. The repository's actual experimental extractor passes plain code
tokens to the GraphCodeBERT checkpoint. The adapter preserves that implemented
experiment rather than introducing a new DFG-input condition.

For CodeT5, CodeT5+220M, and PLBART, only the encoder is executed. These are the
states and attentions analyzed in the paper. On a real CodeT5 fixture, this
produced bit-identical encoder attention and hidden tensors to the old full
encoder-decoder call, while avoiding the irrelevant decoder/docstring pass.

Every adapter:

- loads one checkpoint once, uses `eval()` and inference mode;
- rejects silent truncation and enforces the paper's `<500` subtoken cohort;
- aligns model subtokens to the CodeSearchNet `code_tokens` nodes;
- averages each attention subtoken block and each hidden-state subtoken group;
- emits 4-D attention and 3-D hidden tensors with explicit layer semantics;
- records the checkpoint, input protocol, component, selection, parser, and
  software versions in manifests.

## Verification without saving analysis tensors

Run a 100-program preflight:

```bash
python attention/verify_model_adapter.py \
  --model graphcodebert \
  --code_file attention/exp_data/pilot_100/java.jsonl \
  --lang java \
  --local_files_only \
  --report analysis_results/model_adapter_verification/preflight_100/java/graphcodebert.json
```

Add `--forward --device cuda:0` to execute the checkpoint on all selected
programs while discarding the large tensors after validation. This is the
recommended last gate before scheduling a retained run.

## Paired retained extraction

The combined extractor computes attention and hidden states in one pass and
saves an identical artifact cohort for both analyses. Output directory
arguments are exact paths; no model name is appended implicitly.

```bash
python attention/extract_model_representations.py \
  --model graphcodebert \
  --code_file attention/exp_data/final_3000/java.jsonl \
  --graph_output_dir graph_info/final_3000/java/graphcodebert \
  --embedding_output_dir structural_probe/java/final_3000/graphcodebert \
  --lang java \
  --device cuda:0
```

Validate the paired artifacts before downstream analysis:

```bash
python attention/validate_representation_run.py \
  --graph_dir graph_info/final_3000/java/graphcodebert \
  --embedding_dir structural_probe/java/final_3000/graphcodebert \
  --expected_model graphcodebert \
  --expected_language java \
  --expected_count 3000
```

Do not use these commands for CodeBERT. No 3000-program extraction has been
started by the adapter verification work.
