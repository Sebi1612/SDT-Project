# Methodology and experimental coverage

## Research question

The original paper studies whether code-language-model attention and hidden
representations encode abstract syntax tree (AST) and data-flow graph (DFG)
relations in Python. This project asks which findings persist for Java,
JavaScript, and Go when the paper's experimental approach is extended to those
languages.

## Shared protocol

The extension retains the following choices where applicable:

- 3,000 programs sampled from the CodeSearchNet test partition;
- removal of comments and docstrings before structural analysis;
- model-native tokenization aligned to dataset tokens;
- AST and GraphCodeBERT-style DFG extraction;
- attention threshold 0.05 and best-F-score-head selection;
- directed attention/structure overlap with macro averaging over programs;
- paper-compatible legacy graph-edit distance normalized per node;
- the five DirectProbe tasks and balanced labels used by the paper;
- an 80:20 pair-level DirectProbe split, matching the implemented paper-style
  protocol;
- layers 5, 9, and 12 for the selected 12-layer probe models, with layer 12 in
  the main tables;
- the paper's qualitative t-SNE analysis structure.

Language-specific Tree-sitter grammars and DFG extractors are necessary
adaptations, not changes to the research question.

## Coverage matrix

| Analysis | Python | Java | JavaScript | Go | Models |
| --- | --- | --- | --- | --- | --- |
| AST/DFG attention overlap | Yes | Yes | Yes | Yes | CodeBERT, GraphCodeBERT, UniXcoder, PLBART, CodeT5, CodeT5+220M |
| GED | Paper reference | New run | New run | New run | CodeBERT, GraphCodeBERT, CodeT5 |
| Token/distance t-SNE | Yes | Yes | Yes | Yes | CodeBERT in the main comparison; supplemental outputs for all six attention models |
| DirectProbe | Paper reference | New run | New run | New run | CodeBERT, GraphCodeBERT, CodeT5 |

## Comparability boundaries

“Same approach” means the same analysis definitions, tasks, sampling targets,
layer conventions, and result selection. It does not mean that the underlying
programs or structural extractors are identical. Programs necessarily differ
by language, and equivalent language constructs may be represented differently
by language-specific parsers.

For some multilingual probe tasks the number of source programs considered was
expanded to reach the paper's target count of balanced token pairs. Final task
sizes and labels were preserved, but the source-program composition therefore
differs across languages and tasks.
