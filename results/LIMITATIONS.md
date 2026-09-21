# Limitations and improvement opportunities

## Limitations inherited from or shared with the paper

### Pair-level probing split

DirectProbe splits derived token-pair examples, not complete source programs.
Different pairs from one program may therefore occur in training and test.
This is appropriate for a like-for-like paper extension but does not cleanly
measure generalization to unseen programs. Across the 45 new-language layer-12
configurations, a median 98.96% of test programs also occur in training. A
separate program-disjoint linear control is lower in 35/45 cases, especially
for DFG; it is a diagnostic, not a replacement DirectProbe result.

### Representation evidence is indirect

Attention overlap and probe accuracy do not directly demonstrate program
understanding, causal use of a relation, or downstream behavioral competence.

### t-SNE is qualitative

Each language uses different programs and an independently fitted projection.
Coordinates cannot be aligned across panels, and visual separation can depend
on hyperparameters and random initialization.

### Test partition versus model pretraining

The cohorts come from the CodeSearchNet test partition, not its supervised
training partition. This does not prove that pretrained foundation models never
encountered the same repositories or code during pretraining.

## Multilingual-extension limitations

### Language-specific structural extraction

Each language uses its own Tree-sitter grammar and GraphCodeBERT-style DFG
extractor. A controlled example produces the same binary dependencies across
all four languages, but native signed labels differ: Python and Go versus Java
and JavaScript do not assign the two relation signs identically. Cross-language
DFG per-label results are therefore not a shared semantic ontology.

### Unequal graph and corpus characteristics

Program length, syntax, identifier frequency, and graph density differ by
language. In the complete GraphCodeBERT cohorts, mean DFG density differs
substantially. Observed language differences combine corpus, parser, label,
tokenizer, and model effects; they are not intrinsic language rankings.

### Mixed result provenance

Python GED and DirectProbe use stored paper outputs; new languages use validated
project runs. Python GED lacks the new manifests and was produced under a
different NetworkX environment, so absolute GED differences are descriptive.

### PLBART reproduction

PLBART's recomputed Python AST-recall curve differs more from the stored paper
reference than the other five attention models. Coverage, simple head ordering,
dropout, and a limited Transformers-version check do not explain it. Treat its
absolute multilingual levels as exploratory.

### Model scope

The paper includes additional large and fine-tuned models. The multilingual
extension covers six attention models and only three GED/probing models, as
defined by the project scope.

## Recommended improvements

1. Rerun DirectProbe on a fixed, balanced, program-disjoint 80:20 split.
2. Define and test a language-neutral DFG relation ontology.
3. Match or statistically control cohorts by tokens, AST size, DFG density,
   identifier frequency, and construct mix.
4. Pin model revisions, tokenizer files, libraries, random seeds, and container
   images in an immutable environment manifest.
5. Retain raw artifacts in versioned external storage with checksums.
6. Add repeated seeds and confidence intervals beyond attention overlap.
7. Add behavioral interventions or downstream tasks to test whether detected
   structure is actually used by the models.

