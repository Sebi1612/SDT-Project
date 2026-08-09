The code in this directory is based on the
[GraphCodeBERT parser](https://github.com/microsoft/CodeBERT/tree/master/GraphCodeBERT).

The data-flow rules remain GraphCodeBERT-style rules. Small compatibility
changes support node names emitted by the tree-sitter grammar versions in this
repository, including JavaScript `assignment_expression` and Go
`short_var_declaration`/`range_clause`.
