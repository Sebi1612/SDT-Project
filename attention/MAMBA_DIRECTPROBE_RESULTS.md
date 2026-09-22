# Mamba DirectProbe Results

Model: `state-spaces/mamba-370m-hf`

Representation: final hidden layer 48, hidden size 1024.

| Language | Dist. all | Dist. ID | Sib. all | Sib. ID | DFG |
|---|---:|---:|---:|---:|---:|
| Python | 0.494 | 0.545 | 0.750 | 0.787 | 0.826 |
| Java | 0.537 | 0.574 | 0.725 | 0.808 | 0.880 |
| Go | 0.580 | 0.661 | 0.789 | 0.833 | 0.847 |
| JavaScript | 0.482 | 0.630 | 0.723 | 0.837 | 0.908 |

Mamba is evaluated as a frozen code encoder. DirectProbe uses its
token-level hidden representations; Mamba is not fine-tuned.

Mamba does not expose Transformer self-attention. The analysis therefore
uses hidden representations only.

For JavaScript DFG, only 1,161 `Computed From` candidates were available.
The dataset was balanced by sampling 1,161 examples from each of the
`Computed From`, `No Edge`, and `Comes From` classes, using a fixed random
generator. Transformer comparisons remain directional unless those models
are rerun on the same sampled JavaScript pairs.
