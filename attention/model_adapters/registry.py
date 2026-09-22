"""Single source of truth for non-CodeBERT paper model contracts."""

from typing import Dict, List

from .base import ModelAdapter, ModelSpec


MODEL_SPECS: Dict[str, ModelSpec] = {
    "graphcodebert": ModelSpec(
        name="graphcodebert",
        checkpoint="microsoft/graphcodebert-base",
        family="encoder-only",
        analyzed_component="encoder",
        input_protocol="plain_code_tokens",
        subtoken_marker="Ġ",
        prefix_tokens=1,
        suffix_tokens=1,
        expected_transformer_layers=12,
        expected_attention_heads=12,
        expected_hidden_size=768,
    ),
    "unixcoder": ModelSpec(
        name="unixcoder",
        checkpoint="microsoft/unixcoder-base",
        family="unified-transformer",
        analyzed_component="encoder-only mode",
        input_protocol="<s> <encoder-only> </s> code </s>",
        subtoken_marker="Ġ",
        prefix_tokens=3,
        suffix_tokens=1,
        expected_transformer_layers=12,
        expected_attention_heads=12,
        expected_hidden_size=768,
    ),
    "codet5": ModelSpec(
        name="codet5",
        checkpoint="Salesforce/codet5-base",
        family="encoder-decoder",
        analyzed_component="encoder",
        input_protocol="<s> code </s>",
        subtoken_marker="Ġ",
        prefix_tokens=1,
        suffix_tokens=1,
        expected_transformer_layers=12,
        expected_attention_heads=12,
        expected_hidden_size=768,
    ),
    "plbart": ModelSpec(
        name="plbart",
        checkpoint="uclanlp/plbart-base",
        family="encoder-decoder",
        analyzed_component="encoder",
        input_protocol="<s> code </s>",
        subtoken_marker="▁",
        prefix_tokens=1,
        suffix_tokens=1,
        expected_transformer_layers=6,
        expected_attention_heads=12,
        expected_hidden_size=768,
        # The shared cohort was selected with the paper's CodeBERT-token
        # cutoff. PLBART tokenizes one retained Python program to 501 pieces;
        # with its two special tokens this still fits within the common
        # 512-position inference window and must not reduce cross-model coverage.
        paper_max_subtokens=510,
    ),
    "codet5p_220": ModelSpec(
        name="codet5p_220",
        checkpoint="Salesforce/codet5p-220m",
        family="encoder-decoder",
        analyzed_component="encoder",
        input_protocol="<s> code </s>",
        subtoken_marker="Ġ",
        prefix_tokens=1,
        suffix_tokens=1,
        expected_transformer_layers=12,
        expected_attention_heads=12,
        expected_hidden_size=768,
    ),
    "codegen": ModelSpec(
        name="codegen",
        checkpoint="Salesforce/codegen2-3_7B",
        family="decoder-only",
        analyzed_component="causal decoder",
        input_protocol="code (no added special tokens)",
        subtoken_marker="Ġ",
        prefix_tokens=0,
        suffix_tokens=0,
        expected_transformer_layers=16,
        expected_attention_heads=16,
        expected_hidden_size=4096,
    ),
    "mamba": ModelSpec(
        name="mamba",
        checkpoint="state-spaces/mamba-370m-hf",
        family="state-space",
        analyzed_component="causal state-space backbone",
        input_protocol="code (no added special tokens)",
        subtoken_marker="Ġ",
        prefix_tokens=0,
        suffix_tokens=0,
        expected_transformer_layers=48,
        expected_attention_heads=0,
        expected_hidden_size=1024,
        supports_attention=False,
    ),
}

ALIASES = {
    "codet5plus_220m": "codet5p_220",
    "codet5+220m": "codet5p_220",
}


def _canonical_name(name: str) -> str:
    normalized = name.strip().lower()
    return ALIASES.get(normalized, normalized)


def get_model_spec(name: str) -> ModelSpec:
    canonical = _canonical_name(name)
    if canonical == "codebert":
        raise ValueError(
            "CodeBERT deliberately remains on the verified legacy entry points; "
            "use save_graph_info.py and save_word_embedding.py"
        )
    try:
        return MODEL_SPECS[canonical]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported adapter model {name!r}; choose one of "
            f"{', '.join(supported_model_names())}"
        ) from exc


def supported_model_names() -> List[str]:
    return list(MODEL_SPECS)


def create_model_adapter(
    name: str,
    device: str = "cuda:0",
    local_files_only: bool = False,
) -> ModelAdapter:
    # Kept lazy so metadata/tests do not import all heavyweight model classes.
    from .huggingface import ADAPTER_CLASSES

    spec = get_model_spec(name)
    if spec.name == "mamba":
        from .mamba import MambaAdapter

        adapter_class = MambaAdapter
    else:
        adapter_class = ADAPTER_CLASSES[spec.name]
    return adapter_class(
        spec=spec,
        device=device,
        local_files_only=local_files_only,
    )
