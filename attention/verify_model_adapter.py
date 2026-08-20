"""Bounded preflight/forward verification without saving analysis tensors."""

import argparse
import json
import os

from transformers import (
    AutoConfig,
    AutoTokenizer,
    PLBartTokenizer,
    RobertaTokenizer,
)

from extract_model_representations import (
    DEFAULT_GRAMMARS,
    ast_artifacts,
    load_selected_codes,
)
from model_adapters import create_model_adapter, get_model_spec, supported_model_names
from model_adapters.base import align_subtokens
from save_graph_info import build_parser


def load_metadata(spec, local_files_only):
    common = {
        "local_files_only": local_files_only,
        "trust_remote_code": spec.name == "codegen",
    }
    config = AutoConfig.from_pretrained(spec.checkpoint, **common)
    if spec.name in {"graphcodebert", "unixcoder", "codet5", "codet5p_220"}:
        common.pop("trust_remote_code")
        tokenizer = RobertaTokenizer.from_pretrained(spec.checkpoint, **common)
    elif spec.name == "plbart":
        common.pop("trust_remote_code")
        tokenizer = PLBartTokenizer.from_pretrained(spec.checkpoint, **common)
    else:
        tokenizer = AutoTokenizer.from_pretrained(spec.checkpoint, **common)
    return config, tokenizer


def config_dimensions(config):
    def first(names):
        for name in names:
            value = getattr(config, name, None)
            if value is not None:
                return int(value)
        return None

    return {
        "layers": first(
            ("num_hidden_layers", "num_layers", "n_layer", "encoder_layers")
        ),
        "heads": first(
            ("num_attention_heads", "num_heads", "n_head", "encoder_attention_heads")
        ),
        "hidden": first(("hidden_size", "d_model", "n_embd")),
    }


def write_report(path, report):
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w") as handle:
        json.dump(report, handle, indent=2)
    os.replace(temporary, path)


def verify(args):
    spec = get_model_spec(args.model)
    selected, dataset_size = load_selected_codes(
        args.code_file, args.num_codes, args.seed
    )
    parser, parser_library = build_parser(args.lang, args.grammar_repo)
    config, tokenizer = load_metadata(spec, args.local_files_only)
    observed = config_dimensions(config)
    expected = {
        "layers": spec.expected_transformer_layers,
        "heads": spec.expected_attention_heads,
        "hidden": spec.expected_hidden_size,
    }
    configuration_valid = observed == expected
    adapter = None
    if args.forward:
        adapter = create_model_adapter(
            args.model,
            device=args.device,
            local_files_only=args.local_files_only,
        ).load()

    failures = []
    max_subtokens = 0
    observed_shapes = set()
    for sample_index, (source_index, code) in enumerate(selected):
        try:
            code_tokens = list(code["code_tokens"])
            ast_artifacts(code["code"], code_tokens, parser)
            subtokens = tokenizer.tokenize(" ".join(code_tokens))
            groups = align_subtokens(
                subtokens, code_tokens, spec.subtoken_marker
            )
            if len(groups) != len(code_tokens):
                raise ValueError("Token alignment count mismatch")
            if len(subtokens) > spec.paper_max_subtokens:
                raise ValueError(
                    f"{len(subtokens)} subtokens exceed paper limit "
                    f"{spec.paper_max_subtokens}"
                )
            input_length = len(subtokens) + spec.prefix_tokens + spec.suffix_tokens
            model_capacity = getattr(config, "max_position_embeddings", None)
            if model_capacity is None:
                model_capacity = getattr(config, "n_positions", None)
            if model_capacity is not None and input_length > model_capacity:
                raise ValueError(
                    f"Input length {input_length} exceeds capacity {model_capacity}"
                )
            max_subtokens = max(max_subtokens, len(subtokens))
            if adapter is not None:
                output = adapter.extract(code_tokens)
                observed_shapes.add(
                    (
                        tuple(output.attention.shape[:2]),
                        (output.hidden_repr.shape[0], output.hidden_repr.shape[2]),
                    )
                )
        except Exception as exc:
            failures.append(
                {
                    "sample_index": sample_index,
                    "source_index": source_index,
                    "file_name": code.get("code_file"),
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )

    report = {
        "status": "complete",
        "mode": "forward" if args.forward else "metadata_tokenization_ast",
        "model": spec.name,
        "checkpoint": spec.checkpoint,
        "language": args.lang,
        "code_file": os.path.abspath(args.code_file),
        "dataset_size": dataset_size,
        "selected": len(selected),
        "passed": len(selected) - len(failures),
        "failed": len(failures),
        "coverage": (len(selected) - len(failures)) / len(selected),
        "configuration_expected": expected,
        "configuration_observed": observed,
        "configuration_valid": configuration_valid,
        "max_observed_subtokens": max_subtokens,
        "paper_max_subtokens": spec.paper_max_subtokens,
        "parser_library": os.path.abspath(parser_library),
        "observed_representation_shapes": [
            {
                "attention_layers_heads": list(attention_shape),
                "hidden_states_dimension": list(hidden_shape),
            }
            for attention_shape, hidden_shape in sorted(observed_shapes)
        ],
        "failures": failures,
        "valid": configuration_valid and not failures,
    }
    write_report(args.report, report)
    print(json.dumps(report, indent=2))
    return report


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=supported_model_names())
    parser.add_argument("--code_file", required=True)
    parser.add_argument("--lang", required=True, choices=list(DEFAULT_GRAMMARS))
    parser.add_argument("--num_codes", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--forward", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--grammar_repo")
    parser.add_argument("--report")
    args = parser.parse_args()
    args.grammar_repo = args.grammar_repo or DEFAULT_GRAMMARS[args.lang]
    report = verify(args)
    raise SystemExit(0 if report["valid"] else 1)


if __name__ == "__main__":
    cli()
