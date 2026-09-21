"""Extract attention, AST, and hidden-state artifacts with one model pass.

This is an additive entry point for the paper's non-CodeBERT models.  The
verified CodeBERT entry points are intentionally unchanged and unsupported by
this command.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import pickle
import random
import re
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
import transformers
from tqdm import tqdm

from graph_utils import get_ast_tokens_and_prog_graphs, tokens_to_graph, traverse_node
from model_adapters import create_model_adapter, supported_model_names
from save_graph_info import build_parser
from save_word_embedding import aligned_ast_structure


DEFAULT_GRAMMARS = {
    "python": "attention/tree-sitter-python",
    "java": "tree-sitter-java",
    "go": "tree-sitter-go",
    "javascript": "tree-sitter-javascript",
}


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: str, value: Dict[str, Any]) -> None:
    temporary = path + ".tmp"
    with open(temporary, "w") as handle:
        json.dump(value, handle, indent=2)
    os.replace(temporary, path)


def atomic_pickle(path: str, value: Dict[str, Any]) -> None:
    temporary = path + ".tmp"
    with open(temporary, "wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def load_selected_codes(
    code_file: str, num_codes: int, seed: int
) -> Tuple[List[Tuple[int, Dict[str, Any]]], int]:
    with open(code_file) as handle:
        all_codes = [json.loads(line) for line in handle]
    if num_codes is None:
        indices: Iterable[int] = range(len(all_codes))
    else:
        if num_codes <= 0:
            raise ValueError("--num_codes must be positive")
        if num_codes > len(all_codes):
            raise ValueError(
                f"Requested {num_codes} samples from a {len(all_codes)}-sample file"
            )
        indices = random.Random(seed).sample(range(len(all_codes)), num_codes)
    return [(index, all_codes[index]) for index in indices], len(all_codes)


def safe_artifact_name(sample_index: int, file_name: str) -> str:
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", file_name).strip("._")
    if not safe_name:
        safe_name = "sample"
    return f"{sample_index:06d}_{safe_name}.pkl"


def ast_artifacts(code: str, code_tokens: List[str], parser):
    byte_code = code.encode("utf-8")
    tree = parser.parse(byte_code)
    parser_recovered = tree.root_node.has_error
    collected = []
    traverse_node(
        tree.root_node,
        collected,
        byte_code,
        include_comments=False,
    )
    ast_info, _, is_error = get_ast_tokens_and_prog_graphs(
        collected,
        code_tokens,
        code_tokens,
        byte_code,
        (0, 0),
    )
    ast_tokens = [info["token"] for info in ast_info]
    if is_error or ast_tokens != code_tokens:
        raise ValueError("AST tokens do not exactly match dataset tokens")
    return ast_tokens, tokens_to_graph(ast_info), parser_recovered


def base_manifests(args, adapter, parser_library, selected, dataset_size):
    selection = [
        {
            "sample_index": sample_index,
            "source_index": source_index,
            "file_name": code["code_file"],
        }
        for sample_index, (source_index, code) in enumerate(selected)
    ]
    common = {
        "status": "in_progress",
        "code_file": os.path.abspath(args.code_file),
        "code_file_sha256": sha256_file(args.code_file),
        "language": args.lang,
        "model": adapter.spec.name,
        "model_version": adapter.spec.checkpoint,
        "adapter_contract": adapter.spec.to_dict(),
        "seed": args.seed,
        "device": args.device,
        "random_model": False,
        "inference_mode": True,
        "local_files_only": args.local_files_only,
        "dataset_size": dataset_size,
        "requested_num_codes": args.num_codes,
        "selected_num_codes": len(selected),
        "selection": selection,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "tree_sitter_version": importlib.metadata.version("tree-sitter"),
        "tree_sitter_language_library": os.path.abspath(parser_library),
        "tree_sitter_language_library_sha256": sha256_file(parser_library),
        "grammar_repo": os.path.abspath(args.grammar_repo),
        "artifacts": [],
        "failures": [],
    }
    graph = {**common, "artifacts": [], "failures": []}
    # dfg_comp.py uses the older key spelling.
    graph["lang"] = args.lang
    hidden = {**common, "artifacts": [], "failures": []}
    graph["parse_recoveries"] = []
    hidden["parse_recoveries"] = []
    hidden["graph_manifest"] = os.path.abspath(
        os.path.join(args.graph_output_dir, "graph_manifest.json")
    )
    return graph, hidden


def extract(args):
    if os.path.abspath(args.graph_output_dir) == os.path.abspath(
        args.embedding_output_dir
    ):
        raise ValueError("Graph and embedding output directories must differ")
    if args.model == "codebert":
        raise ValueError(
            "CodeBERT remains on save_graph_info.py/save_word_embedding.py"
        )

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    selected, dataset_size = load_selected_codes(
        args.code_file, args.num_codes, args.seed
    )
    parser, parser_library = build_parser(args.lang, args.grammar_repo)

    print(
        f"Loading {args.model} once on {args.device}; "
        f"selected {len(selected)}/{dataset_size} samples"
    )
    adapter = create_model_adapter(
        args.model,
        device=args.device,
        local_files_only=args.local_files_only,
    ).load()

    os.makedirs(args.graph_output_dir, exist_ok=True)
    os.makedirs(args.embedding_output_dir, exist_ok=True)
    graph_manifest, embedding_manifest = base_manifests(
        args, adapter, parser_library, selected, dataset_size
    )
    graph_manifest_path = os.path.join(
        args.graph_output_dir, "graph_manifest.json"
    )
    embedding_manifest_path = os.path.join(
        args.embedding_output_dir, "embedding_manifest.json"
    )
    atomic_json(graph_manifest_path, graph_manifest)
    atomic_json(embedding_manifest_path, embedding_manifest)

    for sample_index, (source_index, code) in enumerate(tqdm(selected)):
        file_name = code["code_file"]
        artifact_name = safe_artifact_name(sample_index, file_name)
        try:
            code_tokens = list(code["code_tokens"])
            ast_tokens, ast_graph, parser_recovered = ast_artifacts(
                code["code"], code_tokens, parser
            )
            output = adapter.extract(code_tokens)
            tree_dist, code_token_info = aligned_ast_structure(
                code["code"], code_tokens, parser, args.lang
            )
            if tree_dist.shape != (len(code_tokens), len(code_tokens)):
                raise ValueError("Tree-distance shape does not match code tokens")

            graph_payload = {
                "file_name": file_name,
                "sample_index": sample_index,
                "source_index": source_index,
                "code": code["code"],
                "lang": args.lang,
                "model_tokens": [token.replace(" ", "") for token in code_tokens],
                "code_tokens": code_tokens,
                "ast_tokens": ast_tokens,
                "model_graphs": output.attention,
                "ast_graph": ast_graph,
                "adapter_metadata": output.metadata,
                "subtokens": output.subtokens,
                "subtoken_groups": output.subtoken_groups,
            }
            embedding_payload = {
                "hidden_repr": output.hidden_repr,
                "tree_dist": np.asarray(tree_dist),
                "code_token_info": code_token_info,
                "code_tokens": code_tokens,
                "code_file": file_name,
                "sample_index": sample_index,
                "source_index": source_index,
                "source_graph_artifact": artifact_name,
                "language": args.lang,
                "adapter_metadata": output.metadata,
            }
            atomic_pickle(
                os.path.join(args.graph_output_dir, artifact_name),
                graph_payload,
            )
            atomic_pickle(
                os.path.join(args.embedding_output_dir, artifact_name),
                embedding_payload,
            )
            graph_manifest["artifacts"].append(artifact_name)
            embedding_manifest["artifacts"].append(artifact_name)
            if parser_recovered:
                recovery = {
                    "sample_index": sample_index,
                    "source_index": source_index,
                    "file_name": file_name,
                    "reason": (
                        "Tree-sitter error recovery accepted after exact "
                        "dataset-token alignment"
                    ),
                }
                graph_manifest["parse_recoveries"].append(recovery)
                embedding_manifest["parse_recoveries"].append(recovery)
        except Exception as exc:
            failure = {
                "sample_index": sample_index,
                "source_index": source_index,
                "file_name": file_name,
                "reason": f"{type(exc).__name__}: {exc}",
            }
            graph_manifest["failures"].append(failure)
            embedding_manifest["failures"].append(failure)
            print(
                f"Sample {sample_index} ({file_name}) failed: "
                f"{failure['reason']}"
            )
        atomic_json(graph_manifest_path, graph_manifest)
        atomic_json(embedding_manifest_path, embedding_manifest)

    graph_manifest["status"] = "complete"
    graph_manifest["num_saved"] = len(graph_manifest["artifacts"])
    graph_manifest["num_failures"] = len(graph_manifest["failures"])
    graph_manifest["num_parse_recoveries"] = len(
        graph_manifest["parse_recoveries"]
    )
    atomic_json(graph_manifest_path, graph_manifest)

    embedding_manifest["status"] = "complete"
    embedding_manifest["num_saved"] = len(embedding_manifest["artifacts"])
    embedding_manifest["num_failures"] = len(embedding_manifest["failures"])
    embedding_manifest["num_parse_recoveries"] = len(
        embedding_manifest["parse_recoveries"]
    )
    embedding_manifest["graph_manifest_sha256"] = sha256_file(
        graph_manifest_path
    )
    atomic_json(embedding_manifest_path, embedding_manifest)
    print(
        f"[EXTRACTION COMPLETE] {args.model}/{args.lang}: "
        f"{graph_manifest['num_saved']}/{len(selected)} shared graph+hidden "
        f"artifacts"
    )
    return graph_manifest, embedding_manifest


def cli():
    parser = argparse.ArgumentParser(
        description=(
            "Extract paper-compatible attention and hidden states for one "
            "non-CodeBERT model. Output directory arguments are exact paths."
        )
    )
    parser.add_argument(
        "--model", required=True, choices=supported_model_names()
    )
    parser.add_argument("--code_file", required=True)
    parser.add_argument("--graph_output_dir", required=True)
    parser.add_argument("--embedding_output_dir", required=True)
    parser.add_argument("--num_codes", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument(
        "--lang",
        required=True,
        choices=["python", "java", "go", "javascript"],
    )
    parser.add_argument("--grammar_repo")
    args = parser.parse_args()
    args.grammar_repo = args.grammar_repo or DEFAULT_GRAMMARS[args.lang]
    extract(args)


if __name__ == "__main__":
    cli()
