"""Validate paired non-CodeBERT graph and hidden-representation artifacts."""

import argparse
import hashlib
import json
import os
import pickle

import numpy as np

from model_adapters import get_model_spec


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(graph_dir, embedding_dir, expected_model=None, expected_language=None,
             expected_count=None):
    graph_manifest_path = os.path.join(graph_dir, "graph_manifest.json")
    embedding_manifest_path = os.path.join(
        embedding_dir, "embedding_manifest.json"
    )
    with open(graph_manifest_path) as handle:
        graph_manifest = json.load(handle)
    with open(embedding_manifest_path) as handle:
        embedding_manifest = json.load(handle)

    errors = []
    model_name = expected_model or graph_manifest.get("model")
    try:
        spec = get_model_spec(model_name)
    except ValueError as exc:
        errors.append(str(exc))
        spec = None

    for label, manifest in (
        ("graph", graph_manifest),
        ("embedding", embedding_manifest),
    ):
        if manifest.get("status") != "complete":
            errors.append(f"{label} manifest is not complete")
        if manifest.get("model") != model_name:
            errors.append(f"{label} manifest model does not match {model_name}")
        if expected_language and manifest.get("language") != expected_language:
            errors.append(f"{label} manifest language mismatch")
        if expected_count is not None and manifest.get(
            "selected_num_codes"
        ) != expected_count:
            errors.append(f"{label} selected count mismatch")
        if len(manifest.get("artifacts", [])) + len(
            manifest.get("failures", [])
        ) != manifest.get("selected_num_codes"):
            errors.append(f"{label} artifact/failure accounting mismatch")

    graph_artifacts = graph_manifest.get("artifacts", [])
    embedding_artifacts = embedding_manifest.get("artifacts", [])
    if graph_artifacts != embedding_artifacts:
        errors.append("Graph and embedding artifact cohorts differ")
    if embedding_manifest.get("graph_manifest_sha256") != sha256_file(
        graph_manifest_path
    ):
        errors.append("Embedding manifest graph-manifest hash is stale")

    shapes = set()
    for artifact_name in graph_artifacts:
        graph_path = os.path.join(graph_dir, artifact_name)
        embedding_path = os.path.join(embedding_dir, artifact_name)
        if not os.path.isfile(graph_path) or not os.path.isfile(embedding_path):
            errors.append(f"Missing paired file for {artifact_name}")
            continue
        try:
            with open(graph_path, "rb") as handle:
                graph = pickle.load(handle)
            with open(embedding_path, "rb") as handle:
                embedding = pickle.load(handle)
        except Exception as exc:
            errors.append(
                f"Cannot load {artifact_name}: {type(exc).__name__}: {exc}"
            )
            continue

        code_tokens = graph.get("code_tokens", [])
        token_count = len(code_tokens)
        if code_tokens != graph.get("ast_tokens"):
            errors.append(f"AST token mismatch: {artifact_name}")
        if code_tokens != embedding.get("code_tokens"):
            errors.append(f"Graph/embedding token mismatch: {artifact_name}")
        for key in ("sample_index", "source_index"):
            if graph.get(key) != embedding.get(key):
                errors.append(f"{key} mismatch: {artifact_name}")

        attention = np.asarray(graph.get("model_graphs"))
        hidden = np.asarray(embedding.get("hidden_repr"))
        ast_graph = np.asarray(graph.get("ast_graph"))
        tree_dist = np.asarray(embedding.get("tree_dist"))
        if spec is not None:
            expected_attention = (
                spec.expected_transformer_layers,
                spec.expected_attention_heads,
                token_count,
                token_count,
            )
            expected_hidden = (
                spec.expected_transformer_layers + 1,
                token_count,
                spec.expected_hidden_size,
            )
            if attention.shape != expected_attention:
                errors.append(
                    f"Attention shape {attention.shape} != "
                    f"{expected_attention}: {artifact_name}"
                )
            if hidden.shape != expected_hidden:
                errors.append(
                    f"Hidden shape {hidden.shape} != "
                    f"{expected_hidden}: {artifact_name}"
                )
        if ast_graph.shape != (token_count, token_count):
            errors.append(f"AST shape mismatch: {artifact_name}")
        if tree_dist.shape != (token_count, token_count):
            errors.append(f"Tree-distance shape mismatch: {artifact_name}")
        if len(embedding.get("code_token_info", [])) != token_count:
            errors.append(f"Token-info length mismatch: {artifact_name}")
        if not np.isfinite(attention).all() or not np.isfinite(hidden).all():
            errors.append(f"Non-finite representation: {artifact_name}")
        shapes.add((tuple(attention.shape[:2]), tuple(hidden.shape[::2])))

    report = {
        "graph_dir": os.path.abspath(graph_dir),
        "embedding_dir": os.path.abspath(embedding_dir),
        "model": model_name,
        "language": graph_manifest.get("language"),
        "selected": graph_manifest.get("selected_num_codes"),
        "artifacts": len(graph_artifacts),
        "failures": len(graph_manifest.get("failures", [])),
        "observed_shapes": [
            {
                "attention_layers_heads": list(attention_shape),
                "hidden_states_dimension": list(hidden_shape),
            }
            for attention_shape, hidden_shape in sorted(shapes)
        ],
        "errors": errors,
        "valid": not errors,
    }
    print(json.dumps(report, indent=2))
    return not errors


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_dir", required=True)
    parser.add_argument("--embedding_dir", required=True)
    parser.add_argument("--expected_model")
    parser.add_argument(
        "--expected_language",
        choices=["python", "java", "go", "javascript"],
    )
    parser.add_argument("--expected_count", type=int)
    args = parser.parse_args()
    raise SystemExit(
        0
        if validate(
            args.graph_dir,
            args.embedding_dir,
            args.expected_model,
            args.expected_language,
            args.expected_count,
        )
        else 1
    )
