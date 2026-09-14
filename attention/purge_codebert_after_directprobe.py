"""Safely purge retained CodeBERT tensors after DirectProbe data validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_staged_model_analysis import (
    atomic_json,
    purge_manifest_artifacts,
    utc_now,
)


TASKS = ["distance", "distance_id", "siblings", "siblings_id", "dfg"]
LAYERS = [5, 9, 12]


def load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def require_complete_sources(
    graph_dir: Path,
    embedding_dir: Path,
    dp_root: Path,
    language: str,
    expected_programs: int,
) -> list[str]:
    graph = load_json(graph_dir / "graph_manifest.json")
    embedding = load_json(embedding_dir / "embedding_manifest.json")
    errors = []
    if graph.get("status") != "complete":
        errors.append("graph manifest is not complete")
    if embedding.get("status") != "complete":
        errors.append("embedding manifest is not complete")
    if graph.get("model") != "codebert" or embedding.get("model") != "codebert":
        errors.append("model mismatch")
    if graph.get("lang") != language or embedding.get("language") != language:
        errors.append("language mismatch")
    graph_artifacts = graph.get("artifacts", [])
    embedding_artifacts = embedding.get("artifacts", [])
    if graph_artifacts != embedding_artifacts:
        errors.append("graph and embedding artifact cohorts differ")
    if len(graph_artifacts) != expected_programs:
        errors.append("artifact cohort is not the expected size")
    for directory, artifacts, label in (
        (graph_dir, graph_artifacts, "graph"),
        (embedding_dir, embedding_artifacts, "embedding"),
    ):
        missing = [name for name in artifacts if not (directory / name).is_file()]
        if missing:
            errors.append(f"{label} directory has {len(missing)} missing artifacts")

    attention_summary = (
        Path("analysis_results/attention_final_3000")
        / language
        / "section_3_2_summary.json"
    )
    token_tsne = (
        Path("analysis_results")
        / language
        / "tsne"
        / "final_3000"
        / "token_types"
        / "token_type_tsne_manifest.json"
    )
    distance_tsne = (
        Path("analysis_results")
        / language
        / "tsne"
        / "final_3000"
        / "distances"
        / "distance_tsne_manifest.json"
    )
    for path in (attention_summary, token_tsne, distance_tsne):
        if not path.is_file():
            errors.append(f"permanent analysis output is missing: {path}")

    for task in TASKS:
        data_dir = dp_root / "data" / language / task / "codebert"
        manifest_path = data_dir / "dataset_manifest.json"
        if not manifest_path.is_file():
            errors.append(f"DirectProbe manifest is missing: {manifest_path}")
            continue
        task_manifest = load_json(manifest_path)
        if task_manifest.get("status") != "complete":
            errors.append(f"DirectProbe dataset is incomplete: {language}/{task}")
        if task_manifest.get("model") != "codebert":
            errors.append(f"DirectProbe model mismatch: {language}/{task}")
        if task_manifest.get("language") != language:
            errors.append(f"DirectProbe language mismatch: {language}/{task}")
        available_layers = task_manifest.get("layers", [])
        if not all(layer in available_layers for layer in LAYERS):
            errors.append(f"DirectProbe selected layers missing: {language}/{task}")
        for layer in LAYERS:
            for split in ("train", "test"):
                path = (
                    data_dir
                    / "embeddings"
                    / "layers"
                    / split
                    / f"{layer}.txt"
                )
                if not path.is_file() or path.stat().st_size == 0:
                    errors.append(f"DirectProbe data missing: {path}")
            config = (
                dp_root
                / "config_files"
                / language
                / task
                / f"config_codebert_{layer}.ini"
            )
            if not config.is_file():
                errors.append(f"DirectProbe config missing: {config}")
    return errors


def main() -> None:
    cli = argparse.ArgumentParser()
    cli.add_argument(
        "--languages", nargs="+", default=["java", "go", "javascript"]
    )
    cli.add_argument("--graph_root", default="graph_info/final_3000")
    cli.add_argument("--embedding_root", default="structural_probe")
    cli.add_argument("--dp_root", default="DirectProbe/final_3000")
    cli.add_argument("--expected_programs", type=int, default=3000)
    cli.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate every prerequisite and report the exact deletion scope.",
    )
    cli.add_argument(
        "--manifest",
        default="DirectProbe/final_3000/codebert_source_cleanup_manifest.json",
    )
    args = cli.parse_args()

    output = {
        "status": "in_progress",
        "purpose": (
            "manifest-scoped cleanup after permanent attention, t-SNE, and "
            "DirectProbe datasets were validated"
        ),
        "started_utc": utc_now(),
        "model": "codebert",
        "languages": args.languages,
        "required_tasks": TASKS,
        "required_layers": LAYERS,
        "runs": {},
    }
    manifest_path = Path(args.manifest)
    atomic_json(manifest_path, output)
    try:
        for language in args.languages:
            graph_dir = Path(args.graph_root) / language / "codebert"
            embedding_dir = (
                Path(args.embedding_root) / language / "final_3000" / "codebert"
            )
            errors = require_complete_sources(
                graph_dir,
                embedding_dir,
                Path(args.dp_root),
                language,
                args.expected_programs,
            )
            if errors:
                raise ValueError(f"{language}: " + "; ".join(errors))
            if args.dry_run:
                graph = load_json(graph_dir / "graph_manifest.json")
                embedding = load_json(embedding_dir / "embedding_manifest.json")
                graph_paths = [graph_dir / name for name in graph["artifacts"]]
                embedding_paths = [
                    embedding_dir / name for name in embedding["artifacts"]
                ]
                output["runs"][language] = {
                    "status": "validated",
                    "would_purge_artifacts": len(graph_paths) + len(embedding_paths),
                    "would_purge_bytes": sum(
                        path.stat().st_size for path in graph_paths + embedding_paths
                    ),
                }
                atomic_json(manifest_path, output)
                continue
            graph_bytes, graph_count = purge_manifest_artifacts(
                graph_dir,
                "graph_manifest.json",
                Path("analysis_results/attention_final_3000") / language,
            )
            embedding_bytes, embedding_count = purge_manifest_artifacts(
                embedding_dir,
                "embedding_manifest.json",
                Path(args.dp_root) / "data" / language,
            )
            output["runs"][language] = {
                "status": "complete",
                "validated_directprobe_tasks": TASKS,
                "validated_directprobe_layers": LAYERS,
                "purged_artifacts": graph_count + embedding_count,
                "purged_bytes": graph_bytes + embedding_bytes,
                "reextraction_required_for": ["GED", "new probing datasets"],
                "finished_utc": utc_now(),
            }
            atomic_json(manifest_path, output)
    except Exception as exc:
        output.update(
            {
                "status": "failed",
                "failed_utc": utc_now(),
                "reason": f"{type(exc).__name__}: {exc}",
            }
        )
        atomic_json(manifest_path, output)
        raise

    if args.dry_run:
        output.update(
            {
                "status": "dry_run_complete",
                "finished_utc": utc_now(),
                "total_would_purge_artifacts": sum(
                    run["would_purge_artifacts"] for run in output["runs"].values()
                ),
                "total_would_purge_bytes": sum(
                    run["would_purge_bytes"] for run in output["runs"].values()
                ),
            }
        )
        atomic_json(manifest_path, output)
        print(
            "[CODEBERT SOURCE CLEANUP DRY RUN] "
            f"{output['total_would_purge_artifacts']} files, "
            f"{output['total_would_purge_bytes'] / 1e9:.2f} GB"
        )
        return

    output.update(
        {
            "status": "complete",
            "finished_utc": utc_now(),
            "total_purged_artifacts": sum(
                run["purged_artifacts"] for run in output["runs"].values()
            ),
            "total_purged_bytes": sum(
                run["purged_bytes"] for run in output["runs"].values()
            ),
        }
    )
    atomic_json(manifest_path, output)
    print(
        "[CODEBERT SOURCE CLEANUP COMPLETE] "
        f"{output['total_purged_artifacts']} files, "
        f"{output['total_purged_bytes'] / 1e9:.2f} GB"
    )


if __name__ == "__main__":
    main()
