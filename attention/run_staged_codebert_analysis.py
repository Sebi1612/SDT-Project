"""Run a storage-bounded final CodeBERT analysis.

CodeBERT deliberately uses the repository's established extraction programs.
This wrapper gives that path the same validation, logging, Python-reference
comparison, and manifest-scoped cleanup guarantees as the adapter runner.
GED and DirectProbe are intentionally excluded.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from run_staged_model_analysis import (
    atomic_json,
    load_json,
    purge_manifest_artifacts,
    run_command,
    utc_now,
)


def validate_outputs(attention_dir: Path, hidden_dir: Path, expected: int) -> dict:
    summary = load_json(attention_dir / "section_3_2_summary.json")
    comparison = load_json(attention_dir / "python_reference_comparison.json")
    token_manifest = load_json(
        hidden_dir / "token_types" / "token_type_tsne_manifest.json"
    )
    distance_manifest = load_json(
        hidden_dir / "distances" / "distance_tsne_manifest.json"
    )
    errors = []
    if summary.get("status") != "complete":
        errors.append("attention summary is not complete")
    if summary.get("model") != "codebert" or summary.get("language") != "python":
        errors.append("attention summary model/language mismatch")
    if summary.get("expected_programs") != expected:
        errors.append("attention summary program count mismatch")
    if summary.get("ged_status") != "skipped":
        errors.append("GED was not skipped")
    if comparison.get("status") != "complete":
        errors.append("Python paper-reference comparison is not complete")
    if comparison.get("ged_comparison_status") != "skipped":
        errors.append("Python comparison unexpectedly includes GED")
    for name, manifest, perplexities in (
        ("token", token_manifest, [50]),
        ("distance", distance_manifest, [5, 10]),
    ):
        if manifest.get("model") != "codebert":
            errors.append(f"{name} t-SNE model mismatch")
        if manifest.get("language") != "python":
            errors.append(f"{name} t-SNE language mismatch")
        if manifest.get("layers") != [5]:
            errors.append(f"{name} t-SNE layer mismatch")
        if manifest.get("perplexities") != perplexities:
            errors.append(f"{name} t-SNE perplexity mismatch")
        if manifest.get("iterations") != 50000:
            errors.append(f"{name} t-SNE iteration mismatch")
    return {
        "status": "complete" if not errors else "failed",
        "model": "codebert",
        "language": "python",
        "expected_programs": expected,
        "ged": "skipped",
        "directprobe": "skipped",
        "paper_reference_comparison": str(
            (attention_dir / "python_reference_comparison.json").resolve()
        ),
        "errors": errors,
        "valid": not errors,
    }


def complete_staged_representations(
    graph_dir: Path,
    embedding_dir: Path,
    expected: int,
) -> bool:
    """Return true only for a complete, lossless paired retained cohort."""
    try:
        graph = load_json(graph_dir / "graph_manifest.json")
        embedding = load_json(embedding_dir / "embedding_manifest.json")
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    for manifest, directory in (
        (graph, graph_dir),
        (embedding, embedding_dir),
    ):
        artifacts = manifest.get("artifacts", [])
        if (
            manifest.get("status") != "complete"
            or manifest.get("model") != "codebert"
            or manifest.get("num_saved") != expected
            or manifest.get("num_failures") != 0
            or manifest.get("failures")
            or len(artifacts) != expected
            or any(not (directory / name).is_file() for name in artifacts)
        ):
            return False
    return graph.get("artifacts") == embedding.get("artifacts")


def main() -> None:
    cli = argparse.ArgumentParser()
    cli.add_argument("--dataset_root", default="attention/exp_data")
    cli.add_argument("--dataset_pattern", default="exp_0.jsonl")
    cli.add_argument(
        "--graph_stage_root", default="graph_info/staged_python_final_3000"
    )
    cli.add_argument(
        "--embedding_stage_root",
        default="structural_probe/staged_python_final_3000",
    )
    cli.add_argument(
        "--attention_results_root", default="analysis_results/attention_final_3000"
    )
    cli.add_argument(
        "--hidden_results_root", default="analysis_results/python/tsne/final_3000"
    )
    cli.add_argument(
        "--run_manifest",
        default="analysis_results/attention_final_3000/python_run_manifest.json",
    )
    cli.add_argument(
        "--python_reference_root", default="attention/graph_comparision"
    )
    cli.add_argument("--expected_programs", type=int, default=3000)
    cli.add_argument("--device", default="cpu")
    cli.add_argument("--minimum_free_gb", type=float, default=45.0)
    cli.add_argument("--omp_threads", type=int, default=4)
    args = cli.parse_args()

    free_gb = shutil.disk_usage(Path.cwd()).free / 1e9
    if free_gb < args.minimum_free_gb:
        raise RuntimeError(
            f"Only {free_gb:.1f} GB free; require {args.minimum_free_gb:.1f} GB"
        )

    code_file = Path(args.dataset_root) / args.dataset_pattern
    graph_language_root = Path(args.graph_stage_root) / "python"
    graph_dir = graph_language_root / "codebert"
    embedding_dir = (
        Path(args.embedding_stage_root) / "python" / "final_3000" / "codebert"
    )
    attention_dir = Path(args.attention_results_root) / "python"
    hidden_dir = Path(args.hidden_results_root)
    log_dir = attention_dir / "logs"
    manifest_path = Path(args.run_manifest)

    if manifest_path.exists():
        manifest = load_json(manifest_path)
    else:
        manifest = {}
    validation_path = attention_dir / "python_final_validation.json"
    if manifest.get("status") == "complete" and validation_path.is_file():
        validation = load_json(validation_path)
        if validation.get("valid"):
            print("[SKIP] python/codebert already complete and validated")
            return
    manifest.update({
        "status": "in_progress",
        "purpose": "storage-bounded Python CodeBERT non-GED/non-DirectProbe analysis",
        "started_utc": manifest.get("started_utc", utc_now()),
        "python": sys.executable,
        "model": "codebert",
        "language": "python",
        "code_file": str(code_file.resolve()),
        "expected_programs": args.expected_programs,
        "device": args.device,
        "omp_threads": args.omp_threads,
        "phases": manifest.get("phases", {}),
    })
    atomic_json(manifest_path, manifest)

    environment = os.environ.copy()
    environment.update(
        {
            "OMP_NUM_THREADS": str(args.omp_threads),
            "MKL_NUM_THREADS": str(args.omp_threads),
            "OPENBLAS_NUM_THREADS": str(args.omp_threads),
            "TOKENIZERS_PARALLELISM": "false",
            "MPLCONFIGDIR": "/tmp/sdt-python-codebert-matplotlib",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    python = sys.executable
    phases = [
        (
            "extract_graphs",
            [
                python,
                "attention/save_graph_info.py",
                "--model",
                "codebert",
                "--code_file",
                str(code_file),
                "--save_dir",
                str(graph_language_root),
                "--lang",
                "python",
                "--device",
                args.device,
                "--seed",
                "0",
            ],
        ),
        (
            "validate_graphs",
            [
                python,
                "attention/validate_graph_run.py",
                "--graph_dir",
                str(graph_dir),
                "--expected_count",
                str(args.expected_programs),
                "--expected_model",
                "codebert",
                "--expected_language",
                "python",
                "--require_deterministic",
                "--require_parser_provenance",
                "--minimum_coverage",
                "1.0",
            ],
        ),
        (
            "extract_hidden_states",
            [
                python,
                "attention/save_word_embedding.py",
                "--model",
                "codebert",
                "--code_file",
                str(code_file),
                "--graph_loc",
                str(graph_dir),
                "--save_dir",
                str(Path(args.embedding_stage_root)),
                "--exp_name",
                "final_3000",
                "--lang",
                "python",
                "--device",
                args.device,
                "--seed",
                "0",
            ],
        ),
        (
            "validate_representations",
            [
                python,
                "attention/validate_representation_run.py",
                "--graph_dir",
                str(graph_dir),
                "--embedding_dir",
                str(embedding_dir),
                "--expected_model",
                "codebert",
                "--expected_language",
                "python",
                "--expected_count",
                str(args.expected_programs),
            ],
        ),
        (
            "ast_overlap",
            [
                python,
                "attention/graph_comp.py",
                "--graph_loc",
                str(graph_dir),
                "--save_dir",
                str(attention_dir),
                "--all_layers",
                "--bootstrap_samples",
                "1000",
            ],
        ),
        (
            "dfg_overlap",
            [
                python,
                "attention/dfg_comp.py",
                "--graph_loc",
                str(graph_dir),
                "--code_file",
                str(code_file),
                "--save_dir",
                str(attention_dir),
                "--lang",
                "python",
                "--all_layers",
                "--bootstrap_samples",
                "1000",
            ],
        ),
        (
            "attention_summary",
            [
                python,
                "attention/summarize_attention_analysis.py",
                "--results_dir",
                str(attention_dir),
                "--output",
                str(attention_dir / "section_3_2_summary.json"),
                "--model",
                "codebert",
                "--lang",
                "python",
                "--threshold",
                "0.05",
                "--expected_programs",
                str(args.expected_programs),
                "--skip_ged",
            ],
        ),
        (
            "python_reference_comparison",
            [
                python,
                "attention/compare_python_reference.py",
                "--pilot_summary",
                str(attention_dir / "section_3_2_summary.json"),
                "--reference_root",
                args.python_reference_root,
                "--output",
                str(attention_dir / "python_reference_comparison.json"),
                "--skip_ged",
            ],
        ),
        (
            "representative_tsne",
            [
                python,
                "attention/hidden_tsne.py",
                "--embedding_dir",
                str(embedding_dir),
                "--save_dir",
                str(hidden_dir),
                "--lang",
                "python",
                "--layers",
                "5",
                "--token_perplexities",
                "50",
                "--distance_perplexities",
                "5",
                "10",
                "--min_tokens",
                "100",
                "--max_programs",
                "100",
                "--program_selection",
                "shortest",
                "--iterations",
                "50000",
                "--seed",
                "0",
            ],
        ),
    ]

    if complete_staged_representations(
        graph_dir,
        embedding_dir,
        args.expected_programs,
    ):
        phases = phases[4:]
        manifest["representations_reused"] = True
        manifest["resumed_utc"] = utc_now()
        atomic_json(manifest_path, manifest)
        print(
            "[RESUME] python/codebert: reusing the validated paired "
            f"{args.expected_programs}-artifact cohort",
            flush=True,
        )

    try:
        for phase, command in phases:
            duration = run_command(command, log_dir / f"{phase}.log", environment)
            manifest["phases"][phase] = {
                "status": "complete",
                "duration_seconds": duration,
                "finished_utc": utc_now(),
            }
            atomic_json(manifest_path, manifest)

        validation = validate_outputs(
            attention_dir, hidden_dir, args.expected_programs
        )
        if not validation["valid"]:
            raise ValueError("; ".join(validation["errors"]))
        atomic_json(validation_path, validation)

        graph_bytes, graph_count = purge_manifest_artifacts(
            graph_dir, "graph_manifest.json", attention_dir
        )
        embedding_bytes, embedding_count = purge_manifest_artifacts(
            embedding_dir, "embedding_manifest.json", hidden_dir
        )
        manifest.update(
            {
                "status": "complete",
                "finished_utc": utc_now(),
                "validation": str(validation_path.resolve()),
                "large_intermediates_retained": False,
                "purged_artifacts": graph_count + embedding_count,
                "purged_bytes": graph_bytes + embedding_bytes,
                "reextraction_required_for": ["GED", "DirectProbe"],
            }
        )
        manifest.pop("reason", None)
        manifest.pop("failed_utc", None)
        atomic_json(manifest_path, manifest)
    except Exception as exc:
        manifest.update(
            {
                "status": "failed",
                "failed_utc": utc_now(),
                "reason": f"{type(exc).__name__}: {exc}",
            }
        )
        atomic_json(manifest_path, manifest)
        raise

    print(
        "[PYTHON CODEBERT COMPLETE] purged "
        f"{manifest['purged_bytes'] / 1e9:.2f} GB after validation"
    )


if __name__ == "__main__":
    main()
