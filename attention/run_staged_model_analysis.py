"""Run storage-bounded final analyses for the non-CodeBERT model adapters.

Only one model/language artifact cohort is retained at a time.  After paired
artifacts, attention metrics, and representative t-SNE outputs pass strict
validation, the large pickle files are purged while their manifests and all
permanent analysis outputs remain.  A later GED or DirectProbe solver run must
therefore re-extract the corresponding cohort.
"""

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from model_adapters import get_model_spec


DEFAULT_MODELS = [
    "graphcodebert",
    "unixcoder",
    "codet5",
    "plbart",
    "codet5p_220",
]
DEFAULT_LANGUAGES = ["java", "go", "javascript"]


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2)
    os.replace(temporary, path)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_command(command, log_path, environment):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    print(f"[START] {' '.join(command)}", flush=True)
    with log_path.open("w") as log:
        log.write(f"started_utc={utc_now()}\n")
        log.write("command=" + json.dumps(command) + "\n")
        log.flush()
        process = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parent.parent,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        duration = time.monotonic() - started
        log.write(f"\nfinished_utc={utc_now()}\n")
        log.write(f"duration_seconds={duration}\n")
        log.write(f"exit_code={process.returncode}\n")
    if process.returncode:
        tail = log_path.read_text(errors="replace")[-4000:]
        raise RuntimeError(
            f"Command failed with exit code {process.returncode}; "
            f"see {log_path}\n{tail}"
        )
    print(f"[DONE] {log_path.name} ({duration:.1f}s)", flush=True)
    return duration


def load_json(path):
    with open(path) as handle:
        return json.load(handle)


def validate_complete_extraction(graph_dir, embedding_dir, model, language,
                                 expected_programs):
    """Require a lossless paired cohort before starting any analysis."""
    graph_dir = Path(graph_dir)
    embedding_dir = Path(embedding_dir)
    graph_manifest = load_json(graph_dir / "graph_manifest.json")
    embedding_manifest = load_json(
        embedding_dir / "embedding_manifest.json"
    )
    errors = []
    for label, manifest, directory in (
        ("graph", graph_manifest, graph_dir),
        ("embedding", embedding_manifest, embedding_dir),
    ):
        artifacts = manifest.get("artifacts", [])
        failures = manifest.get("failures", [])
        if manifest.get("status") != "complete":
            errors.append(f"{label} extraction is not complete")
        if manifest.get("model") != model:
            errors.append(f"{label} model mismatch")
        if manifest.get("language") != language:
            errors.append(f"{label} language mismatch")
        if manifest.get("selected_num_codes") != expected_programs:
            errors.append(f"{label} selected count is not {expected_programs}")
        if manifest.get("num_saved") != expected_programs:
            errors.append(f"{label} saved count is not {expected_programs}")
        if len(artifacts) != expected_programs:
            errors.append(f"{label} artifact list is not {expected_programs}")
        if manifest.get("num_failures") != 0 or failures:
            errors.append(f"{label} extraction contains failures")
        missing = [name for name in artifacts if not (directory / name).is_file()]
        if missing:
            errors.append(
                f"{label} has {len(missing)} missing manifest-listed artifacts"
            )
    if graph_manifest.get("artifacts") != embedding_manifest.get("artifacts"):
        errors.append("graph and embedding cohorts differ")

    report = {
        "status": "complete" if not errors else "failed",
        "model": model,
        "language": language,
        "expected_programs": expected_programs,
        "graph_artifacts": len(graph_manifest.get("artifacts", [])),
        "embedding_artifacts": len(embedding_manifest.get("artifacts", [])),
        "graph_failures": len(graph_manifest.get("failures", [])),
        "embedding_failures": len(embedding_manifest.get("failures", [])),
        "paired_cohort": (
            graph_manifest.get("artifacts")
            == embedding_manifest.get("artifacts")
        ),
        "errors": errors,
        "valid": not errors,
    }
    if errors:
        raise ValueError("; ".join(errors))
    return report


def validate_permanent_outputs(
    results_dir, model, language, expected_layers, expected_programs,
    tsne_layer
):
    results_dir = Path(results_dir)
    attention_dir = results_dir / "attention"
    summary_path = attention_dir / "section_3_2_summary.json"
    token_manifest_path = (
        results_dir / "hidden_tsne" / "token_types"
        / "token_type_tsne_manifest.json"
    )
    distance_manifest_path = (
        results_dir / "hidden_tsne" / "distances"
        / "distance_tsne_manifest.json"
    )
    summary = load_json(summary_path)
    errors = []
    if summary.get("status") != "complete":
        errors.append("attention summary is not complete")
    if summary.get("model") != model or summary.get("language") != language:
        errors.append("attention summary model/language mismatch")
    if summary.get("num_layers") != expected_layers:
        errors.append("attention summary layer count mismatch")
    if summary.get("expected_programs") != expected_programs:
        errors.append("attention summary expected-program count mismatch")
    if summary.get("ged_status") != "skipped":
        errors.append("GED was not explicitly skipped")

    for layer in range(expected_layers):
        ast_path = attention_dir / "ast" / f"{model}_layer_{layer}.json"
        dfg_path = attention_dir / "dfg" / f"{model}_layer_{layer}.json"
        ast = load_json(ast_path)
        dfg = load_json(dfg_path)
        if ast.get("num_evaluated") != expected_programs:
            errors.append(f"AST layer {layer} lacks full coverage")
        if dfg.get("num_aligned") != expected_programs:
            errors.append(f"DFG layer {layer} lacks full alignment")
        if dfg.get("end_to_end_alignment_rate") != 1.0:
            errors.append(f"DFG layer {layer} lacks end-to-end coverage")

    token_manifest = load_json(token_manifest_path)
    distance_manifest = load_json(distance_manifest_path)
    expected_tsne = (
        (token_manifest_path, token_manifest, [50]),
        (distance_manifest_path, distance_manifest, [5, 10]),
    )
    generated = []
    for manifest_path, manifest, perplexities in expected_tsne:
        if manifest.get("model") != model or manifest.get("language") != language:
            errors.append(f"t-SNE manifest mismatch: {manifest_path}")
        if manifest.get("layers") != [tsne_layer]:
            errors.append(f"t-SNE layer mismatch: {manifest_path}")
        if manifest.get("perplexities") != perplexities:
            errors.append(f"t-SNE perplexity mismatch: {manifest_path}")
        if manifest.get("iterations") != 50000:
            errors.append(f"t-SNE iteration mismatch: {manifest_path}")
        for name in manifest.get("generated_files", []):
            path = manifest_path.parent / name
            generated.append(str(path.resolve()))
            if not path.is_file() or path.stat().st_size == 0:
                errors.append(f"missing t-SNE output: {path}")

    report = {
        "status": "complete" if not errors else "failed",
        "model": model,
        "language": language,
        "expected_programs": expected_programs,
        "expected_attention_layers": expected_layers,
        "representative_tsne_layer": tsne_layer,
        "ged": "skipped",
        "directprobe_solver": "skipped",
        "summary": str(summary_path.resolve()),
        "token_tsne_manifest": str(token_manifest_path.resolve()),
        "distance_tsne_manifest": str(distance_manifest_path.resolve()),
        "generated_tsne_files": generated,
        "errors": errors,
        "valid": not errors,
    }
    if errors:
        raise ValueError("; ".join(errors))
    return report


def purge_manifest_artifacts(directory, manifest_name, permanent_results):
    directory = Path(directory).resolve()
    manifest_path = directory / manifest_name
    manifest = load_json(manifest_path)
    if manifest.get("status") != "complete":
        raise ValueError(f"Refusing to purge incomplete manifest: {manifest_path}")
    artifacts = manifest.get("artifacts", [])
    if not artifacts:
        raise ValueError(f"Refusing to purge an empty cohort: {manifest_path}")

    resolved = []
    for name in artifacts:
        if Path(name).name != name or not name.endswith(".pkl"):
            raise ValueError(f"Unsafe artifact name in manifest: {name!r}")
        path = (directory / name).resolve()
        if path.parent != directory or not path.is_file():
            raise ValueError(f"Missing or unsafe staged artifact: {path}")
        resolved.append(path)
    bytes_purged = sum(path.stat().st_size for path in resolved)
    manifest_sha256_before_purge = sha256_file(manifest_path)
    for path in resolved:
        path.unlink()

    manifest.update({
        "extraction_status_before_purge": "complete",
        "status": "artifacts_purged_after_analysis",
        "artifacts_retained": False,
        "purged_artifact_count": len(resolved),
        "purged_artifact_bytes": bytes_purged,
        "purged_utc": utc_now(),
        "manifest_sha256_before_purge": manifest_sha256_before_purge,
        "permanent_results": str(Path(permanent_results).resolve()),
        "reextraction_required_for": ["GED", "DirectProbe"],
    })
    atomic_json(manifest_path, manifest)
    return bytes_purged, len(resolved)


def run_one(args, language, model, run_manifest, environment):
    key = f"{language}/{model}"
    record = run_manifest["runs"].setdefault(key, {})
    if record.get("status") == "complete":
        print(f"[SKIP] {key} already complete", flush=True)
        return

    free_gb = shutil.disk_usage(Path.cwd()).free / 1e9
    if free_gb < args.minimum_free_gb:
        raise RuntimeError(
            f"Refusing to stage {key}: only {free_gb:.1f} GB free; "
            f"require {args.minimum_free_gb:.1f} GB"
        )

    spec = get_model_spec(model)
    code_file = Path(args.dataset_root) / f"{language}.jsonl"
    graph_dir = Path(args.graph_stage_root) / language / model
    embedding_dir = Path(args.embedding_stage_root) / language / model
    results_dir = Path(args.results_root) / language / model
    log_dir = results_dir / "logs"
    attention_dir = results_dir / "attention"
    tsne_dir = results_dir / "hidden_tsne"
    record.update({
        "status": "in_progress",
        "model": model,
        "language": language,
        "started_utc": record.get("started_utc", utc_now()),
        "code_file": str(code_file.resolve()),
        "graph_stage_dir": str(graph_dir.resolve()),
        "embedding_stage_dir": str(embedding_dir.resolve()),
        "results_dir": str(results_dir.resolve()),
        "phases": record.get("phases", {}),
    })
    atomic_json(args.run_manifest, run_manifest)

    python = sys.executable
    phases = [
        (
            "extract",
            [
                python,
                "attention/extract_model_representations.py",
                "--model", model,
                "--code_file", str(code_file),
                "--graph_output_dir", str(graph_dir),
                "--embedding_output_dir", str(embedding_dir),
                "--lang", language,
                "--device", args.device,
                "--local_files_only",
            ],
        ),
        (
            "validate_representations",
            [
                python,
                "attention/validate_representation_run.py",
                "--graph_dir", str(graph_dir),
                "--embedding_dir", str(embedding_dir),
                "--expected_model", model,
                "--expected_language", language,
                "--expected_count", str(args.expected_programs),
            ],
        ),
        (
            "ast_overlap",
            [
                python,
                "attention/graph_comp.py",
                "--graph_loc", str(graph_dir),
                "--save_dir", str(attention_dir),
                "--all_layers",
            ],
        ),
        (
            "dfg_overlap",
            [
                python,
                "attention/dfg_comp.py",
                "--graph_loc", str(graph_dir),
                "--code_file", str(code_file),
                "--save_dir", str(attention_dir),
                "--lang", language,
                "--all_layers",
            ],
        ),
        (
            "attention_summary",
            [
                python,
                "attention/summarize_attention_analysis.py",
                "--results_dir", str(attention_dir),
                "--output", str(attention_dir / "section_3_2_summary.json"),
                "--model", model,
                "--lang", language,
                "--expected_programs", str(args.expected_programs),
                "--skip_ged",
            ],
        ),
        (
            "representative_tsne",
            [
                python,
                "attention/hidden_tsne.py",
                "--embedding_dir", str(embedding_dir),
                "--save_dir", str(tsne_dir),
                "--lang", language,
                "--layers", str(args.tsne_layer),
                "--token_perplexities", "50",
                "--distance_perplexities", "5", "10",
                "--min_tokens", "100",
                "--max_programs", "100",
                "--program_selection", "shortest",
                "--iterations", "50000",
                "--seed", "0",
            ],
        ),
    ]

    try:
        for phase, command in phases:
            duration = run_command(command, log_dir / f"{phase}.log", environment)
            record["phases"][phase] = {
                "status": "complete",
                "duration_seconds": duration,
                "finished_utc": utc_now(),
            }
            atomic_json(args.run_manifest, run_manifest)
            if phase == "validate_representations":
                extraction_validation = validate_complete_extraction(
                    graph_dir,
                    embedding_dir,
                    model,
                    language,
                    args.expected_programs,
                )
                atomic_json(
                    results_dir / "extraction_validation.json",
                    extraction_validation,
                )

        validation = validate_permanent_outputs(
            results_dir,
            model,
            language,
            spec.expected_transformer_layers,
            args.expected_programs,
            args.tsne_layer,
        )
        atomic_json(results_dir / "final_validation.json", validation)
        graph_bytes, graph_count = purge_manifest_artifacts(
            graph_dir, "graph_manifest.json", results_dir
        )
        embedding_bytes, embedding_count = purge_manifest_artifacts(
            embedding_dir, "embedding_manifest.json", results_dir
        )
        record.update({
            "status": "complete",
            "finished_utc": utc_now(),
            "validation": str((results_dir / "final_validation.json").resolve()),
            "large_intermediates_retained": False,
            "purged_artifacts": graph_count + embedding_count,
            "purged_bytes": graph_bytes + embedding_bytes,
            "reextraction_required_for": ["GED", "DirectProbe"],
        })
        atomic_json(args.run_manifest, run_manifest)
        print(
            f"[STAGE COMPLETE] {key}; purged "
            f"{(graph_bytes + embedding_bytes) / 1e9:.2f} GB",
            flush=True,
        )
    except Exception as exc:
        record.update({
            "status": "failed",
            "failed_utc": utc_now(),
            "reason": f"{type(exc).__name__}: {exc}",
        })
        atomic_json(args.run_manifest, run_manifest)
        raise


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    cli.add_argument("--languages", nargs="+", default=DEFAULT_LANGUAGES)
    cli.add_argument("--dataset_root", default="attention/exp_data/final_3000")
    cli.add_argument(
        "--graph_stage_root", default="graph_info/staged_final_3000"
    )
    cli.add_argument(
        "--embedding_stage_root", default="structural_probe/staged_final_3000"
    )
    cli.add_argument(
        "--results_root", default="analysis_results/final_3000_multimodel"
    )
    cli.add_argument(
        "--run_manifest",
        default="analysis_results/final_3000_multimodel/run_manifest.json",
    )
    cli.add_argument("--expected_programs", type=int, default=3000)
    cli.add_argument("--tsne_layer", type=int, default=5)
    cli.add_argument("--device", default="cpu")
    cli.add_argument("--minimum_free_gb", type=float, default=35.0)
    cli.add_argument("--omp_threads", type=int, default=4)
    args = cli.parse_args()

    args.run_manifest = str(Path(args.run_manifest))
    free_gb = shutil.disk_usage(Path.cwd()).free / 1e9
    if free_gb < args.minimum_free_gb:
        raise RuntimeError(
            f"Only {free_gb:.1f} GB free; require {args.minimum_free_gb:.1f} GB"
        )
    if Path(args.run_manifest).exists():
        run_manifest = load_json(args.run_manifest)
    else:
        run_manifest = {
            "status": "in_progress",
            "purpose": "storage-bounded non-GED/non-DirectProbe final analysis",
            "created_utc": utc_now(),
            "python": sys.executable,
            "models": args.models,
            "languages": args.languages,
            "expected_programs": args.expected_programs,
            "representative_tsne": {
                "layer": args.tsne_layer,
                "token_perplexities": [50],
                "distance_perplexities": [5, 10],
                "iterations": 50000,
            },
            "excluded": ["CodeGen", "GED", "DirectProbe solver runs"],
            "runs": {},
        }
    atomic_json(args.run_manifest, run_manifest)

    environment = os.environ.copy()
    environment.update({
        "OMP_NUM_THREADS": str(args.omp_threads),
        "MKL_NUM_THREADS": str(args.omp_threads),
        "TOKENIZERS_PARALLELISM": "false",
        "MPLCONFIGDIR": "/tmp/model_adapter_verification/matplotlib",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    })
    for language in args.languages:
        for model in args.models:
            run_one(args, language, model, run_manifest, environment)

    run_manifest["status"] = "complete"
    run_manifest["finished_utc"] = utc_now()
    run_manifest["num_complete"] = sum(
        record.get("status") == "complete"
        for record in run_manifest["runs"].values()
    )
    run_manifest["total_purged_bytes"] = sum(
        record.get("purged_bytes", 0)
        for record in run_manifest["runs"].values()
    )
    phase_durations = {}
    for record in run_manifest["runs"].values():
        for phase, phase_record in record.get("phases", {}).items():
            phase_durations[phase] = phase_durations.get(phase, 0.0) + (
                phase_record.get("duration_seconds", 0.0)
            )
    run_manifest["phase_duration_seconds"] = phase_durations
    run_manifest["total_phase_duration_seconds"] = sum(
        phase_durations.values()
    )
    atomic_json(args.run_manifest, run_manifest)
    print(
        f"[ALL STAGED ANALYSES COMPLETE] {run_manifest['num_complete']} runs",
        flush=True,
    )


if __name__ == "__main__":
    main()
