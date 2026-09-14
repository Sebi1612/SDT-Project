"""Run the requested multilingual legacy-GED matrix in resumable shards.

The expensive NetworkX calculation is isolated by layer so twelve layers can
run concurrently without racing on similarity.py's cohort-level output files.
Only one model/language artifact cohort is retained at a time.  Successful
layer shards are merged into the normal Section 3.2 directory and the large
graph artifacts are then removed using their manifest as the deletion scope.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODELS = ("codebert", "graphcodebert", "codet5")
DEFAULT_LANGUAGES = ("java", "go", "javascript")
LAYERS = tuple(range(12))
ARRAY_KEYS = (
    "ast",
    "dfg",
    "common",
    "ast_wo_identifiers",
    "ast_minus_ast_wo_identifiers",
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2)
    os.replace(temporary, path)


def load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def available_memory_gb() -> float:
    with open("/proc/meminfo") as handle:
        values = {
            line.split(":", 1)[0]: int(line.split()[1])
            for line in handle
            if ":" in line
        }
    return values["MemAvailable"] / 1024 / 1024


def result_dir(results_root: Path, language: str, model: str) -> Path:
    if model == "codebert":
        return results_root / "attention_final_3000" / language
    return results_root / "final_3000_multimodel" / language / model / "attention"


def graph_dir(graph_root: Path, language: str, model: str) -> Path:
    if model == "codebert":
        return graph_root / "codebert" / language / model
    return graph_root / "adapters" / language / model


def manifest_language(manifest: dict) -> str | None:
    return manifest.get("language", manifest.get("lang"))


def complete_graph_cohort(
    directory: Path, model: str, language: str, expected: int
) -> bool:
    try:
        manifest = load_json(directory / "graph_manifest.json")
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    artifacts = manifest.get("artifacts", [])
    return (
        manifest.get("status") == "complete"
        and manifest.get("model") == model
        and manifest_language(manifest) == language
        and manifest.get("selected_num_codes") == expected
        and manifest.get("num_saved") == expected
        and manifest.get("num_failures") == 0
        and not manifest.get("failures")
        and len(artifacts) == expected
        and all((directory / name).is_file() for name in artifacts)
    )


def purge_manifest_artifacts(directory: Path, manifest_name: str, reason: str) -> dict:
    manifest_path = directory / manifest_name
    manifest = load_json(manifest_path)
    resolved = []
    root = directory.resolve()
    for name in manifest.get("artifacts", []):
        path = (directory / name).resolve()
        if path.parent != root or path.suffix != ".pkl":
            raise ValueError(f"Unsafe manifest artifact path: {path}")
        resolved.append(path)
    removed = 0
    removed_bytes = 0
    for path in resolved:
        if path.is_file():
            removed_bytes += path.stat().st_size
            path.unlink()
            removed += 1
    manifest.update(
        {
            "status": "artifacts_purged_after_ged",
            "ged_cleanup_reason": reason,
            "ged_cleanup_utc": utc_now(),
            "ged_purged_artifacts": removed,
            "ged_purged_artifact_bytes": removed_bytes,
        }
    )
    atomic_json(manifest_path, manifest)
    return {"files": removed, "bytes": removed_bytes}


def run_logged(command: list[str], log_path: Path, environment: dict) -> float:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log_path.open("w") as log:
        log.write(f"started_utc={utc_now()}\n")
        log.write("command=" + json.dumps(command) + "\n")
        log.flush()
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        duration = time.monotonic() - started
        log.write(f"\nfinished_utc={utc_now()}\n")
        log.write(f"duration_seconds={duration}\n")
        log.write(f"exit_code={completed.returncode}\n")
    if completed.returncode:
        raise RuntimeError(
            f"Command failed with exit code {completed.returncode}: {log_path}"
        )
    return duration


def extract_graphs(
    args, language: str, model: str, environment: dict, cohort: dict
) -> None:
    directory = graph_dir(Path(args.graph_root), language, model)
    if complete_graph_cohort(directory, model, language, args.expected_programs):
        cohort["extraction"] = {"status": "reused", "finished_utc": utc_now()}
        return

    code_file = Path(args.dataset_root) / f"{language}.jsonl"
    log_path = Path(args.log_root) / language / model / "extract.log"
    if model == "codebert":
        language_root = directory.parent
        command = [
            args.python,
            "attention/save_graph_info.py",
            "--model", model,
            "--code_file", str(code_file),
            "--save_dir", str(language_root),
            "--lang", language,
            "--device", args.device,
            "--seed", "0",
        ]
        embedding_cleanup = None
    else:
        embedding_dir = Path(args.embedding_temp_root) / language / model
        command = [
            args.python,
            "attention/extract_model_representations.py",
            "--model", model,
            "--code_file", str(code_file),
            "--graph_output_dir", str(directory),
            "--embedding_output_dir", str(embedding_dir),
            "--lang", language,
            "--device", args.device,
            "--local_files_only",
            "--seed", "0",
        ]
        embedding_cleanup = embedding_dir

    cohort["extraction"] = {
        "status": "running",
        "started_utc": utc_now(),
        "log": str(log_path.resolve()),
    }
    duration = run_logged(command, log_path, environment)
    if not complete_graph_cohort(
        directory, model, language, args.expected_programs
    ):
        raise ValueError(f"Incomplete extracted graph cohort: {language}/{model}")
    cleanup = None
    if embedding_cleanup is not None:
        cleanup = purge_manifest_artifacts(
            embedding_cleanup,
            "embedding_manifest.json",
            "GED needs attention/graph artifacts only",
        )
    cohort["extraction"] = {
        "status": "complete",
        "finished_utc": utc_now(),
        "duration_seconds": duration,
        "log": str(log_path.resolve()),
        "temporary_embedding_cleanup": cleanup,
    }


def shard_paths(shard_root: Path, language: str, model: str, layer: int):
    root = shard_root / language / model / f"layer_{layer}"
    output = root / "similarity_legacy" / model
    return root, output


def valid_shard(
    shard_root: Path,
    language: str,
    model: str,
    layer: int,
    expected: int,
) -> bool:
    _, output = shard_paths(shard_root, language, model, layer)
    try:
        layer_data = load_json(output / f"layer_{layer}_threshold_0.05.json")
        manifest = load_json(output / "similarity_manifest.json")
        with np.load(output / "program_metrics_threshold_0.05.npz") as arrays:
            shape_ok = all(
                arrays[name].shape == (expected, 1, 12) for name in ARRAY_KEYS
            )
            layers_ok = arrays["layers"].tolist() == [layer]
    except (FileNotFoundError, json.JSONDecodeError, KeyError, ValueError):
        return False
    return (
        shape_ok
        and layers_ok
        and layer_data.get("model") == model
        and layer_data.get("language") == language
        and layer_data.get("layer") == layer
        and layer_data.get("distance_mode") == "legacy"
        and layer_data.get("num_evaluated") == expected
        and layer_data.get("num_attempted") == expected
        and manifest.get("status") == "complete"
        and manifest.get("layers") == [layer]
        and manifest.get("num_evaluated") == expected
        and manifest.get("num_failures") == 0
    )


def layer_command(args, language: str, model: str, layer: int) -> list[str]:
    directory = graph_dir(Path(args.graph_root), language, model)
    shard_dir, _ = shard_paths(Path(args.shard_root), language, model, layer)
    code_file = Path(args.dataset_root) / f"{language}.jsonl"
    return [
        args.python,
        "attention/similarity.py",
        "--graphs_dir", str(directory),
        "--code_file", str(code_file),
        "--save_dir", str(shard_dir),
        "--lang", language,
        "--layer", str(layer),
        "--threshold", "0.05",
        "--bootstrap_samples", "1000",
        "--distance_mode", "legacy",
    ]


def run_layer_shards(args, language: str, model: str, environment: dict, cohort: dict):
    shard_root = Path(args.shard_root)
    attempts = {str(layer): 0 for layer in LAYERS}
    pending = [
        layer for layer in LAYERS
        if not valid_shard(
            shard_root, language, model, layer, args.expected_programs
        )
    ]
    active: dict[int, dict] = {}
    cohort["layers"] = {
        str(layer): {
            "status": "complete" if layer not in pending else "pending",
            "attempts": 0,
        }
        for layer in LAYERS
    }

    while pending or active:
        while pending and len(active) < args.workers:
            layer = pending.pop(0)
            attempts[str(layer)] += 1
            attempt = attempts[str(layer)]
            log_path = (
                Path(args.log_root) / language / model
                / f"layer_{layer}_attempt_{attempt}.log"
            )
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log = log_path.open("w")
            command = layer_command(args, language, model, layer)
            log.write(f"started_utc={utc_now()}\n")
            log.write("command=" + json.dumps(command) + "\n")
            log.flush()
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            active[layer] = {
                "process": process,
                "log_handle": log,
                "log_path": log_path,
                "started_monotonic": time.monotonic(),
                "started_utc": utc_now(),
            }
            cohort["layers"][str(layer)] = {
                "status": "running",
                "attempts": attempt,
                "pid": process.pid,
                "started_utc": active[layer]["started_utc"],
                "log": str(log_path.resolve()),
            }

        cohort["active_layers"] = sorted(active)
        cohort["completed_layers"] = sorted(
            int(layer) for layer, state in cohort["layers"].items()
            if state["status"] == "complete"
        )
        yield "checkpoint"
        time.sleep(args.poll_seconds)

        for layer, state in list(active.items()):
            process = state["process"]
            duration = time.monotonic() - state["started_monotonic"]
            if process.poll() is None and duration > args.layer_timeout_hours * 3600:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            if process.poll() is None:
                continue

            state["log_handle"].write(
                f"\nfinished_utc={utc_now()}\n"
                f"duration_seconds={duration}\n"
                f"exit_code={process.returncode}\n"
            )
            state["log_handle"].close()
            success = process.returncode == 0 and valid_shard(
                shard_root, language, model, layer, args.expected_programs
            )
            del active[layer]
            if success:
                cohort["layers"][str(layer)] = {
                    "status": "complete",
                    "attempts": attempts[str(layer)],
                    "duration_seconds_last_attempt": duration,
                    "finished_utc": utc_now(),
                    "log": str(state["log_path"].resolve()),
                }
            elif attempts[str(layer)] < args.max_layer_attempts:
                cohort["layers"][str(layer)] = {
                    "status": "retry_pending",
                    "attempts": attempts[str(layer)],
                    "last_exit_code": process.returncode,
                    "log": str(state["log_path"].resolve()),
                }
                pending.append(layer)
            else:
                raise RuntimeError(
                    f"GED layer failed {attempts[str(layer)]} times: "
                    f"{language}/{model}/layer_{layer}; see {state['log_path']}"
                )


def merge_shards(args, language: str, model: str) -> Path:
    shard_root = Path(args.shard_root)
    canonical = result_dir(Path(args.results_root), language, model)
    output = canonical / "similarity_legacy" / model
    output.mkdir(parents=True, exist_ok=True)

    shard_outputs = [
        shard_paths(shard_root, language, model, layer)[1]
        for layer in LAYERS
    ]
    for layer in LAYERS:
        if not valid_shard(
            shard_root, language, model, layer, args.expected_programs
        ):
            raise ValueError(f"Cannot merge invalid layer {layer}")

    protocol = load_json(shard_outputs[0] / "evaluation_protocol.json")
    atomic_json(output / "evaluation_protocol.json", protocol)

    loaded = []
    try:
        for shard in shard_outputs:
            loaded.append(np.load(shard / "program_metrics_threshold_0.05.npz"))
        first = loaded[0]
        for arrays in loaded[1:]:
            for key in ("artifact", "sample_index", "source_index", "file_name"):
                if not np.array_equal(first[key], arrays[key]):
                    raise ValueError(f"Shard cohort metadata differs for {key}")
            if float(first["threshold"]) != float(arrays["threshold"]):
                raise ValueError("Shard thresholds differ")
        merged = {
            key: np.concatenate([arrays[key] for arrays in loaded], axis=1)
            for key in ARRAY_KEYS
        }
        for key in ("artifact", "sample_index", "source_index", "file_name"):
            merged[key] = first[key]
        merged["layers"] = np.asarray(LAYERS, dtype=np.int64)
        merged["threshold"] = first["threshold"]
        metrics_path = output / "program_metrics_threshold_0.05.npz"
        temporary = metrics_path.with_suffix(".npz.tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **merged)
        os.replace(temporary, metrics_path)
    finally:
        for arrays in loaded:
            arrays.close()

    manifest = load_json(shard_outputs[0] / "similarity_manifest.json")
    manifest.update(
        {
            "status": "complete",
            "layers": list(LAYERS),
            "program_metrics_file": str(metrics_path.resolve()),
            "evaluation_protocol": str((output / "evaluation_protocol.json").resolve()),
            "merged_from_parallel_layer_shards": [
                str(path.resolve()) for path in shard_outputs
            ],
            "merged_utc": utc_now(),
        }
    )
    manifest_path = output / "similarity_manifest.json"
    atomic_json(manifest_path, manifest)

    for layer, shard in zip(LAYERS, shard_outputs):
        data = load_json(shard / f"layer_{layer}_threshold_0.05.json")
        data.update(
            {
                "similarity_manifest": str(manifest_path.resolve()),
                "program_metrics_file": str(metrics_path.resolve()),
                "evaluation_protocol": str((output / "evaluation_protocol.json").resolve()),
            }
        )
        atomic_json(output / f"layer_{layer}_threshold_0.05.json", data)
    return output


def validate_canonical(args, language: str, model: str) -> dict:
    canonical = result_dir(Path(args.results_root), language, model)
    output = canonical / "similarity_legacy" / model
    errors = []
    try:
        manifest = load_json(output / "similarity_manifest.json")
        if manifest.get("status") != "complete" or manifest.get("layers") != list(LAYERS):
            errors.append("merged similarity manifest is incomplete")
        if manifest.get("num_evaluated") != args.expected_programs:
            errors.append("merged manifest does not have full program coverage")
        with np.load(output / "program_metrics_threshold_0.05.npz") as arrays:
            for key in ARRAY_KEYS:
                if arrays[key].shape != (args.expected_programs, 12, 12):
                    errors.append(f"wrong merged array shape for {key}")
            if arrays["layers"].tolist() != list(LAYERS):
                errors.append("wrong merged layer list")
        for layer in LAYERS:
            data = load_json(output / f"layer_{layer}_threshold_0.05.json")
            if (
                data.get("layer") != layer
                or data.get("distance_mode") != "legacy"
                or data.get("num_evaluated") != args.expected_programs
                or data.get("num_attempted") != args.expected_programs
            ):
                errors.append(f"invalid canonical layer {layer}")
    except (FileNotFoundError, json.JSONDecodeError, KeyError, ValueError) as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    return {
        "status": "complete" if not errors else "failed",
        "valid": not errors,
        "language": language,
        "model": model,
        "layers": list(LAYERS),
        "expected_programs": args.expected_programs,
        "distance_mode": "legacy",
        "results_directory": str(output.resolve()),
        "errors": errors,
        "validated_utc": utc_now(),
    }


def summarize(args, language: str, model: str, environment: dict) -> float:
    canonical = result_dir(Path(args.results_root), language, model)
    command = [
        args.python,
        "attention/summarize_attention_analysis.py",
        "--results_dir", str(canonical),
        "--output", str(canonical / "section_3_2_summary.json"),
        "--model", model,
        "--lang", language,
        "--threshold", "0.05",
        "--expected_programs", str(args.expected_programs),
        "--ged_mode", "legacy",
    ]
    return run_logged(
        command,
        Path(args.log_root) / language / model / "summarize.log",
        environment,
    )


def final_validation(args, language: str, model: str) -> dict:
    validation = validate_canonical(args, language, model)
    summary_path = result_dir(Path(args.results_root), language, model) / "section_3_2_summary.json"
    try:
        summary = load_json(summary_path)
        if (
            summary.get("status") != "complete"
            or summary.get("ged_status") != "included"
            or summary.get("ged_mode") != "legacy"
            or summary.get("expected_programs") != args.expected_programs
            or summary.get("num_layers") != 12
        ):
            validation["errors"].append("Section 3.2 summary GED metadata is invalid")
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        validation["errors"].append(f"summary {type(exc).__name__}: {exc}")
    validation["valid"] = not validation["errors"]
    validation["status"] = "complete" if validation["valid"] else "failed"
    validation["summary"] = str(summary_path.resolve())
    validation["validated_utc"] = utc_now()
    return validation


def parse_args():
    cli = argparse.ArgumentParser()
    cli.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    cli.add_argument("--languages", nargs="+", default=list(DEFAULT_LANGUAGES))
    cli.add_argument("--python", default=sys.executable)
    cli.add_argument("--device", default="cpu")
    cli.add_argument("--workers", type=int, default=12)
    cli.add_argument("--expected_programs", type=int, default=3000)
    cli.add_argument("--max_layer_attempts", type=int, default=3)
    cli.add_argument("--layer_timeout_hours", type=float, default=48.0)
    cli.add_argument("--poll_seconds", type=float, default=30.0)
    cli.add_argument("--minimum_memory_gb", type=float, default=80.0)
    cli.add_argument("--minimum_disk_gb", type=float, default=60.0)
    cli.add_argument("--dataset_root", default="attention/exp_data/final_3000")
    cli.add_argument("--graph_root", default="graph_info/ged_final_3000")
    cli.add_argument("--embedding_temp_root", default="structural_probe/ged_temp")
    cli.add_argument("--results_root", default="analysis_results")
    cli.add_argument("--shard_root", default="analysis_results/ged_final_3000/shards")
    cli.add_argument("--log_root", default="analysis_results/ged_final_3000/logs")
    cli.add_argument(
        "--manifest", default="analysis_results/ged_final_3000/ged_queue_manifest.json"
    )
    cli.add_argument("--dry_run", action="store_true")
    return cli.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1 or args.workers > 12:
        raise ValueError("--workers must be between 1 and 12")
    unknown_models = set(args.models) - set(DEFAULT_MODELS)
    unknown_languages = set(args.languages) - set(DEFAULT_LANGUAGES)
    if unknown_models or unknown_languages:
        raise ValueError(
            f"Unsupported requested scope: models={unknown_models}, languages={unknown_languages}"
        )
    for language in args.languages:
        code_file = Path(args.dataset_root) / f"{language}.jsonl"
        if not code_file.is_file():
            raise FileNotFoundError(code_file)
        with code_file.open() as handle:
            if sum(1 for _ in handle) != args.expected_programs:
                raise ValueError(f"Dataset does not contain 3000 lines: {code_file}")
    if not Path(args.python).is_file():
        raise FileNotFoundError(args.python)

    manifest_path = Path(args.manifest)
    manifest = {
        "status": "dry_run" if args.dry_run else "running",
        "started_utc": utc_now(),
        "pid": os.getpid(),
        "scope": {
            "models": args.models,
            "languages": args.languages,
            "layers": list(LAYERS),
            "total_layer_jobs": len(args.models) * len(args.languages) * len(LAYERS),
            "distance_mode": "legacy",
            "expected_programs": args.expected_programs,
        },
        "resource_policy": {
            "parallel_layers_per_cohort": args.workers,
            "sequential_cohorts": True,
            "device": args.device,
            "threads_per_layer": 1,
            "minimum_memory_gb_before_cohort": args.minimum_memory_gb,
            "minimum_disk_gb_before_cohort": args.minimum_disk_gb,
            "layer_timeout_hours": args.layer_timeout_hours,
            "max_layer_attempts": args.max_layer_attempts,
        },
        "cohort_order": [
            {"model": model, "language": language}
            for model in args.models for language in args.languages
        ],
        "cohorts": {},
    }
    atomic_json(manifest_path, manifest)
    if args.dry_run:
        print(f"[GED QUEUE DRY RUN COMPLETE] {manifest_path}")
        return

    try:
        os.nice(8)
    except OSError:
        pass
    environment = os.environ.copy()
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "MPLBACKEND": "Agg",
            "MPLCONFIGDIR": "/tmp/sdt-requested-ged-matplotlib",
        }
    )

    try:
        for model in args.models:
            for language in args.languages:
                key = f"{language}/{model}"
                validation_path = (
                    Path(args.results_root) / "ged_final_3000" / "validation"
                    / language / model / "ged_validation.json"
                )
                if validation_path.is_file() and load_json(validation_path).get("valid"):
                    manifest["cohorts"][key] = {
                        "status": "complete",
                        "resume_action": "validated_skip",
                        "validation": str(validation_path.resolve()),
                    }
                    atomic_json(manifest_path, manifest)
                    continue

                while (
                    available_memory_gb() < args.minimum_memory_gb
                    or shutil.disk_usage(REPO_ROOT).free / 1e9 < args.minimum_disk_gb
                ):
                    manifest["status"] = "waiting_for_capacity"
                    manifest["capacity"] = {
                        "checked_utc": utc_now(),
                        "available_memory_gb": available_memory_gb(),
                        "free_disk_gb": shutil.disk_usage(REPO_ROOT).free / 1e9,
                    }
                    atomic_json(manifest_path, manifest)
                    time.sleep(60)
                manifest["status"] = "running"
                cohort = {
                    "status": "extracting",
                    "model": model,
                    "language": language,
                    "started_utc": utc_now(),
                }
                manifest["cohorts"][key] = cohort
                manifest["active_cohort"] = key
                atomic_json(manifest_path, manifest)

                extract_graphs(args, language, model, environment, cohort)
                cohort["status"] = "running_layers"
                atomic_json(manifest_path, manifest)
                for _ in run_layer_shards(args, language, model, environment, cohort):
                    atomic_json(manifest_path, manifest)

                cohort["status"] = "merging"
                atomic_json(manifest_path, manifest)
                merge_shards(args, language, model)
                canonical_validation = validate_canonical(args, language, model)
                if not canonical_validation["valid"]:
                    raise ValueError("; ".join(canonical_validation["errors"]))

                cohort["status"] = "summarizing"
                atomic_json(manifest_path, manifest)
                cohort["summary_duration_seconds"] = summarize(
                    args, language, model, environment
                )
                validation = final_validation(args, language, model)
                atomic_json(validation_path, validation)
                if not validation["valid"]:
                    raise ValueError("; ".join(validation["errors"]))

                directory = graph_dir(Path(args.graph_root), language, model)
                cleanup = purge_manifest_artifacts(
                    directory,
                    "graph_manifest.json",
                    "all 12 legacy GED layers merged, summarized, and validated",
                )
                cohort.update(
                    {
                        "status": "complete",
                        "finished_utc": utc_now(),
                        "validation": str(validation_path.resolve()),
                        "graph_cleanup": cleanup,
                        "active_layers": [],
                        "completed_layers": list(LAYERS),
                    }
                )
                atomic_json(manifest_path, manifest)

        manifest.update(
            {
                "status": "complete",
                "finished_utc": utc_now(),
                "active_cohort": None,
                "completed_cohorts": len(args.models) * len(args.languages),
                "completed_layer_jobs": len(args.models) * len(args.languages) * 12,
            }
        )
        atomic_json(manifest_path, manifest)
        print(f"[GED QUEUE COMPLETE] {manifest_path}")
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


if __name__ == "__main__":
    main()
