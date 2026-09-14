"""Prepare retained-model DirectProbe datasets with bounded disk usage.

Each model/language representation cohort is extracted, validated, converted
into the five paper-style DirectProbe datasets at layers 5, 9, and 12, and
then purged only after every generated dataset/configuration passes checks.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path

from run_staged_model_analysis import (
    atomic_json,
    purge_manifest_artifacts,
    run_command,
    utc_now,
    validate_complete_extraction,
)


TASK_SETTINGS = {
    "distance": (1300, 160),
    "distance_id": (1300, 450),
    "siblings": (1500, 100),
    "siblings_id": (1500, 300),
    "dfg": (1500, 130),
}
LAYERS = [5, 9, 12]


def program_cap_attempts(initial: int, maximum: int) -> list[int]:
    """Return bounded retry caps while keeping the per-label target fixed."""
    candidates = [
        initial,
        math.ceil(initial * 1.5),
        initial * 2,
        initial * 3,
        initial * 5,
        initial * 8,
        1200,
        2000,
        maximum,
    ]
    return sorted({min(value, maximum) for value in candidates if value >= initial})


def load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def validate_task_dataset(
    output_root: Path, language: str, model: str, task: str
) -> dict:
    data_dir = output_root / "data" / language / task / model
    config_dir = output_root / "config_files" / language / task
    manifest_path = data_dir / "dataset_manifest.json"
    manifest = load_json(manifest_path)
    errors = []
    expected_target = TASK_SETTINGS[task][0]
    if manifest.get("status") != "complete":
        errors.append("dataset manifest is not complete")
    if manifest.get("language") != language:
        errors.append("language mismatch")
    if manifest.get("model") != model:
        errors.append("model mismatch")
    if manifest.get("task") != task:
        errors.append("task mismatch")
    if manifest.get("layers") != LAYERS:
        errors.append("hidden-layer selection mismatch")
    if manifest.get("selected_per_label") != expected_target:
        errors.append("target-per-label was not met")
    if manifest.get("num_program_failures") != 0:
        errors.append("program failures were recorded")
    if manifest.get("num_train", 0) + manifest.get("num_test", 0) != manifest.get(
        "num_examples"
    ):
        errors.append("train/test example accounting mismatch")

    required = [
        data_dir / "labels" / "tags.txt",
        data_dir / "entities" / "train.txt",
        data_dir / "entities" / "test.txt",
    ]
    for layer in LAYERS:
        required.extend(
            [
                data_dir / "embeddings" / "layers" / "train" / f"{layer}.txt",
                data_dir / "embeddings" / "layers" / "test" / f"{layer}.txt",
                config_dir / f"config_{model}_{layer}.ini",
            ]
        )
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        errors.append(f"{len(missing)} required files are missing or empty")
    return {
        "status": "complete" if not errors else "failed",
        "language": language,
        "model": model,
        "task": task,
        "manifest": str(manifest_path.resolve()),
        "num_examples": manifest.get("num_examples"),
        "num_train": manifest.get("num_train"),
        "num_test": manifest.get("num_test"),
        "layers": manifest.get("layers"),
        "errors": errors,
        "valid": not errors,
    }


def main() -> None:
    cli = argparse.ArgumentParser()
    cli.add_argument(
        "--models", nargs="+", default=["graphcodebert", "codet5"]
    )
    cli.add_argument(
        "--languages", nargs="+", default=["java", "go", "javascript"]
    )
    cli.add_argument("--dataset_root", default="attention/exp_data/final_3000")
    cli.add_argument(
        "--graph_stage_root", default="graph_info/directprobe_stage_final_3000"
    )
    cli.add_argument(
        "--embedding_stage_root",
        default="structural_probe/directprobe_stage_final_3000",
    )
    cli.add_argument("--output_root", default="DirectProbe/final_3000")
    cli.add_argument(
        "--run_manifest",
        default="DirectProbe/final_3000/directprobe_preparation_manifest.json",
    )
    cli.add_argument("--expected_programs", type=int, default=3000)
    cli.add_argument("--device", default="cpu")
    cli.add_argument("--minimum_free_gb", type=float, default=55.0)
    cli.add_argument("--omp_threads", type=int, default=4)
    args = cli.parse_args()

    output_root = Path(args.output_root)
    manifest_path = Path(args.run_manifest)
    if manifest_path.exists():
        manifest = load_json(manifest_path)
    else:
        manifest = {
            "status": "in_progress",
            "purpose": "storage-bounded DirectProbe dataset preparation",
            "created_utc": utc_now(),
            "models": args.models,
            "languages": args.languages,
            "tasks": list(TASK_SETTINGS),
            "layers": LAYERS,
            "expected_programs": args.expected_programs,
            "runs": {},
        }
    manifest["status"] = "in_progress"
    atomic_json(manifest_path, manifest)

    environment = os.environ.copy()
    environment.update(
        {
            "OMP_NUM_THREADS": str(args.omp_threads),
            "MKL_NUM_THREADS": str(args.omp_threads),
            "OPENBLAS_NUM_THREADS": str(args.omp_threads),
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    python = sys.executable

    try:
        for language in args.languages:
            for model in args.models:
                key = f"{language}/{model}"
                record = manifest["runs"].setdefault(key, {})
                if record.get("status") == "complete":
                    print(f"[SKIP] {key} probing data already complete", flush=True)
                    continue
                free_gb = shutil.disk_usage(Path.cwd()).free / 1e9
                if free_gb < args.minimum_free_gb:
                    raise RuntimeError(
                        f"Refusing to stage {key}: {free_gb:.1f} GB free; "
                        f"require {args.minimum_free_gb:.1f} GB"
                    )

                code_file = Path(args.dataset_root) / f"{language}.jsonl"
                graph_dir = Path(args.graph_stage_root) / language / model
                embedding_dir = Path(args.embedding_stage_root) / language / model
                log_dir = output_root / "preparation_logs" / language / model
                record.update(
                    {
                        "status": "in_progress",
                        "started_utc": record.get("started_utc", utc_now()),
                        "language": language,
                        "model": model,
                        "code_file": str(code_file.resolve()),
                        "graph_stage_dir": str(graph_dir.resolve()),
                        "embedding_stage_dir": str(embedding_dir.resolve()),
                        "phases": record.get("phases", {}),
                        "tasks": record.get("tasks", {}),
                    }
                )
                atomic_json(manifest_path, manifest)

                try:
                    validate_complete_extraction(
                        graph_dir,
                        embedding_dir,
                        model,
                        language,
                        args.expected_programs,
                    )
                    representations_valid = True
                except (FileNotFoundError, KeyError, TypeError, ValueError):
                    representations_valid = False

                if representations_valid:
                    print(
                        f"[RESUME] {key} has a validated representation cohort",
                        flush=True,
                    )
                else:
                    extract_command = [
                        python,
                        "attention/extract_model_representations.py",
                        "--model",
                        model,
                        "--code_file",
                        str(code_file),
                        "--graph_output_dir",
                        str(graph_dir),
                        "--embedding_output_dir",
                        str(embedding_dir),
                        "--lang",
                        language,
                        "--device",
                        args.device,
                        "--local_files_only",
                    ]
                    duration = run_command(
                        extract_command, log_dir / "extract.log", environment
                    )
                    record["phases"]["extract"] = {
                        "status": "complete",
                        "duration_seconds": duration,
                        "finished_utc": utc_now(),
                    }
                    atomic_json(manifest_path, manifest)

                    validate_command = [
                        python,
                        "attention/validate_representation_run.py",
                        "--graph_dir",
                        str(graph_dir),
                        "--embedding_dir",
                        str(embedding_dir),
                        "--expected_model",
                        model,
                        "--expected_language",
                        language,
                        "--expected_count",
                        str(args.expected_programs),
                    ]
                    duration = run_command(
                        validate_command,
                        log_dir / "validate_representations.log",
                        environment,
                    )
                    validate_complete_extraction(
                        graph_dir,
                        embedding_dir,
                        model,
                        language,
                        args.expected_programs,
                    )
                    record["phases"]["validate_representations"] = {
                        "status": "complete",
                        "duration_seconds": duration,
                        "finished_utc": utc_now(),
                    }
                    atomic_json(manifest_path, manifest)

                for task, (target, max_programs) in TASK_SETTINGS.items():
                    existing_manifest = (
                        output_root
                        / "data"
                        / language
                        / task
                        / model
                        / "dataset_manifest.json"
                    )
                    if existing_manifest.is_file():
                        task_validation = validate_task_dataset(
                            output_root, language, model, task
                        )
                        if task_validation["valid"]:
                            record["tasks"][task] = task_validation
                            atomic_json(manifest_path, manifest)
                            print(
                                f"[RESUME] {key}/{task} dataset is valid",
                                flush=True,
                            )
                            continue

                    duration = None
                    used_program_cap = None
                    attempted_caps = []
                    for program_cap in program_cap_attempts(
                        max_programs, args.expected_programs
                    ):
                        attempted_caps.append(program_cap)
                        command = [
                            python,
                            "attention/create_multilingual_dp.py",
                            "--task",
                            task,
                            "--lang",
                            language,
                            "--model",
                            model,
                            "--embedding_dir",
                            str(embedding_dir),
                            "--graph_dir",
                            str(graph_dir),
                            "--output_root",
                            str(output_root),
                            "--layers",
                            *[str(layer) for layer in LAYERS],
                            "--target_per_label",
                            str(target),
                            "--max_programs",
                            str(program_cap),
                            "--require_target",
                            "--seed",
                            "0",
                        ]
                        attempt_log = log_dir / f"create_{task}_{program_cap}.log"
                        try:
                            duration = run_command(command, attempt_log, environment)
                            used_program_cap = program_cap
                            break
                        except RuntimeError:
                            log_text = attempt_log.read_text(errors="replace")
                            if "was not met with" not in log_text:
                                raise
                            print(
                                f"[EXPAND] {key}/{task}: target not met with "
                                f"{program_cap} programs",
                                flush=True,
                            )
                    if used_program_cap is None or duration is None:
                        raise ValueError(
                            f"{key}/{task}: target {target}/label was not met "
                            f"after caps {attempted_caps}"
                        )
                    task_validation = validate_task_dataset(
                        output_root, language, model, task
                    )
                    if not task_validation["valid"]:
                        raise ValueError(
                            f"{key}/{task}: " + "; ".join(task_validation["errors"])
                        )
                    task_validation["duration_seconds"] = duration
                    task_validation["paper_program_cap"] = max_programs
                    task_validation["used_program_cap"] = used_program_cap
                    task_validation["attempted_program_caps"] = attempted_caps
                    record["tasks"][task] = task_validation
                    atomic_json(manifest_path, manifest)

                graph_bytes, graph_count = purge_manifest_artifacts(
                    graph_dir, "graph_manifest.json", output_root
                )
                embedding_bytes, embedding_count = purge_manifest_artifacts(
                    embedding_dir, "embedding_manifest.json", output_root
                )
                record.update(
                    {
                        "status": "complete",
                        "finished_utc": utc_now(),
                        "large_intermediates_retained": False,
                        "purged_artifacts": graph_count + embedding_count,
                        "purged_bytes": graph_bytes + embedding_bytes,
                        "permanent_directprobe_data": str(output_root.resolve()),
                        "reextraction_required_for": ["GED", "new probing layers"],
                    }
                )
                atomic_json(manifest_path, manifest)
                print(
                    f"[PROBING DATA COMPLETE] {key}; purged "
                    f"{(graph_bytes + embedding_bytes) / 1e9:.2f} GB",
                    flush=True,
                )
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

    manifest.update(
        {
            "status": "complete",
            "finished_utc": utc_now(),
            "num_complete": sum(
                run.get("status") == "complete" for run in manifest["runs"].values()
            ),
            "total_purged_bytes": sum(
                run.get("purged_bytes", 0) for run in manifest["runs"].values()
            ),
        }
    )
    atomic_json(manifest_path, manifest)
    print(f"[ALL PROBING DATA COMPLETE] {manifest['num_complete']} runs")


if __name__ == "__main__":
    main()
