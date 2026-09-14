"""Run only the requested Python replication with bounded model parallelism."""

from __future__ import annotations

import concurrent.futures
import json
import os
import shutil
import sys
import time
from pathlib import Path

from run_staged_model_analysis import atomic_json, run_command, utc_now


REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = (
    REPO_ROOT
    / "analysis_results"
    / "background_runs"
    / "python_analysis_queue.json"
)
LOG_ROOT = (
    REPO_ROOT
    / "analysis_results"
    / "background_runs"
    / "python_analysis_logs"
)
ADAPTER_RESULT_ROOT = REPO_ROOT / "analysis_results" / "final_3000_multimodel"
EXPECTED_ADAPTERS = {
    "graphcodebert",
    "unixcoder",
    "codet5",
    "plbart",
    "codet5p_220",
}


def adapter_command(models: list[str], manifest_name: str) -> list[str]:
    return [
        sys.executable,
        "attention/run_staged_model_analysis.py",
        "--models",
        *models,
        "--languages",
        "python",
        "--dataset_root",
        "attention/exp_data",
        "--dataset_pattern",
        "exp_0.jsonl",
        "--graph_stage_root",
        "graph_info/staged_python_final_3000",
        "--embedding_stage_root",
        "structural_probe/staged_python_final_3000",
        "--results_root",
        "analysis_results/final_3000_multimodel",
        "--run_manifest",
        f"analysis_results/final_3000_multimodel/{manifest_name}",
        "--expected_programs",
        "3000",
        "--tsne_layer",
        "5",
        "--device",
        "cpu",
        "--minimum_free_gb",
        "55",
        "--omp_threads",
        "4",
        "--compare_python_reference",
    ]


def run_lane(name: str, command: list[str], environment: dict) -> dict:
    started = time.monotonic()
    free_before = shutil.disk_usage(REPO_ROOT).free
    duration = run_command(command, LOG_ROOT / f"{name}.log", environment)
    return {
        "status": "complete",
        "duration_seconds": duration,
        "free_bytes_before": free_before,
        "free_bytes_after": shutil.disk_usage(REPO_ROOT).free,
        "log": str((LOG_ROOT / f"{name}.log").resolve()),
        "finished_utc": utc_now(),
        "wall_seconds": time.monotonic() - started,
    }


def merge_adapter_manifests() -> Path:
    sources = [
        ADAPTER_RESULT_ROOT / "python_run_manifest_lane_a.json",
        ADAPTER_RESULT_ROOT / "python_run_manifest_lane_b.json",
    ]
    loaded = [json.loads(path.read_text()) for path in sources]
    errors = []
    runs = {}
    for path, manifest in zip(sources, loaded):
        if manifest.get("status") != "complete":
            errors.append(f"{path.name} is {manifest.get('status')}")
        for key, record in manifest.get("runs", {}).items():
            if key in runs:
                errors.append(f"duplicate adapter run {key}")
            runs[key] = record
    found = {key.rsplit("/", 1)[-1] for key in runs}
    if found != EXPECTED_ADAPTERS:
        errors.append(
            f"adapter set is {sorted(found)}, expected {sorted(EXPECTED_ADAPTERS)}"
        )
    incomplete = [key for key, value in runs.items() if value.get("status") != "complete"]
    if incomplete:
        errors.append(f"incomplete adapter runs: {sorted(incomplete)}")
    if errors:
        raise ValueError("; ".join(errors))

    merged = {
        "status": "complete",
        "purpose": "merged parallel Python adapter analysis",
        "created_utc": utc_now(),
        "models": sorted(EXPECTED_ADAPTERS),
        "languages": ["python"],
        "expected_programs": 3000,
        "excluded": ["CodeGen", "GED", "DirectProbe solver runs"],
        "python_reference_comparison": True,
        "source_manifests": [str(path.resolve()) for path in sources],
        "runs": runs,
        "num_complete": len(runs),
        "total_purged_bytes": sum(
            value.get("purged_bytes", 0) for value in runs.values()
        ),
        "finished_utc": utc_now(),
    }
    output = ADAPTER_RESULT_ROOT / "python_run_manifest.json"
    atomic_json(output, merged)
    return output


def main() -> None:
    try:
        os.nice(5)
    except OSError:
        pass
    free_gb = shutil.disk_usage(REPO_ROOT).free / 1e9
    if free_gb < 100:
        raise RuntimeError(
            f"Parallel Python queue needs at least 100 GB free; found {free_gb:.1f} GB"
        )
    if MANIFEST.exists():
        previous = json.loads(MANIFEST.read_text())
        previous_pid = previous.get("pid")
        if previous.get("status") == "running" and previous_pid:
            try:
                os.kill(int(previous_pid), 0)
            except (OSError, ValueError):
                pass
            else:
                raise RuntimeError(
                    f"Python analysis queue is already active as PID {previous_pid}"
                )

    lanes = {
        "codebert": [
            sys.executable,
            "attention/run_staged_codebert_analysis.py",
            "--device",
            "cpu",
            "--omp_threads",
            "4",
            "--minimum_free_gb",
            "55",
        ],
        # Historical 3,000-program averages balance these at ~69 and ~51 min.
        "adapter_lane_a": adapter_command(
            ["graphcodebert", "codet5", "plbart"],
            "python_run_manifest_lane_a.json",
        ),
        "adapter_lane_b": adapter_command(
            ["unixcoder", "codet5p_220"],
            "python_run_manifest_lane_b.json",
        ),
    }
    manifest = {
        "status": "running",
        "started_utc": utc_now(),
        "pid": os.getpid(),
        "scope": "Python AST/DFG overlap and representative t-SNE only",
        "excluded": ["GED", "Python DirectProbe"],
        "expected_models": [
            "codebert",
            "graphcodebert",
            "unixcoder",
            "codet5",
            "plbart",
            "codet5p_220",
        ],
        "resource_policy": {
            "concurrent_model_lanes": 3,
            "threads_per_lane": 4,
            "maximum_threads": 12,
            "minimum_starting_free_gb": 100,
            "minimum_per_model_free_gb": 55,
        },
        "lanes": {
            name: {"status": "running", "started_utc": utc_now()}
            for name in lanes
        },
        "postprocessing": {},
    }
    atomic_json(MANIFEST, manifest)

    environment = os.environ.copy()
    environment.update(
        {
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "4",
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "MPLBACKEND": "Agg",
            "MPLCONFIGDIR": "/tmp/sdt-parallel-python-matplotlib",
        }
    )
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(run_lane, name, command, environment.copy()): name
            for name, command in lanes.items()
        }
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            try:
                manifest["lanes"][name] = future.result()
            except Exception as exc:
                failure = {
                    "status": "failed",
                    "failed_utc": utc_now(),
                    "reason": f"{type(exc).__name__}: {exc}",
                }
                manifest["lanes"][name] = failure
                failures.append((name, failure))
            atomic_json(MANIFEST, manifest)

    if failures:
        manifest.update(
            {
                "status": "failed",
                "failed_utc": utc_now(),
                "failed_lanes": [name for name, _ in failures],
            }
        )
        atomic_json(MANIFEST, manifest)
        raise RuntimeError(f"Python analysis lanes failed: {failures}")

    merged = merge_adapter_manifests()
    manifest["postprocessing"]["merge_adapter_manifests"] = {
        "status": "complete",
        "output": str(merged.resolve()),
        "finished_utc": utc_now(),
    }
    atomic_json(MANIFEST, manifest)

    plotting_python = Path("/home/abhinav/miniconda3/envs/s4/bin/python")
    if not plotting_python.is_file():
        plotting_python = Path(sys.executable)
    post_commands = [
        (
            "generate_manuscript_outputs",
            [
                str(plotting_python),
                "analysis_results/manuscript/generate_manuscript_figures.py",
            ],
        ),
        (
            "build_notebook",
            [str(plotting_python), "analysis_results/build_multilingual_notebook.py"],
        ),
    ]
    if plotting_python.is_file():
        post_commands.append(
            (
                "execute_notebook",
                [
                    str(plotting_python),
                    "analysis_results/execute_notebook.py",
                    "analysis_results/multilingual_code_llm_results.ipynb",
                ],
            )
        )
    for name, command in post_commands:
        manifest["postprocessing"][name] = {"status": "running"}
        atomic_json(MANIFEST, manifest)
        try:
            manifest["postprocessing"][name] = run_lane(
                f"post_{name}", command, environment
            )
        except Exception as exc:
            manifest["postprocessing"][name] = {
                "status": "failed",
                "failed_utc": utc_now(),
                "reason": f"{type(exc).__name__}: {exc}",
            }
            manifest.update(
                {
                    "status": "failed",
                    "failed_utc": utc_now(),
                    "reason": f"Postprocessing {name} failed: {exc}",
                }
            )
            atomic_json(MANIFEST, manifest)
            raise
        atomic_json(MANIFEST, manifest)

    manifest.update({"status": "complete", "finished_utc": utc_now()})
    atomic_json(MANIFEST, manifest)
    print(f"[PARALLEL PYTHON ANALYSIS COMPLETE] {MANIFEST}")


if __name__ == "__main__":
    main()
