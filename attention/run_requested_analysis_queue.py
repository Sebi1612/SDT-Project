"""Run the requested Python replication and probing-data preparation queue."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

from run_staged_model_analysis import atomic_json, run_command, utc_now


REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = REPO_ROOT / "analysis_results" / "background_runs" / "analysis_queue.json"
LOG_ROOT = REPO_ROOT / "analysis_results" / "background_runs" / "analysis_queue_logs"


def main() -> None:
    try:
        os.nice(5)
    except OSError:
        pass
    if MANIFEST.exists():
        previous = json.loads(MANIFEST.read_text())
        if previous.get("status") == "running":
            raise RuntimeError("The requested analysis queue is already marked running")

    manifest = {
        "status": "running",
        "started_utc": utc_now(),
        "pid": os.getpid(),
        "nice_increment": 5,
        "resource_policy": {
            "device": "cpu",
            "max_threads": 4,
            "sequential_model_extraction": True,
            "minimum_free_gb": 55,
        },
        "phases": {},
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
            "MPLCONFIGDIR": "/tmp/sdt-requested-analysis-matplotlib",
        }
    )
    python = sys.executable
    commands = [
        (
            "prepare_adapter_probing_data",
            [
                python,
                "attention/run_staged_directprobe_preparation.py",
                "--models",
                "graphcodebert",
                "codet5",
                "--languages",
                "java",
                "go",
                "javascript",
                "--device",
                "cpu",
                "--omp_threads",
                "4",
                "--minimum_free_gb",
                "55",
            ],
        ),
        (
            "python_codebert",
            [
                python,
                "attention/run_staged_codebert_analysis.py",
                "--device",
                "cpu",
                "--omp_threads",
                "4",
                "--minimum_free_gb",
                "55",
            ],
        ),
        (
            "python_adapter_models",
            [
                python,
                "attention/run_staged_model_analysis.py",
                "--models",
                "graphcodebert",
                "unixcoder",
                "codet5",
                "plbart",
                "codet5p_220",
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
                "analysis_results/final_3000_multimodel/python_run_manifest.json",
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
            ],
        ),
        (
            "generate_manuscript_outputs",
            [python, "analysis_results/manuscript/generate_manuscript_figures.py"],
        ),
        (
            "build_notebook",
            [python, "analysis_results/build_multilingual_notebook.py"],
        ),
    ]
    base_python = Path("/home/abhinav/miniconda3/bin/python")
    if base_python.is_file():
        commands.append(
            (
                "execute_notebook",
                [
                    str(base_python),
                    "analysis_results/execute_notebook.py",
                    "analysis_results/multilingual_code_llm_results.ipynb",
                ],
            )
        )

    try:
        for phase, command in commands:
            free_before = shutil.disk_usage(REPO_ROOT).free
            duration = run_command(
                command, LOG_ROOT / f"{phase}.log", environment
            )
            manifest["phases"][phase] = {
                "status": "complete",
                "duration_seconds": duration,
                "finished_utc": utc_now(),
                "free_bytes_before": free_before,
                "free_bytes_after": shutil.disk_usage(REPO_ROOT).free,
            }
            atomic_json(MANIFEST, manifest)
    except Exception as exc:
        manifest.update(
            {
                "status": "failed",
                "failed_utc": utc_now(),
                "reason": f"{type(exc).__name__}: {exc}",
            }
        )
        atomic_json(MANIFEST, manifest)
        raise

    manifest.update({"status": "complete", "finished_utc": utc_now()})
    atomic_json(MANIFEST, manifest)
    print(f"[REQUESTED ANALYSIS QUEUE COMPLETE] {MANIFEST}")


if __name__ == "__main__":
    main()
