"""Run five resource-bounded DirectProbe task lanes with a local licence.

Each task is an independent lane. Within a lane, model/language/layer
configurations remain sequential, while the five tasks run concurrently. A
lane first finishes CodeBERT, waits for validated adapter datasets, and then
finishes GraphCodeBERT and CodeT5.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import signal
import subprocess
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON = Path("/home/abhinav/miniconda3/envs/attention/bin/python")
DP_ROOT = REPO_ROOT / "DirectProbe" / "final_3000"
LICENSE = Path(
    "/home/abhinav/sdt_project/.config/gurobi/dev-sebastian/gurobi.lic"
)
QUEUE_MANIFEST = DP_ROOT / "requested_probing_parallel_queue_manifest.json"
PREPARATION_MANIFEST = DP_ROOT / "directprobe_preparation_manifest.json"
TASKS = ["siblings", "siblings_id", "dfg", "distance", "distance_id"]
LAYERS = ["5", "9", "12"]
LANGUAGES = ["java", "go", "javascript"]
REQUIRED_RESULTS = ("clusters.txt", "prediction.txt", "dis.txt", "log.txt")


def utc_now() -> str:
    import datetime as dt

    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    os.replace(temporary, path)


def run_group(task: str, models: list[str], phase: str, env: dict) -> dict:
    manifest_name = f"directprobe_run_manifest_{task}_{phase}_requested.json"
    log_path = DP_ROOT / f"directprobe_queue_{task}_{phase}.log"
    command = [
        str(PYTHON),
        "DirectProbe/run_multilingual_pilot.py",
        "--dp_dir",
        "DirectProbe",
        "--pilot_root",
        "DirectProbe/final_3000",
        "--languages",
        *LANGUAGES,
        "--models",
        *models,
        "--tasks",
        task,
        "--layers",
        *LAYERS,
        "--workers",
        "4",
        "--timeout",
        "172800",
        "--required_solver",
        "gurobi",
        "--quarantine_incomplete",
        "--manifest_name",
        manifest_name,
    ]
    started = time.monotonic()
    with log_path.open("a") as log:
        log.write(f"\nstarted_utc={utc_now()}\n")
        log.write("command=" + json.dumps(command) + "\n")
        log.flush()
        process = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        duration = time.monotonic() - started
        log.write(f"finished_utc={utc_now()}\n")
        log.write(f"duration_seconds={duration}\n")
        log.write(f"exit_code={process.returncode}\n")
    if process.returncode:
        raise RuntimeError(f"DirectProbe runner exited {process.returncode}: {log_path}")
    run_manifest_path = DP_ROOT / manifest_name
    run_manifest = json.loads(run_manifest_path.read_text())
    if run_manifest.get("status") != "complete":
        raise RuntimeError(
            f"DirectProbe group is {run_manifest.get('status')}: {manifest_name}"
        )
    if run_manifest.get("num_failed") != 0:
        raise RuntimeError(f"DirectProbe group contains failures: {manifest_name}")
    wrong_solver = [
        run
        for run in run_manifest.get("runs", [])
        if run.get("status") == "complete" and run.get("solver") != "gurobi"
    ]
    if wrong_solver:
        raise RuntimeError(f"Non-Gurobi results detected in {manifest_name}")
    return {
        "status": "complete",
        "models": models,
        "duration_seconds": duration,
        "manifest": str(run_manifest_path.resolve()),
        "log": str(log_path.resolve()),
        "num_complete": run_manifest.get("num_complete"),
        "finished_utc": utc_now(),
    }


def adapter_data_complete() -> bool:
    if not PREPARATION_MANIFEST.is_file():
        return False
    preparation = json.loads(PREPARATION_MANIFEST.read_text())
    if preparation.get("status") != "complete":
        return False
    expected = {
        f"{language}/{model}"
        for language in LANGUAGES
        for model in ("graphcodebert", "codet5")
    }
    complete = {
        key
        for key, run in preparation.get("runs", {}).items()
        if run.get("status") == "complete"
    }
    return expected <= complete


def wait_for_adapter_data(timeout_seconds: int = 21600) -> float:
    started = time.monotonic()
    while not adapter_data_complete():
        if time.monotonic() - started >= timeout_seconds:
            raise TimeoutError(
                f"Adapter probing data did not complete within {timeout_seconds}s"
            )
        time.sleep(15)
    return time.monotonic() - started


def process_is_active(pid: int) -> bool:
    """Return false for missing processes and unreaped zombie children."""
    try:
        state = Path(f"/proc/{pid}/stat").read_text().split()[2]
    except (FileNotFoundError, IndexError, PermissionError):
        return False
    return state != "Z"


def result_is_complete(result_dir: Path) -> bool:
    return all((result_dir / name).is_file() for name in REQUIRED_RESULTS)


def retire_stopped_legacy_queue(runner_pid: int, wrapper_pid: int) -> None:
    """Terminate the stopped serial parent after its active child has exited."""
    for pid in (runner_pid, wrapper_pid):
        if not process_is_active(pid):
            continue
        try:
            os.kill(pid, signal.SIGTERM)
            os.kill(pid, signal.SIGCONT)
        except ProcessLookupError:
            pass


def wait_for_handoff(args: argparse.Namespace) -> dict | None:
    if not args.handoff_active_pid:
        return None
    result_dir = (
        DP_ROOT
        / "results"
        / args.handoff_language
        / args.handoff_task
        / args.handoff_model
        / str(args.handoff_layer)
    )
    started = time.monotonic()
    while process_is_active(args.handoff_active_pid):
        if time.monotonic() - started >= args.handoff_timeout:
            raise TimeoutError(
                f"Legacy active configuration {args.handoff_active_pid} exceeded "
                f"the {args.handoff_timeout}s handoff timeout"
            )
        time.sleep(10)
    if not result_is_complete(result_dir):
        raise RuntimeError(
            f"Legacy configuration exited without complete results: {result_dir}"
        )
    retire_stopped_legacy_queue(
        args.handoff_runner_pid, args.handoff_wrapper_pid
    )
    return {
        "status": "complete",
        "result_dir": str(result_dir.resolve()),
        "legacy_active_pid": args.handoff_active_pid,
        "legacy_runner_pid": args.handoff_runner_pid,
        "legacy_wrapper_pid": args.handoff_wrapper_pid,
        "wait_seconds": time.monotonic() - started,
        "finished_utc": utc_now(),
    }


def run_task_lane(task: str, environment: dict, args: argparse.Namespace) -> dict:
    lane = {
        "status": "running",
        "task": task,
        "started_utc": utc_now(),
        "phases": {},
    }
    if task == args.handoff_task and args.handoff_active_pid:
        lane["phases"]["legacy_handoff"] = wait_for_handoff(args)
    lane["phases"]["codebert"] = run_group(
        task, ["codebert"], "codebert", environment
    )
    lane["adapter_wait_seconds"] = wait_for_adapter_data()
    lane["phases"]["adapter_models"] = run_group(
        task, ["graphcodebert", "codet5"], "adapters", environment
    )
    lane.update({"status": "complete", "finished_utc": utc_now()})
    return lane


def parse_args() -> argparse.Namespace:
    cli = argparse.ArgumentParser()
    cli.add_argument("--handoff-wrapper-pid", type=int, default=0)
    cli.add_argument("--handoff-runner-pid", type=int, default=0)
    cli.add_argument("--handoff-active-pid", type=int, default=0)
    cli.add_argument("--handoff-task", choices=TASKS, default="siblings")
    cli.add_argument("--handoff-language", choices=LANGUAGES, default="java")
    cli.add_argument(
        "--handoff-model",
        choices=["codebert", "graphcodebert", "codet5"],
        default="codebert",
    )
    cli.add_argument("--handoff-layer", choices=[5, 9, 12], type=int, default=9)
    cli.add_argument("--handoff-timeout", type=int, default=172800)
    return cli.parse_args()


def main() -> None:
    args = parse_args()
    if not LICENSE.is_file():
        raise FileNotFoundError(f"Required project-local Gurobi licence: {LICENSE}")
    if not PYTHON.is_file():
        raise FileNotFoundError(f"Attention environment Python: {PYTHON}")
    if bool(args.handoff_active_pid) != bool(args.handoff_runner_pid):
        raise ValueError("Handoff active and runner PIDs must be supplied together")
    try:
        os.nice(5)
    except OSError:
        pass

    if args.handoff_runner_pid:
        if not process_is_active(args.handoff_runner_pid):
            raise RuntimeError(
                f"Legacy serial runner is not active: {args.handoff_runner_pid}"
            )
        os.kill(args.handoff_runner_pid, signal.SIGSTOP)

    manifest = {
        "status": "running",
        "started_utc": utc_now(),
        "pid": os.getpid(),
        "languages": LANGUAGES,
        "models": ["codebert", "graphcodebert", "codet5"],
        "tasks": TASKS,
        "layers": [int(layer) for layer in LAYERS],
        "matrix_size": 135,
        "workers_per_configuration": 4,
        "concurrent_task_lanes": 5,
        "maximum_concurrent_configurations": 5,
        "maximum_directprobe_workers": 20,
        "gurobi_threads_per_lp": 1,
        "timeout_seconds_per_configuration": 172800,
        "license_scope": "subprocess environment only",
        "license_file": str(LICENSE),
        "legacy_handoff": {
            "wrapper_pid": args.handoff_wrapper_pid,
            "runner_pid": args.handoff_runner_pid,
            "active_pid": args.handoff_active_pid,
            "task": args.handoff_task if args.handoff_active_pid else None,
        },
        "lanes": {
            task: {
                "status": "running",
                "task": task,
                "started_utc": utc_now(),
            }
            for task in TASKS
        },
    }
    atomic_json(QUEUE_MANIFEST, manifest)

    environment = os.environ.copy()
    environment.update(
        {
            "GRB_LICENSE_FILE": str(LICENSE),
            "DIRECTPROBE_N_JOBS": "4",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
    )

    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(TASKS)) as executor:
        futures = {
            executor.submit(run_task_lane, task, environment.copy(), args): task
            for task in TASKS
        }
        for future in concurrent.futures.as_completed(futures):
            task = futures[future]
            try:
                manifest["lanes"][task] = future.result()
            except Exception as exc:
                failure = {
                    "status": "failed",
                    "task": task,
                    "failed_utc": utc_now(),
                    "reason": f"{type(exc).__name__}: {exc}",
                }
                manifest["lanes"][task] = failure
                failures.append(failure)
            atomic_json(QUEUE_MANIFEST, manifest)

    if failures:
        manifest.update(
            {
                "status": "complete_with_failures",
                "finished_utc": utc_now(),
                "num_failed_lanes": len(failures),
            }
        )
    else:
        manifest.update(
            {
                "status": "complete",
                "finished_utc": utc_now(),
                "num_complete_lanes": len(TASKS),
            }
        )
    atomic_json(QUEUE_MANIFEST, manifest)
    if failures:
        raise RuntimeError(f"{len(failures)} DirectProbe task lanes failed")
    print(f"[PARALLEL PROBING QUEUE COMPLETE] {QUEUE_MANIFEST}")


if __name__ == "__main__":
    main()
