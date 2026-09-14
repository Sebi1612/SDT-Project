"""Keep the requested DirectProbe matrix alive and resumable in the background."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON = Path("/home/abhinav/miniconda3/envs/attention/bin/python")
QUEUE_RUNNER = REPO_ROOT / "DirectProbe" / "run_requested_probe_queue.py"
DP_ROOT = REPO_ROOT / "DirectProbe" / "final_3000"
QUEUE_MANIFEST = DP_ROOT / "requested_probing_parallel_queue_manifest.json"
WATCHDOG_MANIFEST = DP_ROOT / "requested_probing_watchdog_manifest.json"
WATCHDOG_LOG = DP_ROOT / "requested_probing_watchdog.log"
LICENSE = Path(
    "/home/abhinav/sdt_project/.config/gurobi/dev-sebastian/gurobi.lic"
)
REQUIRED_RESULTS = ("clusters.txt", "prediction.txt", "dis.txt", "log.txt")
LANGUAGES = ("java", "go", "javascript")
MODELS = ("codebert", "graphcodebert", "codet5")
TASKS = ("siblings", "siblings_id", "dfg", "distance", "distance_id")
LAYERS = (5, 9, 12)
MATRIX_SIZE = 135


def utc_now() -> str:
    import datetime as dt

    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    os.replace(temporary, path)


def process_is_watchdog(pid: int) -> bool:
    try:
        state = Path(f"/proc/{pid}/stat").read_text().split()[2]
        command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ")
    except (FileNotFoundError, IndexError, PermissionError):
        return False
    return state != "Z" and b"run_requested_probe_watchdog.py" in command


def result_counts() -> dict:
    complete = partial = missing = 0
    for language in LANGUAGES:
        for model in MODELS:
            for task in TASKS:
                for layer in LAYERS:
                    result_dir = (
                        DP_ROOT / "results" / language / task / model / str(layer)
                    )
                    count = sum(
                        (result_dir / name).is_file() for name in REQUIRED_RESULTS
                    )
                    if count == len(REQUIRED_RESULTS):
                        complete += 1
                    elif result_dir.exists() and any(result_dir.iterdir()):
                        partial += 1
                    else:
                        missing += 1
    return {
        "complete": complete,
        "partial": partial,
        "missing": missing,
        "total": MATRIX_SIZE,
    }


def available_memory_gb() -> float:
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        values[key] = int(value.strip().split()[0])
    return values["MemAvailable"] / 1024 / 1024


def wait_for_capacity(minimum_disk_gb: float, minimum_memory_gb: float) -> dict:
    while True:
        disk_gb = shutil.disk_usage(REPO_ROOT).free / 1e9
        memory_gb = available_memory_gb()
        if disk_gb >= minimum_disk_gb and memory_gb >= minimum_memory_gb:
            return {"free_disk_gb": disk_gb, "available_memory_gb": memory_gb}
        with WATCHDOG_LOG.open("a") as log:
            log.write(
                f"{utc_now()} waiting for capacity: disk={disk_gb:.1f} GB, "
                f"memory={memory_gb:.1f} GiB\n"
            )
        time.sleep(300)


def main() -> None:
    if not LICENSE.is_file():
        raise FileNotFoundError(f"Project-local Gurobi license missing: {LICENSE}")
    if not PYTHON.is_file():
        raise FileNotFoundError(f"Python environment missing: {PYTHON}")
    if WATCHDOG_MANIFEST.is_file():
        previous = json.loads(WATCHDOG_MANIFEST.read_text())
        previous_pid = previous.get("pid")
        if (
            previous.get("status") == "running"
            and previous_pid
            and process_is_watchdog(int(previous_pid))
        ):
            raise RuntimeError(f"Probe watchdog already active as PID {previous_pid}")

    try:
        os.nice(5)
    except OSError:
        pass
    manifest = {
        "status": "running",
        "started_utc": utc_now(),
        "pid": os.getpid(),
        "queue_runner": str(QUEUE_RUNNER),
        "queue_manifest": str(QUEUE_MANIFEST),
        "matrix_size": MATRIX_SIZE,
        "maximum_restart_attempts": 20,
        "restart_backoff_seconds": 60,
        "minimum_free_disk_gb": 50,
        "minimum_available_memory_gb": 64,
        "resource_limit": {
            "task_lanes": 5,
            "configurations_per_lane": 1,
            "workers_per_configuration": 4,
            "maximum_workers": 20,
            "gurobi_threads_per_lp": 1,
        },
        "license_scope": "child-process environment only",
        "license_file": str(LICENSE),
        "initial_results": result_counts(),
        "attempts": [],
    }
    atomic_json(WATCHDOG_MANIFEST, manifest)

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
    for attempt_number in range(1, 21):
        capacity = wait_for_capacity(50, 64)
        started = time.monotonic()
        attempt = {
            "attempt": attempt_number,
            "status": "running",
            "started_utc": utc_now(),
            "capacity_at_start": capacity,
            "results_at_start": result_counts(),
        }
        manifest["attempts"].append(attempt)
        atomic_json(WATCHDOG_MANIFEST, manifest)
        with WATCHDOG_LOG.open("a") as log:
            log.write(
                f"\n{utc_now()} starting queue attempt {attempt_number}/20\n"
            )
            log.flush()
            process = subprocess.run(
                [str(PYTHON), str(QUEUE_RUNNER)],
                cwd=REPO_ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        counts = result_counts()
        queue_status = None
        if QUEUE_MANIFEST.is_file():
            queue_status = json.loads(QUEUE_MANIFEST.read_text()).get("status")
        attempt.update(
            {
                "status": "complete" if process.returncode == 0 else "failed",
                "exit_code": process.returncode,
                "queue_status": queue_status,
                "duration_seconds": time.monotonic() - started,
                "results_at_finish": counts,
                "finished_utc": utc_now(),
            }
        )
        atomic_json(WATCHDOG_MANIFEST, manifest)
        if (
            process.returncode == 0
            and queue_status == "complete"
            and counts["complete"] == MATRIX_SIZE
        ):
            manifest.update(
                {
                    "status": "complete",
                    "finished_utc": utc_now(),
                    "final_results": counts,
                }
            )
            atomic_json(WATCHDOG_MANIFEST, manifest)
            print(f"[PROBE WATCHDOG COMPLETE] {MATRIX_SIZE}/{MATRIX_SIZE}")
            return
        time.sleep(60)

    manifest.update(
        {
            "status": "failed_after_maximum_restarts",
            "failed_utc": utc_now(),
            "final_results": result_counts(),
        }
    )
    atomic_json(WATCHDOG_MANIFEST, manifest)
    raise RuntimeError("DirectProbe queue did not finish after 20 attempts")


if __name__ == "__main__":
    main()
