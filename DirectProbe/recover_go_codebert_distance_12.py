"""Recover the completed-cluster prediction phase for Go/CodeBERT distance L12."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

from run_multilingual_pilot import parse_result


REPO_ROOT = Path(__file__).resolve().parent.parent
DP_DIR = REPO_ROOT / "DirectProbe"
DP_ROOT = DP_DIR / "final_3000"
PYTHON = Path("/home/abhinav/miniconda3/envs/attention/bin/python")
LICENSE = Path(
    "/home/abhinav/sdt_project/.config/gurobi/dev-sebastian/gurobi.lic"
)
CONFIG = (
    DP_ROOT
    / "config_files/go/distance/config_codebert_12_prediction_recovery.ini"
)
CANONICAL = DP_ROOT / "results/go/distance/codebert/12"
RECOVERY = DP_ROOT / "recovery/go_codebert_distance_12_prediction"
SOURCE = (
    DP_ROOT
    / "interrupted_results_archive/20260831T111518Z/results/go/distance/codebert/12"
)
MANIFEST = DP_ROOT / "go_codebert_distance_12_recovery_manifest.json"
CAPTURE_LOG = DP_ROOT / "go_codebert_distance_12_recovery.log"
REQUIRED_RESULTS = ("clusters.txt", "prediction.txt", "dis.txt", "log.txt")


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_json(value: dict) -> None:
    temporary = MANIFEST.with_suffix(MANIFEST.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    os.replace(temporary, MANIFEST)


def process_state(pid: int) -> str | None:
    try:
        return Path(f"/proc/{pid}/stat").read_text().split()[2]
    except (FileNotFoundError, IndexError, PermissionError):
        return None


def process_is_active(pid: int) -> bool:
    state = process_state(pid)
    return state is not None and state != "Z"


def process_command(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    except (FileNotFoundError, PermissionError, UnicodeDecodeError):
        return ""


def stop_stalled_process(pid: int) -> str:
    if not process_is_active(pid):
        return "already_inactive"
    os.kill(pid, signal.SIGTERM)
    if process_state(pid) == "T":
        os.kill(pid, signal.SIGCONT)
    deadline = time.monotonic() + 30
    while process_is_active(pid) and time.monotonic() < deadline:
        time.sleep(1)
    if process_is_active(pid):
        os.kill(pid, signal.SIGKILL)
        deadline = time.monotonic() + 30
        while process_is_active(pid) and time.monotonic() < deadline:
            time.sleep(1)
        if process_is_active(pid):
            raise RuntimeError(f"Stalled process {pid} did not terminate")
        return "killed"
    return "terminated"


def archive_existing_recovery() -> str | None:
    if not RECOVERY.exists() or not any(RECOVERY.iterdir()):
        return None
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = (
        DP_ROOT
        / "interrupted_results_archive"
        / stamp
        / "recovery/go_codebert_distance_12_prediction"
    )
    archive.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(RECOVERY), str(archive))
    return str(archive)


def atomic_copy(source: Path, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".recovery.tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def archive_partial_canonical() -> str | None:
    if not CANONICAL.exists() or not any(CANONICAL.iterdir()):
        return None
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = (
        DP_ROOT
        / "interrupted_results_archive"
        / stamp
        / "replaced_full_rerun/go/distance/codebert/12"
    )
    archive.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(CANONICAL), str(archive))
    return str(archive)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accelerator-pid", type=int, required=True)
    parser.add_argument("--stalled-pid", type=int, required=True)
    args = parser.parse_args()

    if "run_requested_distance_accelerator.py" not in process_command(
        args.accelerator_pid
    ):
        raise RuntimeError(
            f"PID {args.accelerator_pid} is not the distance accelerator"
        )
    stalled_command = process_command(args.stalled_pid)
    if (
        "main.py --config_file" not in stalled_command
        or "/go/distance/config_codebert_12.ini" not in stalled_command
    ):
        raise RuntimeError(f"PID {args.stalled_pid} is not the expected stalled run")
    if not LICENSE.is_file() or not PYTHON.is_file() or not CONFIG.is_file():
        raise FileNotFoundError("A required Python, license, or config file is missing")
    clusters = SOURCE / "clusters.txt"
    original_log = SOURCE / "log.txt"
    if not clusters.is_file() or not original_log.is_file():
        raise FileNotFoundError("Archived clusters or original log are missing")
    if sum(1 for _ in clusters.open()) != 5200:
        raise ValueError("Expected exactly 5,200 recovered cluster assignments")
    if "Gurobi IS found" not in original_log.read_text(errors="replace"):
        raise ValueError("Original probing log does not confirm Gurobi")

    manifest = {
        "status": "running",
        "started_utc": utc_now(),
        "accelerator_pid": args.accelerator_pid,
        "stalled_pid": args.stalled_pid,
        "strategy": "reuse validated clusters and rerun prediction only",
        "archived_cluster_source": str(SOURCE),
        "canonical_result_dir": str(CANONICAL),
        "isolated_recovery_dir": str(RECOVERY),
        "config": str(CONFIG),
        "license_scope": "recovery subprocess environment only",
        "license_file": str(LICENSE),
        "cluster_assignments": 5200,
    }
    atomic_json(manifest)

    full_rerun_paused = False
    recovery_process: subprocess.Popen | None = None
    termination_requested = False

    def request_termination(signum, _frame):
        nonlocal termination_requested
        termination_requested = True
        if recovery_process is not None and recovery_process.poll() is None:
            recovery_process.terminate()

    signal.signal(signal.SIGTERM, request_termination)
    signal.signal(signal.SIGINT, request_termination)

    try:
        archived = archive_existing_recovery()
        if archived:
            manifest["previous_recovery_archive"] = archived
        atomic_json(manifest)

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
        started = time.monotonic()
        with CAPTURE_LOG.open("a") as capture:
            capture.write(
                f"\nstarted_utc={utc_now()}\n"
                "mode=prediction-only cluster recovery\n"
            )
            capture.flush()
            recovery_process = subprocess.Popen(
                [str(PYTHON), "main.py", "--config_file", str(CONFIG)],
                cwd=DP_DIR,
                env=environment,
                stdout=capture,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            manifest["recovery_pid"] = recovery_process.pid
            manifest["recovery_started_utc"] = utc_now()
            atomic_json(manifest)
            try:
                return_code = recovery_process.wait(timeout=21600)
            except subprocess.TimeoutExpired:
                os.killpg(recovery_process.pid, signal.SIGTERM)
                recovery_process.wait(timeout=30)
                raise TimeoutError("Prediction-only recovery exceeded six hours")
        if termination_requested:
            raise RuntimeError("Recovery received a termination signal")
        if return_code != 0:
            raise RuntimeError(f"Prediction recovery exited {return_code}")

        expected = ("prediction.txt", "dis.txt", "log.txt")
        missing = [name for name in expected if not (RECOVERY / name).is_file()]
        if missing:
            raise RuntimeError(f"Recovery outputs are missing: {missing}")
        if not (RECOVERY / "prediction.txt").stat().st_size:
            raise ValueError("Recovered predictions are empty")
        if not (RECOVERY / "dis.txt").stat().st_size:
            raise ValueError("Recovered inter-cluster distances are empty")

        current_command = process_command(args.stalled_pid)
        if (
            "main.py --config_file" not in current_command
            or "/go/distance/config_codebert_12.ini" not in current_command
        ):
            raise RuntimeError(
                "The concurrent full-rerun PID changed before recovery finished"
            )
        os.kill(args.stalled_pid, signal.SIGSTOP)
        full_rerun_paused = True
        manifest["full_rerun_paused_utc"] = utc_now()
        partial_archive = archive_partial_canonical()
        if partial_archive:
            manifest["replaced_full_rerun_archive"] = partial_archive
        CANONICAL.mkdir(parents=True, exist_ok=True)
        atomic_copy(clusters, CANONICAL / "clusters.txt")
        atomic_copy(RECOVERY / "prediction.txt", CANONICAL / "prediction.txt")
        atomic_copy(RECOVERY / "dis.txt", CANONICAL / "dis.txt")
        merged_log = (
            original_log.read_text(errors="replace")
            + "\n\n"
            + "=== PREDICTION-ONLY RECOVERY ===\n"
            + (RECOVERY / "log.txt").read_text(errors="replace")
        )
        temporary_log = CANONICAL / "log.txt.recovery.tmp"
        temporary_log.write_text(merged_log)
        os.replace(temporary_log, CANONICAL / "log.txt")

        manifest["full_rerun_resolution"] = stop_stalled_process(
            args.stalled_pid
        )
        full_rerun_paused = False
        manifest["full_rerun_stopped_utc"] = utc_now()

        missing = [name for name in REQUIRED_RESULTS if not (CANONICAL / name).is_file()]
        if missing:
            raise RuntimeError(f"Canonical recovery is missing: {missing}")
        summary = parse_result(CANONICAL)
        if summary["solver"] != "gurobi":
            raise ValueError("Recovered canonical result lost Gurobi provenance")
        if summary["num_test"] != 1300:
            raise ValueError(
                f"Expected 1,300 predictions, found {summary['num_test']}"
            )

        manifest.update(
            {
                "status": "complete",
                "duration_seconds": time.monotonic() - started,
                "finished_utc": utc_now(),
                "validation": {
                    "solver": summary["solver"],
                    "num_test": summary["num_test"],
                    "accuracy": summary["accuracy"],
                    "num_clusters": summary["num_clusters"],
                    "required_files": list(REQUIRED_RESULTS),
                },
            }
        )
        atomic_json(manifest)
        print("[GO CODEBERT DISTANCE L12 RECOVERY COMPLETE]")
    except Exception as exc:
        manifest.update(
            {
                "status": "failed",
                "reason": f"{type(exc).__name__}: {exc}",
                "failed_utc": utc_now(),
                "fallback": (
                    "the concurrent accelerator-owned full rerun remains the "
                    "fallback"
                ),
            }
        )
        atomic_json(manifest)
        raise
    finally:
        if full_rerun_paused and process_is_active(args.stalled_pid):
            os.kill(args.stalled_pid, signal.SIGCONT)


if __name__ == "__main__":
    main()
