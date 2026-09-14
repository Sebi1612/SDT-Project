"""Finish the requested distance probes with six safe concurrent slots.

This runner hands off the configuration already owned by the original serial
distance lane without interrupting it.  The serial parent is stopped so that
it cannot start a duplicate configuration; five additional configurations
are started immediately.  Once the inherited configuration finishes, all six
slots are available to this scheduler.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import os
import re
import shutil
import signal
import subprocess
import time
from collections import deque
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DP_DIR = REPO_ROOT / "DirectProbe"
DP_ROOT = DP_DIR / "final_3000"
PYTHON = Path("/home/abhinav/miniconda3/envs/attention/bin/python")
LICENSE = Path(
    "/home/abhinav/sdt_project/.config/gurobi/dev-sebastian/gurobi.lic"
)
MANIFEST = DP_ROOT / "requested_distance_accelerator_manifest.json"
LOG_DIR = DP_ROOT / "distance_accelerator_logs"
LOCK_FILE = DP_ROOT / "requested_distance_accelerator.lock"
REQUIRED_RESULTS = ("clusters.txt", "prediction.txt", "dis.txt", "log.txt")
LANGUAGES = ("java", "go", "javascript")
MODELS = ("codebert", "graphcodebert", "codet5")
LAYERS = (5, 9, 12)
TARGET_CONCURRENCY = 6
WORKERS_PER_CONFIGURATION = 4
TIMEOUT_SECONDS = 172800
MAX_ATTEMPTS_PER_CONFIGURATION = 3
MINIMUM_AVAILABLE_MEMORY_GIB = 80.0
MINIMUM_FREE_DISK_GB = 50.0
LAUNCH_INTERVAL_SECONDS = 10
POLL_SECONDS = 10
CONFIG_PATTERN = re.compile(
    r"/config_files/(java|go|javascript)/distance/"
    r"config_(codebert|graphcodebert|codet5)_(5|9|12)\.ini(?:\s|$)"
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def key_name(key: tuple[str, str, int]) -> str:
    return f"{key[0]}/{key[1]}/distance/L{key[2]}"


def result_dir(key: tuple[str, str, int]) -> Path:
    language, model, layer = key
    return DP_ROOT / "results" / language / "distance" / model / str(layer)


def config_path(key: tuple[str, str, int]) -> Path:
    language, model, layer = key
    return (
        DP_ROOT
        / "config_files"
        / language
        / "distance"
        / f"config_{model}_{layer}.ini"
    )


def result_is_complete(key: tuple[str, str, int]) -> bool:
    directory = result_dir(key)
    return all((directory / name).is_file() for name in REQUIRED_RESULTS)


def result_uses_gurobi(key: tuple[str, str, int]) -> bool:
    log = result_dir(key) / "log.txt"
    return log.is_file() and "Gurobi IS found" in log.read_text(errors="replace")


def result_is_valid(key: tuple[str, str, int]) -> bool:
    return result_is_complete(key) and result_uses_gurobi(key)


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


def find_legacy_runner() -> int:
    candidates = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        command = process_command(pid)
        if (
            "run_multilingual_pilot.py" in command
            and "--tasks distance" in command
            and "--models codebert" in command
            and "final_3000" in command
        ):
            candidates.append(pid)
    if len(candidates) != 1:
        raise RuntimeError(
            "Expected exactly one active serial distance runner, found "
            f"{candidates}"
        )
    return candidates[0]


def direct_children(pid: int) -> list[int]:
    try:
        text = Path(f"/proc/{pid}/task/{pid}/children").read_text().strip()
    except (FileNotFoundError, PermissionError):
        return []
    return [int(value) for value in text.split()] if text else []


def parse_distance_key(command: str) -> tuple[str, str, int] | None:
    match = CONFIG_PATTERN.search(command)
    if not match:
        return None
    return match.group(1), match.group(2), int(match.group(3))


def find_legacy_active(runner_pid: int) -> tuple[int, tuple[str, str, int]] | None:
    matches = []
    for child_pid in direct_children(runner_pid):
        if not process_is_active(child_pid):
            continue
        command = process_command(child_pid)
        key = parse_distance_key(command)
        if key is not None and "main.py --config_file" in command:
            matches.append((child_pid, key))
    if len(matches) > 1:
        raise RuntimeError(f"Multiple legacy distance children found: {matches}")
    return matches[0] if matches else None


def available_memory_gib() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024 / 1024
    raise RuntimeError("MemAvailable is absent from /proc/meminfo")


def free_disk_gb() -> float:
    return shutil.disk_usage(REPO_ROOT).free / 1e9


def quarantine_incomplete(
    key: tuple[str, str, int], manifest: dict
) -> Path | None:
    directory = result_dir(key)
    if not directory.exists() or not any(directory.iterdir()):
        return None
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    relative = directory.relative_to(DP_ROOT)
    archive = DP_ROOT / "interrupted_results_archive" / stamp / relative
    if archive.exists():
        archive = archive.with_name(archive.name + f"_{time.time_ns()}")
    archive.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(directory), str(archive))
    manifest.setdefault("quarantined_incomplete_results", []).append(
        {
            "configuration": key_name(key),
            "source": str(directory),
            "archive": str(archive),
            "quarantined_utc": utc_now(),
        }
    )
    return archive


def terminate_process_group(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=30)


def retire_legacy_runner(runner_pid: int) -> None:
    if not process_is_active(runner_pid):
        return
    try:
        os.kill(runner_pid, signal.SIGTERM)
        os.kill(runner_pid, signal.SIGCONT)
    except ProcessLookupError:
        pass


def resume_legacy_runner(runner_pid: int) -> None:
    if process_is_active(runner_pid):
        try:
            os.kill(runner_pid, signal.SIGCONT)
        except ProcessLookupError:
            pass


def snapshot_active(active: dict[int, dict]) -> list[dict]:
    return [
        {
            "configuration": key_name(item["key"]),
            "pid": pid,
            "attempt": item["attempt"],
            "started_utc": item["started_utc"],
            "log": str(item["log_path"]),
        }
        for pid, item in sorted(active.items())
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-runner-pid", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not PYTHON.is_file():
        raise FileNotFoundError(f"Python environment is missing: {PYTHON}")
    if not LICENSE.is_file():
        raise FileNotFoundError(f"Project-local Gurobi license is missing: {LICENSE}")
    for key in ((la, mo, ly) for la in LANGUAGES for mo in MODELS for ly in LAYERS):
        if not config_path(key).is_file():
            raise FileNotFoundError(f"Missing configuration: {config_path(key)}")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    lock_handle = LOCK_FILE.open("w")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("A distance accelerator is already active") from exc

    all_keys = [
        (language, model, layer)
        for language in LANGUAGES
        for model in MODELS
        for layer in LAYERS
    ]
    initial_complete = [key for key in all_keys if result_is_valid(key)]
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "dry_run",
                    "target_concurrency": TARGET_CONCURRENCY,
                    "initial_complete": len(initial_complete),
                    "remaining": len(all_keys) - len(initial_complete),
                    "available_memory_gib": available_memory_gib(),
                    "free_disk_gb": free_disk_gb(),
                },
                indent=2,
            )
        )
        return

    runner_pid = args.legacy_runner_pid or find_legacy_runner()
    command = process_command(runner_pid)
    if "run_multilingual_pilot.py" not in command or "--tasks distance" not in command:
        raise RuntimeError(f"PID {runner_pid} is not the serial distance runner")

    os.kill(runner_pid, signal.SIGSTOP)
    time.sleep(1)
    legacy_active = find_legacy_active(runner_pid)

    try:
        os.nice(5)
    except OSError:
        pass

    manifest = {
        "status": "running",
        "started_utc": utc_now(),
        "pid": os.getpid(),
        "target_concurrent_distance_configurations": TARGET_CONCURRENCY,
        "workers_per_configuration": WORKERS_PER_CONFIGURATION,
        "maximum_directprobe_workers": (
            TARGET_CONCURRENCY * WORKERS_PER_CONFIGURATION
        ),
        "gurobi_threads_per_lp": 1,
        "minimum_available_memory_gib_to_launch": MINIMUM_AVAILABLE_MEMORY_GIB,
        "minimum_free_disk_gb_to_launch": MINIMUM_FREE_DISK_GB,
        "timeout_seconds_per_configuration": TIMEOUT_SECONDS,
        "maximum_attempts_per_configuration": MAX_ATTEMPTS_PER_CONFIGURATION,
        "license_scope": "child-process environment only",
        "license_file": str(LICENSE),
        "legacy_handoff": {
            "runner_pid": runner_pid,
            "runner_stopped": True,
            "active_pid": legacy_active[0] if legacy_active else 0,
            "active_configuration": (
                key_name(legacy_active[1]) if legacy_active else None
            ),
        },
        "initial_complete": len(initial_complete),
        "runs": {},
        "active": [],
    }
    atomic_json(manifest)

    child_environment = os.environ.copy()
    child_environment.update(
        {
            "GRB_LICENSE_FILE": str(LICENSE),
            "DIRECTPROBE_N_JOBS": str(WORKERS_PER_CONFIGURATION),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
    )

    inherited_key = legacy_active[1] if legacy_active else None
    inherited_pid = legacy_active[0] if legacy_active else 0
    pending = deque(
        key
        for key in all_keys
        if key not in initial_complete and key != inherited_key
    )
    attempts = {key: 0 for key in all_keys}
    active: dict[int, dict] = {}
    failures: list[dict] = []
    completed = set(initial_complete)
    last_launch = 0.0
    succeeded = False
    termination_requested = False

    def request_termination(signum, _frame):
        nonlocal termination_requested
        termination_requested = True
        manifest["termination_signal"] = signum

    signal.signal(signal.SIGTERM, request_termination)
    signal.signal(signal.SIGINT, request_termination)

    try:
        while pending or active or inherited_pid:
            if termination_requested:
                raise RuntimeError("Distance accelerator received a termination signal")

            state_changed = False
            if inherited_pid and not process_is_active(inherited_pid):
                if result_is_valid(inherited_key):
                    completed.add(inherited_key)
                    manifest["runs"][key_name(inherited_key)] = {
                        "status": "complete",
                        "source": "legacy_handoff",
                        "pid": inherited_pid,
                        "finished_utc": utc_now(),
                    }
                else:
                    quarantine_incomplete(inherited_key, manifest)
                    pending.appendleft(inherited_key)
                    manifest["runs"][key_name(inherited_key)] = {
                        "status": "legacy_incomplete_requeued",
                        "pid": inherited_pid,
                        "detected_utc": utc_now(),
                    }
                manifest["legacy_handoff"]["status"] = "finished"
                manifest["legacy_handoff"]["finished_utc"] = utc_now()
                inherited_pid = 0
                state_changed = True

            for pid, item in list(active.items()):
                process = item["process"]
                timed_out = time.monotonic() - item["started_monotonic"] > TIMEOUT_SECONDS
                if timed_out and process.poll() is None:
                    terminate_process_group(process)
                    item["timed_out"] = True
                return_code = process.poll()
                if return_code is None:
                    continue
                item["log_handle"].close()
                del active[pid]
                key = item["key"]
                valid = return_code == 0 and result_is_valid(key)
                record = {
                    "status": "complete" if valid else "failed_attempt",
                    "attempt": item["attempt"],
                    "pid": pid,
                    "return_code": return_code,
                    "duration_seconds": time.monotonic() - item["started_monotonic"],
                    "log": str(item["log_path"]),
                    "finished_utc": utc_now(),
                }
                if item.get("timed_out"):
                    record["timed_out"] = True
                manifest["runs"][key_name(key)] = record
                if valid:
                    completed.add(key)
                elif attempts[key] < MAX_ATTEMPTS_PER_CONFIGURATION:
                    pending.append(key)
                else:
                    failures.append({"configuration": key_name(key), **record})
                state_changed = True

            inherited_slots = 1 if inherited_pid and process_is_active(inherited_pid) else 0
            slots = TARGET_CONCURRENCY - inherited_slots - len(active)
            now = time.monotonic()
            if pending and slots > 0 and now - last_launch >= LAUNCH_INTERVAL_SECONDS:
                memory = available_memory_gib()
                disk = free_disk_gb()
                manifest["capacity"] = {
                    "available_memory_gib": memory,
                    "free_disk_gb": disk,
                    "checked_utc": utc_now(),
                }
                if memory >= MINIMUM_AVAILABLE_MEMORY_GIB and disk >= MINIMUM_FREE_DISK_GB:
                    key = pending.popleft()
                    if result_is_valid(key):
                        completed.add(key)
                    else:
                        quarantine_incomplete(key, manifest)
                        attempts[key] += 1
                        path = config_path(key)
                        log_path = LOG_DIR / (
                            f"{key[0]}_{key[1]}_{key[2]}_attempt{attempts[key]}.log"
                        )
                        log_handle = log_path.open("a")
                        log_handle.write(
                            f"started_utc={utc_now()}\n"
                            f"configuration={key_name(key)}\n"
                        )
                        log_handle.flush()
                        process = subprocess.Popen(
                            [str(PYTHON), "main.py", "--config_file", str(path)],
                            cwd=DP_DIR,
                            env=child_environment,
                            stdout=log_handle,
                            stderr=subprocess.STDOUT,
                            text=True,
                            start_new_session=True,
                        )
                        active[process.pid] = {
                            "process": process,
                            "key": key,
                            "attempt": attempts[key],
                            "started_monotonic": time.monotonic(),
                            "started_utc": utc_now(),
                            "log_path": log_path,
                            "log_handle": log_handle,
                        }
                        manifest["runs"][key_name(key)] = {
                            "status": "running",
                            "attempt": attempts[key],
                            "pid": process.pid,
                            "started_utc": active[process.pid]["started_utc"],
                            "log": str(log_path),
                        }
                    last_launch = now
                    state_changed = True
                else:
                    manifest["capacity_wait"] = {
                        "available_memory_gib": memory,
                        "free_disk_gb": disk,
                        "detected_utc": utc_now(),
                    }

            if state_changed or int(time.monotonic()) % 30 < POLL_SECONDS:
                manifest["active"] = snapshot_active(active)
                manifest["legacy_handoff"]["active"] = bool(inherited_pid)
                manifest["num_complete"] = len(completed)
                manifest["num_pending"] = len(pending)
                manifest["num_active_accelerated"] = len(active)
                manifest["num_active_total"] = len(active) + (
                    1 if inherited_pid and process_is_active(inherited_pid) else 0
                )
                manifest["heartbeat_utc"] = utc_now()
                atomic_json(manifest)
            time.sleep(POLL_SECONDS)

        invalid = [key_name(key) for key in all_keys if not result_is_valid(key)]
        if failures or invalid:
            raise RuntimeError(
                f"Distance accelerator ended with failures={failures}, invalid={invalid}"
            )

        retire_legacy_runner(runner_pid)
        manifest.update(
            {
                "status": "complete",
                "num_complete": len(all_keys),
                "num_pending": 0,
                "num_active_accelerated": 0,
                "num_active_total": 0,
                "active": [],
                "legacy_runner_retired": True,
                "finished_utc": utc_now(),
            }
        )
        atomic_json(manifest)
        succeeded = True
        print(f"[DISTANCE ACCELERATOR COMPLETE] {len(all_keys)}/{len(all_keys)}")
    except Exception as exc:
        manifest.update(
            {
                "status": "failed",
                "reason": f"{type(exc).__name__}: {exc}",
                "failed_utc": utc_now(),
            }
        )
        atomic_json(manifest)
        raise
    finally:
        if not succeeded:
            for item in active.values():
                terminate_process_group(item["process"])
                item["log_handle"].close()
            resume_legacy_runner(runner_pid)


if __name__ == "__main__":
    main()
