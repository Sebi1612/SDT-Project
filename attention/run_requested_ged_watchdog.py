"""Restart the resumable GED queue after ordinary non-zero exits."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2)
    os.replace(temporary, path)


def parse_args():
    cli = argparse.ArgumentParser()
    cli.add_argument("--python", default=sys.executable)
    cli.add_argument("--workers", type=int, default=12)
    cli.add_argument("--max_queue_attempts", type=int, default=20)
    cli.add_argument("--retry_delay_seconds", type=float, default=60.0)
    cli.add_argument(
        "--queue_manifest",
        default="analysis_results/ged_final_3000/ged_queue_manifest.json",
    )
    cli.add_argument(
        "--manifest",
        default="analysis_results/ged_final_3000/ged_watchdog_manifest.json",
    )
    cli.add_argument(
        "--log", default="analysis_results/ged_final_3000/ged_watchdog.log"
    )
    return cli.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    log_path = Path(args.log)
    queue_manifest = Path(args.queue_manifest)
    manifest = {
        "status": "running",
        "started_utc": utc_now(),
        "pid": os.getpid(),
        "workers": args.workers,
        "max_queue_attempts": args.max_queue_attempts,
        "queue_manifest": str(queue_manifest.resolve()),
        "attempts": [],
    }
    atomic_json(manifest_path, manifest)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("a") as log:
        for attempt in range(1, args.max_queue_attempts + 1):
            started = utc_now()
            command = [
                args.python,
                "attention/run_requested_ged_queue.py",
                "--python", args.python,
                "--workers", str(args.workers),
                "--manifest", str(queue_manifest),
            ]
            log.write(f"{started} starting GED queue attempt {attempt}\n")
            log.flush()
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            queue_status = None
            if queue_manifest.is_file():
                try:
                    queue_status = json.loads(queue_manifest.read_text()).get("status")
                except json.JSONDecodeError:
                    queue_status = "unreadable"
            record = {
                "attempt": attempt,
                "started_utc": started,
                "finished_utc": utc_now(),
                "exit_code": completed.returncode,
                "queue_status": queue_status,
            }
            manifest["attempts"].append(record)
            if completed.returncode == 0 and queue_status == "complete":
                manifest.update({"status": "complete", "finished_utc": utc_now()})
                atomic_json(manifest_path, manifest)
                log.write(f"{utc_now()} GED watchdog complete\n")
                log.flush()
                return
            atomic_json(manifest_path, manifest)
            log.write(
                f"{utc_now()} attempt {attempt} exited {completed.returncode}; "
                f"queue status {queue_status}; retrying\n"
            )
            log.flush()
            time.sleep(args.retry_delay_seconds)

    manifest.update(
        {
            "status": "failed",
            "failed_utc": utc_now(),
            "reason": "maximum queue attempts exhausted",
        }
    )
    atomic_json(manifest_path, manifest)
    raise RuntimeError("GED watchdog exhausted all queue attempts")


if __name__ == "__main__":
    main()
