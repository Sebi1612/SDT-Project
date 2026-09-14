import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


REQUIRED_RESULTS = ('clusters.txt', 'prediction.txt', 'dis.txt', 'log.txt')


def parse_prediction(path):
    correct = 0
    total = 0
    per_label = defaultdict(lambda: [0, 0])
    with open(path) as handle:
        for line in handle:
            fields = line.rstrip('\n').split('\t')
            if len(fields) < 2:
                continue
            truth = fields[0]
            closest = fields[1].split(',', 1)[0]
            prediction = closest.split('-', 1)[1]
            total += 1
            per_label[truth][1] += 1
            if prediction == truth:
                correct += 1
                per_label[truth][0] += 1
    return {
        'accuracy': correct / total if total else None,
        'num_test': total,
        'per_label_accuracy': {
            label: values[0] / values[1] if values[1] else None
            for label, values in sorted(per_label.items())
        },
        'per_label_test_count': {
            label: values[1] for label, values in sorted(per_label.items())
        },
    }


def parse_result(result_dir):
    result_dir = Path(result_dir)
    cluster_ids = [
        int(line.strip())
        for line in (result_dir / 'clusters.txt').read_text().splitlines()
        if line.strip()
    ]
    distances = []
    for line in (result_dir / 'dis.txt').read_text().splitlines():
        if ':' in line:
            distances.append(float(line.rsplit(':', 1)[1].strip()))
    log = (result_dir / 'log.txt').read_text()
    solver = (
        'gurobi' if 'Gurobi IS found' in log else
        'hard_svm' if 'Gurobi is NOT found' in log else 'unknown'
    )
    output = {
        'num_clusters': len(set(cluster_ids)),
        'cluster_sizes': dict(Counter(cluster_ids)),
        'solver': solver,
        'minimum_intercluster_distance': min(distances) if distances else None,
        'mean_intercluster_distance': float(np.mean(distances)) if distances else None,
        'maximum_intercluster_distance': max(distances) if distances else None,
    }
    output.update(parse_prediction(result_dir / 'prediction.txt'))
    return output


def write_manifest(path, manifest):
    temporary = str(path) + '.tmp'
    with open(temporary, 'w') as handle:
        json.dump(manifest, handle, indent=2)
    os.replace(temporary, path)


def run_matrix(
    dp_dir,
    pilot_root,
    languages,
    models,
    tasks,
    layers,
    workers,
    timeout,
    required_solver=None,
    manifest_name='directprobe_run_manifest.json',
    quarantine_incomplete=False,
):
    dp_dir = Path(dp_dir).resolve()
    pilot_root = Path(pilot_root).resolve()
    if Path(manifest_name).name != manifest_name:
        raise ValueError('--manifest_name must be a file name, not a path')
    manifest_path = pilot_root / manifest_name
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = {
            'status': 'in_progress',
            'python': sys.executable,
            'workers': workers,
            'timeout_seconds': timeout,
            'required_solver': required_solver,
            'runs': [],
        }
    manifest['required_solver'] = required_solver
    existing = {
        (run['language'], run.get('model', 'codebert'), run['task'], run['layer']): run
        for run in manifest['runs']
        if run.get('status') == 'complete'
        and (not required_solver or run.get('solver') == required_solver)
    }

    environment = os.environ.copy()
    environment['DIRECTPROBE_N_JOBS'] = str(workers)
    environment.setdefault('OMP_NUM_THREADS', '1')
    environment.setdefault('MKL_NUM_THREADS', '1')
    total = len(languages) * len(models) * len(tasks) * len(layers)
    completed = 0
    failures = []
    recovery_run = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())

    for language in languages:
        for model in models:
            for task in tasks:
                for layer in layers:
                    key = (language, model, task, layer)
                    config_path = (
                        pilot_root / 'config_files' / language / task
                        / f'config_{model}_{layer}.ini'
                    )
                    result_dir = (
                        pilot_root / 'results' / language / task / model
                        / str(layer)
                    )
                    results_exist = all(
                        (result_dir / name).exists() for name in REQUIRED_RESULTS
                    )
                    if key not in existing and results_exist:
                        summary = parse_result(result_dir)
                        if required_solver and summary['solver'] != required_solver:
                            raise RuntimeError(
                                f'Expected solver {required_solver}, found '
                                f"{summary['solver']} in {result_dir}"
                            )
                        record = {
                            'status': 'complete',
                            'language': language,
                            'model': model,
                            'task': task,
                            'layer': layer,
                            'duration_seconds': None,
                            'workers': workers,
                            'config': str(config_path),
                            'result_dir': str(result_dir),
                            **summary,
                        }
                        manifest['runs'].append(record)
                        existing[key] = record
                    if key in existing and results_exist:
                        completed += 1
                        print(
                            f'[SKIP {completed}/{total}] '
                            f'{language}/{model}/{task}/L{layer}'
                        )
                        continue
                    if result_dir.exists() and any(result_dir.iterdir()):
                        if not quarantine_incomplete:
                            raise RuntimeError(
                                'Incomplete non-empty result directory: '
                                f'{result_dir}'
                            )
                        relative = result_dir.relative_to(pilot_root)
                        archive_dir = (
                            pilot_root / 'interrupted_results_archive'
                            / recovery_run / relative
                        )
                        if archive_dir.exists():
                            archive_dir = archive_dir.with_name(
                                archive_dir.name + f'_{time.time_ns()}'
                            )
                        archive_dir.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(result_dir), str(archive_dir))
                        recovery = {
                            'language': language,
                            'model': model,
                            'task': task,
                            'layer': layer,
                            'source': str(result_dir),
                            'archive': str(archive_dir),
                            'recovered_utc': time.strftime(
                                '%Y-%m-%dT%H:%M:%SZ', time.gmtime()
                            ),
                        }
                        manifest.setdefault(
                            'quarantined_incomplete_results', []
                        ).append(recovery)
                        write_manifest(manifest_path, manifest)
                        print(
                            '[QUARANTINE] incomplete result moved to '
                            f'{archive_dir}'
                        )

                    started = time.monotonic()
                    try:
                        process = subprocess.run(
                            [
                                sys.executable,
                                'main.py',
                                '--config_file',
                                str(config_path),
                            ],
                            cwd=dp_dir,
                            env=environment,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True,
                            timeout=timeout,
                            check=False,
                        )
                        duration = time.monotonic() - started
                        if process.returncode != 0:
                            raise RuntimeError(
                                f'exit code {process.returncode}: '
                                + process.stdout[-3000:]
                            )
                        missing = [
                            name for name in REQUIRED_RESULTS
                            if not (result_dir / name).exists()
                        ]
                        if missing:
                            raise RuntimeError(f'Missing result files: {missing}')
                        summary = parse_result(result_dir)
                        if required_solver and summary['solver'] != required_solver:
                            raise RuntimeError(
                                f'Expected solver {required_solver}, found '
                                f"{summary['solver']} in {result_dir}"
                            )
                        record = {
                            'status': 'complete',
                            'language': language,
                            'model': model,
                            'task': task,
                            'layer': layer,
                            'duration_seconds': duration,
                            'workers': workers,
                            'config': str(config_path),
                            'result_dir': str(result_dir),
                            **summary,
                        }
                        manifest['runs'] = [
                            run for run in manifest['runs']
                            if (
                                run['language'],
                                run.get('model', 'codebert'),
                                run['task'],
                                run['layer'],
                            ) != key
                        ]
                        manifest['runs'].append(record)
                        completed += 1
                        print(
                            f'[DONE {completed}/{total}] '
                            f'{language}/{model}/{task}/L{layer} '
                            f"acc={summary['accuracy']:.3f} "
                            f"clusters={summary['num_clusters']} "
                            f'time={duration:.1f}s'
                        )
                    except Exception as exc:
                        record = {
                            'status': 'failed',
                            'language': language,
                            'model': model,
                            'task': task,
                            'layer': layer,
                            'reason': f'{type(exc).__name__}: {exc}',
                        }
                        manifest['runs'] = [
                            run for run in manifest['runs']
                            if (
                                run['language'],
                                run.get('model', 'codebert'),
                                run['task'],
                                run['layer'],
                            ) != key
                        ]
                        manifest['runs'].append(record)
                        failures.append(record)
                        print(
                            f'[FAILED] {language}/{model}/{task}/L{layer}: {exc}'
                        )
                    write_manifest(manifest_path, manifest)

    manifest['status'] = 'complete' if not failures else 'complete_with_failures'
    manifest['num_complete'] = sum(
        run['status'] == 'complete' for run in manifest['runs']
    )
    manifest['num_failed'] = sum(
        run['status'] == 'failed' for run in manifest['runs']
    )
    write_manifest(manifest_path, manifest)
    return manifest


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument('--dp_dir', default='DirectProbe')
    cli.add_argument('--pilot_root', default='DirectProbe/pilot_100')
    cli.add_argument(
        '--languages', nargs='+',
        default=['python', 'java', 'go', 'javascript'],
    )
    cli.add_argument('--models', nargs='+', default=['codebert'])
    cli.add_argument(
        '--tasks', nargs='+',
        default=['distance', 'distance_id', 'siblings', 'siblings_id', 'dfg'],
    )
    cli.add_argument('--layers', nargs='+', type=int, default=[0, 5, 9, 12])
    cli.add_argument('--workers', type=int, default=1)
    cli.add_argument('--timeout', type=int, default=600)
    cli.add_argument(
        '--required_solver', choices=['gurobi', 'hard_svm'], default=None,
        help='Fail a run rather than silently accepting a different solver.',
    )
    cli.add_argument(
        '--manifest_name', default='directprobe_run_manifest.json',
        help='Separate manifests allow safe language-level parallel runners.',
    )
    cli.add_argument(
        '--quarantine_incomplete',
        action='store_true',
        help=(
            'Move an interrupted non-empty result directory into a timestamped '
            'archive and rerun that configuration.'
        ),
    )
    args = cli.parse_args()
    manifest = run_matrix(
        args.dp_dir,
        args.pilot_root,
        args.languages,
        args.models,
        args.tasks,
        args.layers,
        args.workers,
        args.timeout,
        args.required_solver,
        args.manifest_name,
        args.quarantine_incomplete,
    )
    print(
        f"[DIRECTPROBE MATRIX COMPLETE] {manifest['num_complete']} complete, "
        f"{manifest['num_failed']} failed"
    )


if __name__ == '__main__':
    main()
