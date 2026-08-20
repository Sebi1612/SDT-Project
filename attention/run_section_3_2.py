"""Run the paper's Section 3.2 attention analysis without the notebook."""

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from collections import Counter

from transformers import RobertaTokenizer


SUPPORTED_LANGUAGES = ('python', 'java', 'go', 'javascript')
CODEBERT_VERSION = 'microsoft/codebert-base'


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(repo_root, path):
    return path if os.path.isabs(path) else os.path.join(repo_root, path)


def validate_codebert_dataset(path, expected_count, tokenizer):
    rows = 0
    partitions = Counter()
    maximum_tokens = 0
    excessive = []
    with open(path) as handle:
        for index, line in enumerate(handle):
            record = json.loads(line)
            code_tokens = record.get('code_tokens')
            if not isinstance(code_tokens, list) or not code_tokens:
                raise ValueError(f'{path} row {index} has no code_tokens list')
            model_tokens = tokenizer.tokenize(' '.join(code_tokens))
            token_count = len(model_tokens)
            maximum_tokens = max(maximum_tokens, token_count)
            if token_count >= 500:
                excessive.append({
                    'row': index,
                    'code_file': record.get('code_file'),
                    'model_tokens': token_count,
                })
            partitions[str(record.get('partition'))] += 1
            rows += 1

    if rows != expected_count:
        raise ValueError(f'{path} contains {rows} rows; expected {expected_count}')
    if excessive:
        preview = ', '.join(
            f"row {item['row']} ({item['model_tokens']} tokens)"
            for item in excessive[:5]
        )
        raise ValueError(
            f'{path} contains {len(excessive)} programs with at least 500 '
            f'CodeBERT tokens: {preview}'
        )
    return {
        'path': os.path.abspath(path),
        'sha256': sha256(path),
        'rows': rows,
        'partitions': dict(partitions),
        'maximum_codebert_tokens': maximum_tokens,
        'length_policy': 'strictly fewer than 500 CodeBERT subtokens',
    }


def run_command(command, repo_root, dry_run):
    print(f"[RUN] {shlex.join(command)}", flush=True)
    if not dry_run:
        subprocess.run(command, cwd=repo_root, check=True)


def write_manifest(path, manifest):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = path + '.tmp'
    with open(temporary, 'w') as handle:
        json.dump(manifest, handle, indent=2)
    os.replace(temporary, path)


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument(
        '--languages',
        nargs='+',
        default=['java', 'go', 'javascript'],
        choices=SUPPORTED_LANGUAGES,
    )
    cli.add_argument(
        '--dataset_dir', default='attention/exp_data/pilot_100'
    )
    cli.add_argument(
        '--dataset_pattern',
        default='{language}.jsonl',
        help=(
            'Filename pattern inside --dataset_dir. The {language} field is '
            'replaced with each requested language.'
        ),
    )
    cli.add_argument('--graph_root', default='graph_info/pilot_100')
    cli.add_argument(
        '--results_root', default='analysis_results/attention_pilot_100'
    )
    cli.add_argument(
        '--python_reference_root', default='attention/graph_comparision'
    )
    cli.add_argument('--expected_count', default=100, type=int)
    cli.add_argument('--device', default='cuda:0')
    cli.add_argument('--seed', default=0, type=int)
    cli.add_argument('--bootstrap_samples', default=1000, type=int)
    cli.add_argument(
        '--ged_mode',
        choices=('legacy', 'fixed'),
        default='legacy',
        help=(
            "Primary GED mode. 'legacy' reproduces the paper and is much "
            "slower; 'fixed' uses the newer scalable fixed-node distance."
        ),
    )
    cli.add_argument(
        '--also_run_fixed_ged',
        action='store_true',
        help=(
            'After the primary legacy GED, also retain a separate fixed-node '
            'distance analysis and summary.'
        ),
    )
    cli.add_argument(
        '--skip_ged',
        action='store_true',
        help=(
            'Run AST/DFG overlap analysis only. GED can be computed later '
            'from the retained graph artifacts.'
        ),
    )
    cli.add_argument('--skip_graph_generation', action='store_true')
    cli.add_argument('--skip_python_reference_comparison', action='store_true')
    cli.add_argument(
        '--allow_partial_coverage',
        action='store_true',
        help='Report graph/DFG failures instead of requiring every row to pass.',
    )
    cli.add_argument('--dry_run', action='store_true')
    args = cli.parse_args()

    if len(args.languages) != len(set(args.languages)):
        raise ValueError('--languages must not contain duplicates')
    if args.expected_count <= 0:
        raise ValueError('--expected_count must be positive')
    if args.bootstrap_samples < 0:
        raise ValueError('--bootstrap_samples cannot be negative')
    if args.also_run_fixed_ged and args.ged_mode != 'legacy':
        raise ValueError(
            '--also_run_fixed_ged is only meaningful with --ged_mode legacy'
        )
    if args.skip_ged and args.also_run_fixed_ged:
        raise ValueError('--skip_ged cannot be combined with --also_run_fixed_ged')
    if (
        'python' in args.languages
        and not args.skip_python_reference_comparison
        and (args.ged_mode != 'legacy' or args.skip_ged)
    ):
        raise ValueError(
            'Python paper-reference comparison requires a legacy GED run; '
            'otherwise pass --skip_python_reference_comparison'
        )

    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    dataset_dir = resolve(repo_root, args.dataset_dir)
    graph_root = resolve(repo_root, args.graph_root)
    results_root = resolve(repo_root, args.results_root)
    python_reference_root = resolve(repo_root, args.python_reference_root)
    manifest_name = (
        'section_3_2_dry_run_manifest.json'
        if args.dry_run
        else 'section_3_2_run_manifest.json'
    )
    manifest_path = os.path.join(results_root, manifest_name)

    tokenizer = RobertaTokenizer.from_pretrained(CODEBERT_VERSION)
    datasets = {}
    for language in args.languages:
        try:
            dataset_name = args.dataset_pattern.format(language=language)
        except (KeyError, IndexError, ValueError) as exc:
            raise ValueError(f'Invalid --dataset_pattern: {exc}') from exc
        dataset_path = os.path.join(dataset_dir, dataset_name)
        datasets[language] = validate_codebert_dataset(
            dataset_path, args.expected_count, tokenizer
        )

    manifest = {
        'status': 'dry_run' if args.dry_run else 'in_progress',
        'paper_section': '3.2',
        'model': 'codebert',
        'model_version': CODEBERT_VERSION,
        'languages': args.languages,
        'expected_count_per_language': args.expected_count,
        'primary_threshold': 0.05,
        'threshold_sweep': [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4],
        'strict_full_coverage': not args.allow_partial_coverage,
        'seed': args.seed,
        'bootstrap_samples': args.bootstrap_samples,
        'ged_status': 'skipped' if args.skip_ged else 'included',
        'ged_mode': None if args.skip_ged else args.ged_mode,
        'ged_method': (
            None
            if args.skip_ged
            else (
                'legacy_networkx_optimize_first_candidate'
                if args.ged_mode == 'legacy'
                else 'exact_fixed_node_edge_edit_distance'
            )
        ),
        'ged_paper_comparable': False if args.skip_ged else args.ged_mode == 'legacy',
        'also_run_fixed_ged': args.also_run_fixed_ged,
        'datasets': datasets,
        'python_reference_root': (
            os.path.abspath(python_reference_root)
            if 'python' in args.languages
            and not args.skip_python_reference_comparison
            else None
        ),
        'commands': [],
    }
    write_manifest(manifest_path, manifest)

    def execute(command):
        manifest['commands'].append(command)
        write_manifest(manifest_path, manifest)
        run_command(command, repo_root, args.dry_run)

    try:
        for language in args.languages:
            code_file = datasets[language]['path']
            language_graph_root = os.path.join(graph_root, language)
            graph_dir = os.path.join(language_graph_root, 'codebert')
            language_results = os.path.join(results_root, language)

            if not args.skip_graph_generation:
                execute([
                    sys.executable,
                    os.path.join(script_dir, 'save_graph_info.py'),
                    '--model', 'codebert',
                    '--code_file', code_file,
                    '--save_dir', language_graph_root,
                    '--lang', language,
                    '--seed', str(args.seed),
                    '--device', args.device,
                ])

            validation_command = [
                sys.executable,
                os.path.join(script_dir, 'validate_graph_run.py'),
                '--graph_dir', graph_dir,
                '--expected_count', str(args.expected_count),
                '--expected_model', 'codebert',
                '--expected_language', language,
                '--require_deterministic',
                '--require_parser_provenance',
            ]
            if not args.allow_partial_coverage:
                validation_command.extend(['--minimum_coverage', '1.0'])
            execute(validation_command)

            execute([
                sys.executable,
                os.path.join(script_dir, 'graph_comp.py'),
                '--graph_loc', graph_dir,
                '--save_dir', language_results,
                '--all_layers',
                '--bootstrap_samples', str(args.bootstrap_samples),
            ])
            execute([
                sys.executable,
                os.path.join(script_dir, 'dfg_comp.py'),
                '--graph_loc', graph_dir,
                '--code_file', code_file,
                '--save_dir', language_results,
                '--lang', language,
                '--all_layers',
                '--bootstrap_samples', str(args.bootstrap_samples),
            ])
            if not args.skip_ged:
                execute([
                    sys.executable,
                    os.path.join(script_dir, 'similarity.py'),
                    '--graphs_dir', graph_dir,
                    '--code_file', code_file,
                    '--save_dir', language_results,
                    '--lang', language,
                    '--all_layers',
                    '--threshold', '0.05',
                    '--bootstrap_samples', str(args.bootstrap_samples),
                    '--distance_mode', args.ged_mode,
                ])
            summary_path = os.path.join(
                language_results, 'section_3_2_summary.json'
            )
            summary_command = [
                sys.executable,
                os.path.join(script_dir, 'summarize_attention_analysis.py'),
                '--results_dir', language_results,
                '--output', summary_path,
                '--model', 'codebert',
                '--lang', language,
                '--threshold', '0.05',
                '--ged_mode', args.ged_mode,
            ]
            if args.skip_ged:
                summary_command.append('--skip_ged')
            if not args.allow_partial_coverage:
                summary_command.extend([
                    '--expected_programs', str(args.expected_count)
                ])
            execute(summary_command)
            if args.also_run_fixed_ged:
                execute([
                    sys.executable,
                    os.path.join(script_dir, 'similarity.py'),
                    '--graphs_dir', graph_dir,
                    '--code_file', code_file,
                    '--save_dir', language_results,
                    '--lang', language,
                    '--all_layers',
                    '--threshold', '0.05',
                    '--bootstrap_samples', str(args.bootstrap_samples),
                    '--distance_mode', 'fixed',
                ])
                fixed_summary_command = [
                    sys.executable,
                    os.path.join(
                        script_dir, 'summarize_attention_analysis.py'
                    ),
                    '--results_dir', language_results,
                    '--output', os.path.join(
                        language_results, 'section_3_2_summary_fixed.json'
                    ),
                    '--model', 'codebert',
                    '--lang', language,
                    '--threshold', '0.05',
                    '--ged_mode', 'fixed',
                ]
                if not args.allow_partial_coverage:
                    fixed_summary_command.extend([
                        '--expected_programs', str(args.expected_count)
                    ])
                execute(fixed_summary_command)
            if (
                language == 'python'
                and not args.skip_python_reference_comparison
            ):
                execute([
                    sys.executable,
                    os.path.join(script_dir, 'compare_python_reference.py'),
                    '--pilot_summary', summary_path,
                    '--reference_root', python_reference_root,
                    '--output', os.path.join(
                        language_results, 'python_reference_comparison.json'
                    ),
                ])
    except Exception as exc:
        manifest['status'] = 'failed'
        manifest['reason'] = f'{type(exc).__name__}: {exc}'
        write_manifest(manifest_path, manifest)
        raise

    if not args.dry_run:
        manifest['status'] = 'complete'
        write_manifest(manifest_path, manifest)
    print(f'[SECTION 3.2 RUN COMPLETE] {manifest_path}')


if __name__ == '__main__':
    main()
