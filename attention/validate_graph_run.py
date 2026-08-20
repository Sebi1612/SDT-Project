import argparse
import json
import os
import pickle

import numpy as np


def validate(
    graph_dir,
    expected_count=None,
    expected_model=None,
    expected_language=None,
    require_deterministic=False,
    require_parser_provenance=False,
    minimum_coverage=None,
):
    manifest_path = os.path.join(graph_dir, 'graph_manifest.json')
    with open(manifest_path) as manifest_file:
        manifest = json.load(manifest_file)

    artifacts = manifest.get('artifacts', [])
    failures = manifest.get('failures', [])
    errors = []
    shapes = set()

    if manifest.get('status') != 'complete':
        errors.append(f"manifest status is {manifest.get('status')!r}, not 'complete'")
    if len(artifacts) != len(set(artifacts)):
        errors.append('manifest contains duplicate artifact names')
    if len(artifacts) + len(failures) != manifest.get('selected_num_codes'):
        errors.append('artifact + failure count does not equal selected_num_codes')
    if expected_count is not None and manifest.get('selected_num_codes') != expected_count:
        errors.append(
            f"selected_num_codes is {manifest.get('selected_num_codes')}, "
            f'expected {expected_count}'
        )
    if expected_model is not None and manifest.get('model') != expected_model:
        errors.append(
            f"manifest model is {manifest.get('model')!r}, "
            f'expected {expected_model!r}'
        )
    if (
        expected_model == 'codebert'
        and manifest.get('model_version') != 'microsoft/codebert-base'
    ):
        errors.append(
            f"CodeBERT model version is {manifest.get('model_version')!r}, "
            "expected 'microsoft/codebert-base'"
        )
    if expected_language is not None and manifest.get('lang') != expected_language:
        errors.append(
            f"manifest language is {manifest.get('lang')!r}, "
            f'expected {expected_language!r}'
        )
    if require_deterministic:
        if manifest.get('inference_mode') is not True:
            errors.append('manifest does not confirm inference_mode=true')
        if manifest.get('random_model') is not False:
            errors.append('manifest does not confirm random_model=false')
        if manifest.get('seed') is None:
            errors.append('manifest does not record a random seed')
        if not manifest.get('model_version'):
            errors.append('manifest does not record a model version')
    if require_parser_provenance:
        for field in (
            'tree_sitter_version',
            'tree_sitter_language_library',
            'tree_sitter_language_library_sha256',
            'grammar_repo',
        ):
            if not manifest.get(field):
                errors.append(f'manifest does not record {field}')

    for artifact_name in artifacts:
        artifact_path = os.path.join(graph_dir, artifact_name)
        if not os.path.isfile(artifact_path):
            errors.append(f'missing artifact: {artifact_name}')
            continue

        try:
            with open(artifact_path, 'rb') as artifact_file:
                artifact = pickle.load(artifact_file)
        except Exception as exc:
            errors.append(f'cannot load {artifact_name}: {type(exc).__name__}: {exc}')
            continue

        code_tokens = artifact.get('code_tokens')
        expected_model_tokens = [token.replace(' ', '') for token in code_tokens]
        if expected_model_tokens != artifact.get('model_tokens'):
            errors.append(f'model token mismatch: {artifact_name}')
        if code_tokens != artifact.get('ast_tokens'):
            errors.append(f'AST token mismatch: {artifact_name}')

        attention = artifact.get('model_graphs')
        ast_graph = artifact.get('ast_graph')
        token_count = len(code_tokens)
        if not isinstance(attention, np.ndarray) or attention.ndim != 4:
            errors.append(f'invalid attention tensor: {artifact_name}')
            continue
        if attention.shape[-2:] != (token_count, token_count):
            errors.append(f'attention/token shape mismatch: {artifact_name}')
        if not np.isfinite(attention).all():
            errors.append(f'non-finite attention value: {artifact_name}')
        if not isinstance(ast_graph, np.ndarray) or ast_graph.shape != (token_count, token_count):
            errors.append(f'AST/token shape mismatch: {artifact_name}')
        elif not np.isin(ast_graph, [0, 1]).all():
            errors.append(f'AST graph is not binary: {artifact_name}')
        shapes.add(tuple(attention.shape[:2]))

    unlisted_pickles = sorted(
        name for name in os.listdir(graph_dir)
        if name.endswith('.pkl') and name not in set(artifacts)
    )
    selected = manifest.get('selected_num_codes', 0)
    coverage = len(artifacts) / selected if selected else 0.0
    if minimum_coverage is not None:
        if not 0 <= minimum_coverage <= 1:
            errors.append('minimum_coverage must be between 0 and 1')
        elif coverage < minimum_coverage:
            errors.append(
                f'graph coverage is {coverage:.2%}, below required '
                f'{minimum_coverage:.2%}'
            )
    report = {
        'graph_dir': os.path.abspath(graph_dir),
        'language': manifest.get('lang'),
        'model': manifest.get('model'),
        'model_version': manifest.get('model_version'),
        'random_model': manifest.get('random_model'),
        'seed': manifest.get('seed'),
        'inference_mode': manifest.get('inference_mode'),
        'torch_version': manifest.get('torch_version'),
        'transformers_version': manifest.get('transformers_version'),
        'tree_sitter_version': manifest.get('tree_sitter_version'),
        'tree_sitter_language_library': manifest.get(
            'tree_sitter_language_library'
        ),
        'tree_sitter_language_library_sha256': manifest.get(
            'tree_sitter_language_library_sha256'
        ),
        'grammar_repo': manifest.get('grammar_repo'),
        'status': manifest.get('status'),
        'selected': selected,
        'artifacts': len(artifacts),
        'failures': len(failures),
        'coverage': coverage,
        'layer_head_shapes': [list(shape) for shape in sorted(shapes)],
        'unlisted_pickle_count': len(unlisted_pickles),
        'errors': errors,
        'valid': not errors,
    }
    print(json.dumps(report, indent=2))
    return not errors


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_dir', required=True)
    parser.add_argument('--expected_count', type=int)
    parser.add_argument('--expected_model')
    parser.add_argument(
        '--expected_language',
        choices=['python', 'java', 'go', 'javascript'],
    )
    parser.add_argument('--require_deterministic', action='store_true')
    parser.add_argument('--require_parser_provenance', action='store_true')
    parser.add_argument('--minimum_coverage', type=float)
    args = parser.parse_args()
    raise SystemExit(0 if validate(
        args.graph_dir,
        args.expected_count,
        args.expected_model,
        args.expected_language,
        args.require_deterministic,
        args.require_parser_provenance,
        args.minimum_coverage,
    ) else 1)
