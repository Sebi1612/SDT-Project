import argparse
import json
import os
import pickle

import numpy as np


def validate(graph_dir, expected_count=None):
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
    report = {
        'graph_dir': os.path.abspath(graph_dir),
        'language': manifest.get('lang'),
        'model': manifest.get('model'),
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
    args = parser.parse_args()
    raise SystemExit(0 if validate(args.graph_dir, args.expected_count) else 1)
