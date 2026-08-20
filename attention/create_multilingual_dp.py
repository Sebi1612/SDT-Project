import argparse
import configparser
import json
import os
import pickle
from collections import Counter, defaultdict

import numpy as np
from sklearn.model_selection import train_test_split

from dfg_comp import build_parser, get_dfg_adj


KEYWORDS = {
    'python': {
        'def', 'for', 'if', 'None', 'else', 'False', 'True', 'or', 'and',
        'return', 'not', 'elif', 'with', 'try', 'raise', 'except', 'break',
        'while', 'assert', 'print', 'continue', 'class',
    },
    'java': {
        'class', 'interface', 'enum', 'if', 'else', 'switch', 'case', 'for',
        'while', 'do', 'true', 'false', 'null', 'return', 'break', 'continue',
        'throw', 'try', 'catch', 'finally', 'new',
    },
    'go': {
        'func', 'type', 'var', 'const', 'if', 'else', 'switch', 'case',
        'select', 'for', 'range', 'true', 'false', 'nil', 'return', 'break',
        'continue', 'goto', 'fallthrough', 'defer', 'go',
    },
    'javascript': {
        'function', 'class', 'let', 'const', 'var', 'if', 'else', 'switch',
        'case', 'for', 'while', 'do', 'true', 'false', 'null', 'undefined',
        'return', 'break', 'continue', 'throw', 'try', 'catch', 'finally',
        'new',
    },
}

TASK_LABELS = {
    'distance': ['2', '3', '4', '5', '6'],
    'distance_id': ['2', '3', '4', '5', '6'],
    'siblings': ['NotSibling', 'Sibling'],
    'siblings_id': ['NotSibling', 'Sibling'],
    'dfg': ['NoEdge', 'ComesFrom', 'ComputedFrom'],
}


def is_identifier(info):
    return 'identifier' in info['type']


def load_embedding_index(embedding_dir):
    manifest_path = os.path.join(embedding_dir, 'embedding_manifest.json')
    with open(manifest_path) as handle:
        manifest = json.load(handle)
    if manifest.get('status') != 'complete':
        raise ValueError('Embedding manifest is not complete')
    return manifest_path, manifest


def pair_record(artifact, row, column, label, token_info):
    return {
        'artifact': artifact,
        'row': int(row),
        'column': int(column),
        'label': str(label),
        'first_token': token_info[row]['token'],
        'first_type': token_info[row]['type'],
        'second_token': token_info[column]['token'],
        'second_type': token_info[column]['type'],
    }


def distance_candidates(artifact, data, identifier_only):
    token_info = data['code_token_info']
    tree_dist = data['tree_dist']
    language = data['language']
    records = []
    for row, first in enumerate(token_info):
        if first['token'] not in KEYWORDS[language]:
            continue
        for column, second in enumerate(token_info):
            distance = int(tree_dist[row, column])
            if distance not in {2, 3, 4, 5, 6}:
                continue
            if identifier_only and not is_identifier(second):
                continue
            records.append(
                pair_record(artifact, row, column, distance, token_info)
            )
    return records


def sibling_candidates(artifact, data, ast_graph, identifier_only, rng):
    token_info = data['code_token_info']
    language = data['language']
    ast_graph = np.asarray(ast_graph, dtype=bool)
    records = []
    for row, first in enumerate(token_info):
        if first['token'] not in KEYWORDS[language]:
            continue
        valid_columns = [
            column
            for column, second in enumerate(token_info)
            if column != row and (not identifier_only or is_identifier(second))
        ]
        positive = [column for column in valid_columns if ast_graph[row, column]]
        negative = [column for column in valid_columns if not ast_graph[row, column]]
        if len(negative) > len(positive):
            negative = rng.choice(
                negative, size=len(positive), replace=False
            ).tolist()
        for column in positive:
            records.append(
                pair_record(artifact, row, column, 'Sibling', token_info)
            )
        for column in negative:
            records.append(
                pair_record(artifact, row, column, 'NotSibling', token_info)
            )
    return records


def dfg_candidates(artifact, data, graph, parser, rng):
    token_info = data['code_token_info']
    identifier = np.asarray([is_identifier(info) for info in token_info])
    typed, tokens = get_dfg_adj(
        graph['code'],
        parser,
        lang=data['language'],
        expected_tokens=data['code_tokens'],
        typed=True,
    )
    if tokens != data['code_tokens']:
        raise ValueError('Typed DFG tokens do not match hidden-state tokens')

    records = []
    for row in np.flatnonzero(identifier):
        comes = [
            int(column)
            for column in np.flatnonzero(typed[row] == 1)
            if identifier[column]
        ]
        computed = [
            int(column)
            for column in np.flatnonzero(typed[row] == -1)
            if identifier[column]
        ]
        # Match the paper's Python construction: the first token is an
        # identifier, positive partners are identifiers by DFG definition,
        # while NoEdge partners may be any token type.
        unrelated = [
            column for column in range(len(token_info))
            if column != row and typed[row, column] == 0
        ]
        negative_count = int(np.ceil(max(len(comes), len(computed)) / 2))
        if len(unrelated) > negative_count:
            unrelated = rng.choice(
                unrelated, size=negative_count, replace=False
            ).tolist()
        for column in comes:
            records.append(
                pair_record(artifact, row, column, 'ComesFrom', token_info)
            )
        for column in computed:
            records.append(
                pair_record(artifact, row, column, 'ComputedFrom', token_info)
            )
        for column in unrelated:
            records.append(
                pair_record(artifact, row, column, 'NoEdge', token_info)
            )
    return records


def deduplicate(records):
    unique = {}
    for record in records:
        key = (
            record['artifact'], record['row'], record['column'], record['label']
        )
        unique[key] = record
    return list(unique.values())


def balanced_sample(records, labels, target_per_label, rng):
    grouped = defaultdict(list)
    for record in deduplicate(records):
        grouped[record['label']].append(record)
    missing = [label for label in labels if not grouped[label]]
    if missing:
        raise ValueError(f'No candidates for labels: {missing}')
    per_label = min(target_per_label, *(len(grouped[label]) for label in labels))
    selected = []
    for label in labels:
        candidates = sorted(
            grouped[label],
            key=lambda record: (
                record['artifact'], record['row'], record['column']
            ),
        )
        indices = rng.choice(len(candidates), size=per_label, replace=False)
        selected.extend(candidates[index] for index in sorted(indices))
    selected.sort(key=lambda record: (record['artifact'], record['row'], record['column']))
    return selected, {label: len(grouped[label]) for label in labels}, per_label


def seeded_program_subset(artifacts, maximum, rng):
    artifacts = list(artifacts)
    if maximum is None or maximum >= len(artifacts):
        return artifacts
    if maximum <= 0:
        raise ValueError('--max_programs must be positive')
    indices = np.sort(rng.choice(len(artifacts), size=maximum, replace=False))
    return [artifacts[index] for index in indices]


def stratified_indices(records, seed):
    indices = np.arange(len(records))
    labels = [record['label'] for record in records]
    train, test = train_test_split(
        indices,
        test_size=0.2,
        random_state=seed,
        shuffle=True,
        stratify=labels,
    )
    return np.sort(train), np.sort(test)


def entity_text(record):
    return (
        f"({record['first_type']},{record['second_type']}):"
        f"{record['first_token']}:{record['second_token']}"
    )


def write_lines(path, lines):
    with open(path, 'w') as handle:
        handle.write('\n'.join(lines))


def create_config(config_path, output_dir, task_data_dir, layer):
    config = configparser.ConfigParser()
    config['run'] = {'output_path': os.path.abspath(output_dir)}
    config['data'] = {
        'common': os.path.abspath(task_data_dir),
        'label_set_path': '${common}/labels/tags.txt',
        'entities_path': '${common}/entities/train.txt',
        'embeddings_path': f'${{common}}/embeddings/layers/train/{layer}.txt',
        'test_entities_path': '${common}/entities/test.txt',
        'test_embeddings_path': f'${{common}}/embeddings/layers/test/{layer}.txt',
    }
    config['clustering'] = {
        'enable_cuda': 'False',
        'rate': '0.01',
        'mode': 'probing',
        'probing_cluster_path': os.path.abspath(output_dir),
    }
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as handle:
        config.write(handle)


def generate_task(
    task,
    lang,
    embedding_dir,
    graph_dir,
    output_root,
    layers,
    target_per_label,
    seed,
    max_programs,
    require_target=False,
    model=None,
):
    manifest_path, embedding_manifest = load_embedding_index(embedding_dir)
    if embedding_manifest.get('language') != lang:
        raise ValueError('Embedding language does not match --lang')
    model_name = embedding_manifest.get('model')
    if not model_name:
        raise ValueError('Embedding manifest does not record a model')
    if model is not None and model_name != model:
        raise ValueError(
            f'Embedding model is {model_name!r}, expected {model!r}'
        )
    with open(os.path.join(graph_dir, 'graph_manifest.json')) as handle:
        graph_manifest = json.load(handle)
    if graph_manifest.get('model') != model_name:
        raise ValueError('Graph and embedding manifests use different models')
    graph_artifacts = set(graph_manifest['artifacts'])
    common_artifacts = [
        name for name in embedding_manifest['artifacts'] if name in graph_artifacts
    ]
    if not common_artifacts:
        raise ValueError('Graph and embedding manifests have no common artifacts')
    with open(os.path.join(embedding_dir, common_artifacts[0]), 'rb') as handle:
        hidden_state_count = pickle.load(handle)['hidden_repr'].shape[0]
    invalid_layers = [
        layer for layer in layers
        if layer < 0 or layer >= hidden_state_count
    ]
    if invalid_layers:
        raise ValueError(
            f'Hidden-state indices {invalid_layers} are invalid for '
            f'{model_name}; available indices are 0..{hidden_state_count - 1}'
        )
    rng = np.random.default_rng(seed)
    artifacts = seeded_program_subset(common_artifacts, max_programs, rng)
    parser = build_parser(lang) if task == 'dfg' else None
    candidates = []
    failures = []

    for artifact in artifacts:
        try:
            with open(os.path.join(embedding_dir, artifact), 'rb') as handle:
                data = pickle.load(handle)
            with open(os.path.join(graph_dir, artifact), 'rb') as handle:
                graph = pickle.load(handle)
            if task == 'distance':
                candidates.extend(distance_candidates(artifact, data, False))
            elif task == 'distance_id':
                candidates.extend(distance_candidates(artifact, data, True))
            elif task == 'siblings':
                candidates.extend(
                    sibling_candidates(
                        artifact, data, graph['ast_graph'], False, rng
                    )
                )
            elif task == 'siblings_id':
                candidates.extend(
                    sibling_candidates(
                        artifact, data, graph['ast_graph'], True, rng
                    )
                )
            elif task == 'dfg':
                candidates.extend(
                    dfg_candidates(artifact, data, graph, parser, rng)
                )
            else:
                raise ValueError(f'Unknown task: {task}')
        except Exception as exc:
            failures.append({
                'artifact': artifact,
                'reason': f'{type(exc).__name__}: {exc}',
            })

    labels = TASK_LABELS[task]
    selected, candidate_counts, per_label = balanced_sample(
        candidates, labels, target_per_label, rng
    )
    if require_target and per_label != target_per_label:
        raise ValueError(
            f'Target {target_per_label}/label was not met with '
            f'{len(artifacts)} programs; candidate counts: {candidate_counts}'
        )
    train_indices, test_indices = stratified_indices(selected, seed)
    for index, record in enumerate(selected):
        record['split'] = 'train' if index in set(train_indices) else 'test'

    task_data_dir = os.path.join(output_root, 'data', lang, task, model_name)
    entity_dir = os.path.join(task_data_dir, 'entities')
    label_dir = os.path.join(task_data_dir, 'labels')
    train_embedding_dir = os.path.join(
        task_data_dir, 'embeddings', 'layers', 'train'
    )
    test_embedding_dir = os.path.join(
        task_data_dir, 'embeddings', 'layers', 'test'
    )
    for directory in (
        entity_dir, label_dir, train_embedding_dir, test_embedding_dir
    ):
        os.makedirs(directory, exist_ok=True)

    write_lines(
        os.path.join(entity_dir, 'train.txt'),
        [f'{entity_text(selected[i])}\t{selected[i]["label"]}' for i in train_indices],
    )
    write_lines(
        os.path.join(entity_dir, 'test.txt'),
        [f'{entity_text(selected[i])}\t{selected[i]["label"]}' for i in test_indices],
    )
    write_lines(os.path.join(label_dir, 'tags.txt'), labels)

    required_artifacts = sorted({record['artifact'] for record in selected})
    embeddings = {}
    for artifact in required_artifacts:
        with open(os.path.join(embedding_dir, artifact), 'rb') as handle:
            embeddings[artifact] = pickle.load(handle)['hidden_repr']

    is_difference = task in {'distance', 'distance_id'}
    for layer in layers:
        values = []
        for record in selected:
            hidden = embeddings[record['artifact']][layer]
            first = hidden[record['row']]
            second = hidden[record['column']]
            values.append(
                first - second if is_difference else np.concatenate((first, second))
            )
        values = np.asarray(values, dtype=np.float32)
        np.savetxt(
            os.path.join(train_embedding_dir, f'{layer}.txt'),
            values[train_indices],
            fmt='%.8g',
        )
        np.savetxt(
            os.path.join(test_embedding_dir, f'{layer}.txt'),
            values[test_indices],
            fmt='%.8g',
        )

        config_path = os.path.join(
            output_root,
            'config_files',
            lang,
            task,
            f'config_{model_name}_{layer}.ini',
        )
        result_dir = os.path.join(
            output_root, 'results', lang, task, model_name, str(layer)
        )
        create_config(config_path, result_dir, task_data_dir, layer)

    manifest = {
        'status': 'complete',
        'task': task,
        'language': lang,
        'model': model_name,
        'layers': layers,
        'seed': seed,
        'representation': 'difference' if is_difference else 'concatenation',
        'labels': labels,
        'target_per_label': target_per_label,
        'selected_per_label': per_label,
        'candidate_counts': candidate_counts,
        'num_programs_considered': len(artifacts),
        'num_common_programs': len(common_artifacts),
        'selected_program_artifacts': artifacts,
        'num_program_failures': len(failures),
        'program_failures': failures,
        'num_examples': len(selected),
        'num_train': len(train_indices),
        'num_test': len(test_indices),
        'train_label_counts': dict(Counter(selected[i]['label'] for i in train_indices)),
        'test_label_counts': dict(Counter(selected[i]['label'] for i in test_indices)),
        'embedding_manifest': os.path.abspath(manifest_path),
        'graph_manifest': os.path.abspath(
            os.path.join(graph_dir, 'graph_manifest.json')
        ),
        'pairs': selected,
    }
    with open(os.path.join(task_data_dir, 'dataset_manifest.json'), 'w') as handle:
        json.dump(manifest, handle, indent=2)
    print(
        f'[DIRECTPROBE DATASET] {lang}/{task}: {len(selected)} examples '
        f'({per_label}/label), {len(failures)} failures'
    )
    return manifest


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument(
        '--task',
        required=True,
        choices=['distance', 'distance_id', 'siblings', 'siblings_id', 'dfg'],
    )
    cli.add_argument(
        '--lang',
        required=True,
        choices=['python', 'java', 'go', 'javascript'],
    )
    cli.add_argument('--embedding_dir', required=True)
    cli.add_argument('--graph_dir', required=True)
    cli.add_argument('--output_root', default='DirectProbe/pilot_100')
    cli.add_argument('--layers', nargs='+', type=int, default=[0, 5, 9, 12])
    cli.add_argument('--target_per_label', type=int, default=50)
    cli.add_argument('--max_programs', type=int, default=100)
    cli.add_argument(
        '--require_target', action='store_true',
        help='Fail instead of silently reducing the balanced per-label count.',
    )
    cli.add_argument('--seed', type=int, default=0)
    cli.add_argument(
        '--model',
        help='Optional expected model name; otherwise read from the manifests.',
    )
    args = cli.parse_args()
    generate_task(
        args.task,
        args.lang,
        args.embedding_dir,
        args.graph_dir,
        args.output_root,
        args.layers,
        args.target_per_label,
        args.seed,
        args.max_programs,
        args.require_target,
        args.model,
    )


if __name__ == '__main__':
    main()
