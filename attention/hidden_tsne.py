import argparse
import csv
import inspect
import json
import os
import pickle
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score


LANGUAGE_TOKENS = {
    'python': {
        'declaration': {'def', 'class'},
        'control_flow': {'if', 'elif', 'else', 'for', 'while', 'with'},
        'return_exception': {'return', 'raise', 'try', 'except', 'finally'},
        'literal': {'True', 'False', 'None'},
    },
    'java': {
        'declaration': {'class', 'interface', 'enum', 'record', 'new'},
        'control_flow': {'if', 'else', 'for', 'while', 'do', 'switch', 'case'},
        'return_exception': {'return', 'throw', 'try', 'catch', 'finally'},
        'literal': {'true', 'false', 'null'},
    },
    'go': {
        'declaration': {'func', 'type', 'var', 'const', 'struct', 'interface'},
        'control_flow': {'if', 'else', 'for', 'range', 'switch', 'case', 'select'},
        'return_exception': {'return', 'defer', 'go', 'break', 'continue', 'fallthrough'},
        'literal': {'true', 'false', 'nil'},
    },
    'javascript': {
        'declaration': {'function', 'class', 'let', 'const', 'var', 'new'},
        'control_flow': {'if', 'else', 'for', 'while', 'do', 'switch', 'case'},
        'return_exception': {'return', 'throw', 'try', 'catch', 'finally'},
        'literal': {'true', 'false', 'null', 'undefined'},
    },
}

# The Python entries reproduce the token types used for Figure 10 in the
# paper.  The other lists are language-specific counterparts: identifiers,
# common declaration/control-flow keywords, delimiters, and operators.  Match
# against tree-sitter node types first because literals such as strings keep
# their full source text in ``token`` but have the stable node type ``\"``.
TSNE_TOKEN_TYPES = {
    'python': {
        'def', 'identifier', '(', ')', ':', 'if', 'else', ',', 'for',
        'while', 'elif', '.', 'or', 'and', '<', '>', '==', '!=', '"',
        '-', '+',
    },
    'java': {
        'identifier', '(', ')', '{', '}', ',', '.', ';', 'return', 'if',
        'else', 'for', 'while', 'new', '=', '==', '!=', '<', '>', '+', '"',
    },
    'go': {
        'func', 'identifier', '(', ')', '{', '}', ',', '.', 'return', 'if',
        'else', 'for', 'case', ':=', '=', '==', '!=', '<', '>', '+', '"',
    },
    'javascript': {
        'function', 'identifier', '(', ')', '{', '}', ',', '.', 'return',
        'if', 'else', 'for', 'while', '=', '===', '!=', '&&', '||', '<',
        '>', '+', '"',
    },
}

ASSIGNMENT = {'=', '+=', '-=', '*=', '/=', '%=', ':=', '&&=', '||=', '??=', '**='}
LOGICAL_COMPARISON = {
    'and', 'or', 'not', '&&', '||', '!', '==', '!=', '===', '!==',
    '<', '>', '<=', '>=', 'is', 'in', 'instanceof', '??',
}
DELIMITERS = {'(', ')', '[', ']', '{', '}', ',', ':', ';', '.'}


def token_role(token, node_type, lang):
    if 'identifier' in node_type:
        return 'identifier'
    for role, values in LANGUAGE_TOKENS[lang].items():
        if token in values:
            return role
    if token in ASSIGNMENT:
        return 'assignment_operator'
    if token in LOGICAL_COMPARISON:
        return 'logical_comparison'
    if token in DELIMITERS:
        return 'delimiter'
    return None


def selected_token_type(token, node_type, lang):
    """Return the paper-style token-type label, or None if it is not selected."""
    if 'identifier' in node_type:
        return 'identifier'
    selected = TSNE_TOKEN_TYPES[lang]
    if node_type in selected:
        return node_type
    if token in selected:
        return token
    return None


def seeded_subset(values, maximum, seed):
    """Select a deterministic random subset while preserving manifest order."""
    values = list(values)
    if maximum is None or maximum >= len(values):
        return values
    if maximum <= 0:
        raise ValueError('Subset size must be positive or None')
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(values), size=maximum, replace=False))
    return [values[index] for index in indices]


def select_program_subset(artifacts, token_lengths, maximum, strategy, seed):
    """Select the paper-sized program subset using an explicit strategy."""
    artifacts = list(artifacts)
    if strategy == 'shortest':
        ordered = sorted(
            artifacts, key=lambda artifact: (token_lengths[artifact], artifact)
        )
        return ordered if maximum is None else ordered[:maximum]
    if strategy == 'random':
        return seeded_subset(artifacts, maximum, seed)
    raise ValueError(f'Unknown program selection strategy: {strategy}')


def load_embedding_manifest(embedding_dir, lang):
    path = os.path.join(embedding_dir, 'embedding_manifest.json')
    with open(path) as handle:
        manifest = json.load(handle)
    if manifest.get('status') != 'complete':
        raise ValueError('Embedding manifest is not complete')
    if manifest.get('language') != lang:
        raise ValueError('Embedding manifest language does not match --lang')
    return path, manifest


def resolve_layers(embedding_dir, manifest, requested_layers):
    """Resolve model-aware hidden-state indices (embedding through last layer)."""
    artifacts = manifest.get('artifacts', [])
    if not artifacts:
        raise ValueError('Embedding manifest has no artifacts')
    with open(os.path.join(embedding_dir, artifacts[0]), 'rb') as handle:
        hidden = np.asarray(pickle.load(handle)['hidden_repr'])
    if hidden.ndim != 3:
        raise ValueError(f'Expected 3-D hidden representations, got {hidden.shape}')
    state_count = hidden.shape[0]
    if requested_layers is None:
        last = state_count - 1
        layers = sorted({0, last // 2, last})
    else:
        layers = list(requested_layers)
    invalid = [layer for layer in layers if layer < 0 or layer >= state_count]
    if invalid:
        raise ValueError(
            f'Hidden-state indices {invalid} are invalid for model '
            f"{manifest.get('model')!r}, which stores 0..{state_count - 1}"
        )
    return layers, state_count


def load_artifact_token_lengths(embedding_dir, manifest):
    """Load aligned token counts without unpickling every hidden-state tensor."""
    graph_manifest_path = manifest.get('graph_manifest')
    if graph_manifest_path and os.path.exists(graph_manifest_path):
        with open(graph_manifest_path) as handle:
            graph_manifest = json.load(handle)
        code_file = graph_manifest.get('code_file')
        if code_file and os.path.exists(code_file):
            with open(code_file) as handle:
                lengths = [len(json.loads(line)['code_tokens']) for line in handle]
            artifact_lengths = {}
            for artifact in manifest['artifacts']:
                prefix = artifact.split('_', 1)[0]
                if not prefix.isdigit() or int(prefix) >= len(lengths):
                    break
                artifact_lengths[artifact] = lengths[int(prefix)]
            else:
                return artifact_lengths, os.path.abspath(code_file)

    # Compatibility fallback for older manifests whose artifact names do not
    # carry source indices.
    artifact_lengths = {}
    for artifact in manifest['artifacts']:
        with open(os.path.join(embedding_dir, artifact), 'rb') as handle:
            artifact_lengths[artifact] = len(pickle.load(handle)['code_tokens'])
    return artifact_lengths, 'embedding_artifacts'


def balanced_indices(labels, maximum, seed):
    if maximum is None or maximum >= len(labels):
        return np.arange(len(labels), dtype=int)
    if maximum <= 0:
        raise ValueError('Maximum number of points must be positive or None')
    rng = np.random.default_rng(seed)
    groups = defaultdict(list)
    for index, label in enumerate(labels):
        groups[label].append(index)
    quota = max(1, maximum // len(groups))
    selected = []
    for label in sorted(groups):
        values = np.asarray(groups[label])
        count = min(quota, len(values))
        selected.extend(rng.choice(values, size=count, replace=False).tolist())
    if len(selected) < maximum:
        remaining = np.setdiff1d(np.arange(len(labels)), np.asarray(selected))
        count = min(maximum - len(selected), len(remaining))
        if count:
            selected.extend(rng.choice(remaining, size=count, replace=False).tolist())
    return np.asarray(sorted(selected), dtype=int)


def run_tsne(values, perplexity, seed, metric='euclidean', iterations=50000):
    if len(values) <= perplexity:
        raise ValueError(
            f'Perplexity {perplexity} requires more than {perplexity} points'
        )
    arguments = dict(
        n_components=2,
        perplexity=perplexity,
        metric=metric,
        init='random' if metric == 'precomputed' else 'pca',
        learning_rate='auto',
        n_iter_without_progress=300,
        random_state=seed,
    )
    iteration_argument = (
        'max_iter' if 'max_iter' in inspect.signature(TSNE).parameters else 'n_iter'
    )
    arguments[iteration_argument] = iterations
    return TSNE(**arguments).fit_transform(values)


def save_coordinates(path, coordinates, labels, tokens):
    with open(path, 'w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['x', 'y', 'role', 'token'])
        for point, label, token in zip(coordinates, labels, tokens):
            writer.writerow([float(point[0]), float(point[1]), label, token])


def scatter(axis, coordinates, labels, title):
    labels = np.asarray(labels)
    colour_map = plt.get_cmap('tab10')
    for index, label in enumerate(sorted(set(labels))):
        mask = labels == label
        axis.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            s=9,
            alpha=0.72,
            label=label,
            color=colour_map(index % 10),
        )
    axis.set_title(title)
    axis.set_xticks([])
    axis.set_yticks([])


def token_type_analysis(
    embedding_dir,
    output_dir,
    lang,
    layers,
    perplexities,
    min_tokens,
    max_programs,
    program_selection,
    max_points,
    seed,
    iterations,
):
    manifest_path, manifest = load_embedding_manifest(embedding_dir, lang)
    layer_values = {layer: [] for layer in layers}
    labels = []
    tokens = []
    eligible_all = []
    raw_counts = Counter()

    token_lengths, eligibility_source = load_artifact_token_lengths(
        embedding_dir, manifest
    )
    eligible_all = [
        artifact_name for artifact_name in manifest['artifacts']
        if token_lengths[artifact_name] >= min_tokens
    ]

    eligible = select_program_subset(
        eligible_all, token_lengths, max_programs, program_selection, seed
    )
    for artifact_name in eligible:
        with open(os.path.join(embedding_dir, artifact_name), 'rb') as handle:
            data = pickle.load(handle)
        for index, info in enumerate(data['code_token_info']):
            token_type = selected_token_type(info['token'], info['type'], lang)
            if token_type is None:
                continue
            raw_counts[token_type] += 1
            labels.append(token_type)
            tokens.append(info['token'])
            for layer in layers:
                layer_values[layer].append(data['hidden_repr'][layer, index])

    if len(set(labels)) < 2:
        raise ValueError('Fewer than two token roles are available for t-SNE')
    indices = balanced_indices(labels, max_points, seed)
    selected_labels = np.asarray(labels)[indices]
    selected_tokens = np.asarray(tokens)[indices]
    selected_counts = Counter(selected_labels)

    os.makedirs(output_dir, exist_ok=True)
    scores = {}
    generated = []
    for layer in layers:
        values = np.asarray(layer_values[layer], dtype=np.float32)[indices]
        silhouette_sample_size = min(2000, len(values))
        scores[str(layer)] = {
            'silhouette_cosine': float(
                silhouette_score(
                    values,
                    selected_labels,
                    metric='cosine',
                    sample_size=silhouette_sample_size,
                    random_state=seed,
                )
            ),
            'silhouette_sample_size': silhouette_sample_size,
        }
        for perplexity in perplexities:
            coordinates = run_tsne(
                values, perplexity, seed, iterations=iterations
            )
            stem = f'token_types_layer_{layer}_perplexity_{perplexity}'
            csv_path = os.path.join(output_dir, stem + '.csv')
            png_path = os.path.join(output_dir, stem + '.png')
            save_coordinates(
                csv_path, coordinates, selected_labels, selected_tokens
            )
            fig, axis = plt.subplots(figsize=(7, 6))
            scatter(
                axis,
                coordinates,
                selected_labels,
                f'{lang} {manifest["model"]} layer {layer}, '
                f'perplexity {perplexity}',
            )
            axis.legend(fontsize=7, markerscale=2, bbox_to_anchor=(1.02, 1))
            fig.tight_layout()
            fig.savefig(png_path, dpi=200, bbox_inches='tight')
            plt.close(fig)
            generated.extend([os.path.basename(csv_path), os.path.basename(png_path)])

    metadata = {
        'analysis': 'token_type_tsne',
        'language': lang,
        'model': manifest['model'],
        'seed': seed,
        'layers': layers,
        'perplexities': perplexities,
        'iterations': iterations,
        'minimum_program_tokens': min_tokens,
        'eligibility_source': eligibility_source,
        'maximum_programs': max_programs,
        'program_selection': program_selection,
        'num_embedding_artifacts': len(manifest['artifacts']),
        'num_eligible_programs_total': len(eligible_all),
        'num_eligible_programs': len(eligible),
        'eligible_artifacts': eligible,
        'num_candidate_points': len(labels),
        'num_selected_points': len(indices),
        'role_counts_before_balancing': dict(raw_counts),
        'role_counts_after_balancing': dict(selected_counts),
        'scores': scores,
        'embedding_manifest': os.path.abspath(manifest_path),
        'generated_files': generated,
    }
    with open(os.path.join(output_dir, 'token_type_tsne_manifest.json'), 'w') as handle:
        json.dump(metadata, handle, indent=2)
    return metadata


def distance_analysis(
    embedding_dir,
    output_dir,
    lang,
    layers,
    perplexities,
    min_tokens,
    seed,
    iterations,
):
    manifest_path, manifest = load_embedding_manifest(embedding_dir, lang)
    token_lengths, eligibility_source = load_artifact_token_lengths(
        embedding_dir, manifest
    )
    candidates = [
        (token_lengths[artifact_name], artifact_name)
        for artifact_name in manifest['artifacts']
        if token_lengths[artifact_name] >= min_tokens
    ]
    if not candidates:
        raise ValueError('No program meets the distance t-SNE length requirement')

    # The paper uses randomly selected code samples.  Make that selection
    # reproducible and record it instead of relying on visual hand-picking.
    rng = np.random.default_rng(seed)
    length, artifact_name = candidates[int(rng.integers(len(candidates)))]
    with open(os.path.join(embedding_dir, artifact_name), 'rb') as handle:
        data = pickle.load(handle)
    labels = np.asarray([
        token_role(info['token'], info['type'], lang) or 'other'
        for info in data['code_token_info']
    ])
    tokens = np.asarray(data['code_tokens'])
    tree_dist = np.asarray(data['tree_dist'], dtype=np.float64)
    generated = []
    correlations = {}
    os.makedirs(output_dir, exist_ok=True)

    for layer in layers:
        hidden = np.asarray(data['hidden_repr'][layer], dtype=np.float64)
        hidden_dist = squareform(pdist(hidden, metric='euclidean'))
        upper = np.triu_indices(length, k=1)
        correlation = spearmanr(tree_dist[upper], hidden_dist[upper])
        correlations[str(layer)] = {
            'spearman_r': float(correlation.statistic),
            'pvalue': float(correlation.pvalue),
        }
        for perplexity in perplexities:
            ast_coordinates = run_tsne(
                tree_dist,
                perplexity,
                seed,
                metric='precomputed',
                iterations=iterations,
            )
            hidden_coordinates = run_tsne(
                hidden_dist,
                perplexity,
                seed,
                metric='precomputed',
                iterations=iterations,
            )
            stem = f'distance_layer_{layer}_perplexity_{perplexity}'
            save_coordinates(
                os.path.join(output_dir, stem + '_ast.csv'),
                ast_coordinates,
                labels,
                tokens,
            )
            save_coordinates(
                os.path.join(output_dir, stem + '_hidden.csv'),
                hidden_coordinates,
                labels,
                tokens,
            )
            fig, axes = plt.subplots(1, 2, figsize=(13, 5))
            scatter(axes[0], ast_coordinates, labels, 'AST tree distance')
            scatter(axes[1], hidden_coordinates, labels, f'Hidden layer {layer}')
            handles, legend_labels = axes[1].get_legend_handles_labels()
            fig.legend(handles, legend_labels, loc='center right', fontsize=7)
            fig.suptitle(
                f'{lang} {manifest["model"]}, perplexity {perplexity}; '
                f'Spearman r={correlations[str(layer)]["spearman_r"]:.3f}'
            )
            fig.tight_layout(rect=(0, 0, 0.9, 0.95))
            png_path = os.path.join(output_dir, stem + '.png')
            fig.savefig(png_path, dpi=200, bbox_inches='tight')
            plt.close(fig)
            generated.extend([
                stem + '_ast.csv', stem + '_hidden.csv', os.path.basename(png_path)
            ])

    metadata = {
        'analysis': 'ast_vs_hidden_distance_tsne',
        'language': lang,
        'model': manifest['model'],
        'seed': seed,
        'layers': layers,
        'perplexities': perplexities,
        'iterations': iterations,
        'selection_rule': 'seeded random artifact with at least minimum_program_tokens',
        'minimum_program_tokens': min_tokens,
        'eligibility_source': eligibility_source,
        'num_eligible_programs': len(candidates),
        'selected_artifact': artifact_name,
        'num_tokens': length,
        'spearman_ast_vs_hidden_distance': correlations,
        'embedding_manifest': os.path.abspath(manifest_path),
        'generated_files': generated,
    }
    with open(os.path.join(output_dir, 'distance_tsne_manifest.json'), 'w') as handle:
        json.dump(metadata, handle, indent=2)
    return metadata


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument('--embedding_dir', required=True)
    cli.add_argument('--save_dir', required=True)
    cli.add_argument(
        '--lang', required=True, choices=['python', 'java', 'go', 'javascript']
    )
    cli.add_argument('--layers', nargs='+', type=int)
    cli.add_argument(
        '--token_perplexities', nargs='+', type=int, default=[5, 10, 20, 30, 40, 50]
    )
    cli.add_argument('--distance_perplexities', nargs='+', type=int, default=[5, 10])
    cli.add_argument('--min_tokens', type=int, default=100)
    cli.add_argument('--max_programs', type=int, default=100)
    cli.add_argument(
        '--program_selection', choices=['shortest', 'random'], default='shortest',
        help='The original Python notebook intended to retain the 100 shortest eligible programs.',
    )
    cli.add_argument(
        '--max_points', type=int, default=None,
        help='Optional balanced point cap; by default all selected token types are used.',
    )
    cli.add_argument('--iterations', type=int, default=50000)
    cli.add_argument('--seed', type=int, default=0)
    args = cli.parse_args()

    _, embedding_manifest = load_embedding_manifest(args.embedding_dir, args.lang)
    args.layers, _ = resolve_layers(
        args.embedding_dir, embedding_manifest, args.layers
    )
    token_dir = os.path.join(args.save_dir, 'token_types')
    distance_dir = os.path.join(args.save_dir, 'distances')
    token_metadata = token_type_analysis(
        args.embedding_dir,
        token_dir,
        args.lang,
        args.layers,
        args.token_perplexities,
        args.min_tokens,
        args.max_programs,
        args.program_selection,
        args.max_points,
        args.seed,
        args.iterations,
    )
    distance_metadata = distance_analysis(
        args.embedding_dir,
        distance_dir,
        args.lang,
        args.layers,
        args.distance_perplexities,
        args.min_tokens,
        args.seed,
        args.iterations,
    )
    print(
        f'[T-SNE COMPLETE] {args.lang} | '
        f"{token_metadata['num_selected_points']} token-role points | "
        f"distance sample {distance_metadata['num_tokens']} tokens"
    )


if __name__ == '__main__':
    main()
