"""Shared, auditable evaluation protocol for structural attention analyses."""

import json
import os

import numpy as np


PROTOCOL_VERSION = 'structural_attention_v1'
DEFAULT_THRESHOLDS = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
DEFAULT_PRIMARY_THRESHOLD = 0.05
DEFAULT_BOOTSTRAP_SAMPLES = 1000
DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_BOOTSTRAP_SEED = 0


def threshold_index(thresholds, primary_threshold):
    """Return an unambiguous index for the predeclared primary threshold."""
    matches = [
        index
        for index, threshold in enumerate(thresholds)
        if np.isclose(threshold, primary_threshold, rtol=0, atol=1e-12)
    ]
    if len(matches) != 1:
        raise ValueError(
            f'Primary threshold {primary_threshold} must occur exactly once in '
            f'--thresholds; received {thresholds}'
        )
    return matches[0]


def bootstrap_mean_ci(
    values,
    num_resamples=DEFAULT_BOOTSTRAP_SAMPLES,
    confidence_level=DEFAULT_CONFIDENCE_LEVEL,
    seed=DEFAULT_BOOTSTRAP_SEED,
    batch_size=25,
):
    """Percentile CI for a macro mean, resampling whole programs.

    The first axis must identify independent programs. All remaining dimensions
    are evaluated with the same resampled program indices, preserving paired
    layer/head comparisons.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim < 1 or values.shape[0] == 0:
        raise ValueError('Bootstrap values must contain at least one program')
    if not np.isfinite(values).all():
        raise ValueError('Bootstrap values must be finite')
    if num_resamples < 0:
        raise ValueError('--bootstrap_samples cannot be negative')
    if not 0 < confidence_level < 1:
        raise ValueError('--confidence_level must be between 0 and 1')
    if num_resamples == 0:
        return None, None

    program_count = values.shape[0]
    flat = values.reshape(program_count, -1)
    means = np.empty((num_resamples, flat.shape[1]), dtype=np.float64)
    rng = np.random.default_rng(seed)
    for start in range(0, num_resamples, batch_size):
        stop = min(start + batch_size, num_resamples)
        indices = rng.integers(
            0, program_count, size=(stop - start, program_count)
        )
        means[start:stop] = flat[indices].mean(axis=1)

    tail = (1 - confidence_level) / 2
    lower, upper = np.quantile(means, [tail, 1 - tail], axis=0)
    return lower.reshape(values.shape[1:]), upper.reshape(values.shape[1:])


def confidence_intervals_to_dict(lower, upper):
    """Serialize one layer's head-level confidence limits."""
    if lower is None:
        return None
    return {
        str(head): {
            'lower': float(lower[head]),
            'upper': float(upper[head]),
        }
        for head in range(len(lower))
    }


def protocol_metadata(
    analysis,
    model=None,
    language=None,
    primary_threshold=DEFAULT_PRIMARY_THRESHOLD,
    thresholds=None,
    bootstrap_samples=DEFAULT_BOOTSTRAP_SAMPLES,
    confidence_level=DEFAULT_CONFIDENCE_LEVEL,
    bootstrap_seed=DEFAULT_BOOTSTRAP_SEED,
):
    """Describe choices that affect interpretation of reported numbers."""
    metadata = {
        'protocol_version': PROTOCOL_VERSION,
        'analysis': analysis,
        'model': model,
        'language': language,
        'model_extraction_validated_for': ['codebert'],
        'supported_languages': ['python', 'java', 'go', 'javascript'],
        'primary_threshold': primary_threshold,
        'threshold_predicate': 'attention > threshold',
        'averaging': 'macro mean over programs',
        'bootstrap_unit': 'program',
        'bootstrap_method': 'percentile',
        'bootstrap_samples': bootstrap_samples,
        'confidence_level': confidence_level,
        'bootstrap_seed': bootstrap_seed,
        'graph_direction': 'directed',
        'parser_failures': 'excluded and reported in coverage counts',
        'empty_dfgs': 'included as valid zero-edge graphs',
        'identifier_policy': "tree-sitter type contains 'identifier'",
        'dfg_relation_policy': (
            'binary edges for attention evaluation; typed labels, when '
            'requested separately, are native to each GraphCodeBERT extractor'
        ),
        'typed_dfg_cross_language_comparison_supported': False,
        'known_limitations': [
            'GraphCodeBERT DFG state is name-based and does not fully model '
            'lexical scope shadowing.',
            'Passing alignment validates token identity and graph shape, not '
            'complete semantic correctness for every language construct.',
        ],
    }
    if thresholds is not None:
        metadata['threshold_sweep'] = list(thresholds)
    if analysis == 'similarity':
        metadata.update({
            'distance': 'exact fixed-node edge edit distance',
            'distance_normalization': (
                'edge insertions plus deletions divided by number of nodes'
            ),
        })
    return metadata


def write_protocol(output_dir, metadata):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'evaluation_protocol.json')
    with open(path, 'w') as handle:
        json.dump(metadata, handle, indent=2)
    return os.path.abspath(path)
