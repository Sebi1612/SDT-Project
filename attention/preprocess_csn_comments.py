"""Create comment-free, token-aligned CodeSearchNet experiment files."""

import argparse
import hashlib
import json
import os
from collections import Counter

from comment_preprocessing import COMMENT_NODE_TYPES, strip_comments
from dfg_comp import build_parser, get_dfg_adj
from graph_utils import get_ast_tokens_and_prog_graphs, traverse_node


SUPPORTED_LANGUAGES = ('java', 'go', 'javascript')


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def aligned_token_info(code, code_tokens, parser, include_comments):
    byte_code = code.encode('utf-8')
    tree = parser.parse(byte_code)
    if tree.root_node.has_error:
        raise ValueError('Tree-sitter reported a parse error')
    collected = []
    traverse_node(
        tree.root_node,
        collected,
        byte_code,
        include_comments=include_comments,
    )
    token_info, _, is_error = get_ast_tokens_and_prog_graphs(
        collected,
        code_tokens,
        code_tokens,
        byte_code,
        (0, 0),
    )
    aligned = [info['token'] for info in token_info]
    if is_error or aligned != code_tokens:
        raise ValueError('Parser tokens do not exactly match CodeSearchNet tokens')
    return token_info


def preprocess_record(record, parser, language):
    """Clean one CSN record and prove post-cleaning token alignment."""
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(f'Unsupported preprocessing language: {language}')
    code = record['code']
    code_tokens = list(record['code_tokens'])

    original_info = aligned_token_info(
        code, code_tokens, parser, include_comments=True
    )
    cleaned_code, stats = strip_comments(code, parser)
    comment_token_indices = [
        index
        for index, info in enumerate(original_info)
        if info['type'] in COMMENT_NODE_TYPES
    ]
    comment_token_indices = set(comment_token_indices)
    cleaned_tokens = [
        token
        for index, token in enumerate(code_tokens)
        if index not in comment_token_indices
    ]

    # Some upstream datasets may already have comment-free code_tokens while
    # retaining comments in the raw code, as the established Python data does.
    # In that case original alignment contains no comment token; validate the
    # existing token list against the cleaned source before accepting it.
    aligned_token_info(
        cleaned_code,
        cleaned_tokens,
        parser,
        include_comments=False,
    )

    cleaned_record = dict(record)
    cleaned_record['code'] = cleaned_code
    cleaned_record['code_tokens'] = cleaned_tokens
    stats['comment_tokens_removed'] = len(comment_token_indices)
    stats['tokens_before'] = len(code_tokens)
    stats['tokens_after'] = len(cleaned_tokens)
    return cleaned_record, stats


def preprocess_file(
    input_path,
    output_path,
    manifest_path,
    language,
    drop_failures=False,
    require_dfg_alignment=False,
):
    if os.path.abspath(input_path) == os.path.abspath(output_path):
        raise ValueError('Refusing to overwrite the input dataset')

    parser = build_parser(language)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(manifest_path)), exist_ok=True)
    temporary_path = output_path + '.tmp'
    input_partitions = Counter()
    output_partitions = Counter()
    failures = []
    input_rows = 0
    output_rows = 0
    changed_rows = 0
    comments_removed = 0
    comment_tokens_removed = 0
    comment_types = Counter()

    with open(input_path) as source, open(temporary_path, 'w') as output:
        for source_index, line in enumerate(source):
            input_rows += 1
            record = json.loads(line)
            input_partitions[str(record.get('partition'))] += 1
            stage = 'comment_preprocessing_and_ast_alignment'
            try:
                cleaned, stats = preprocess_record(record, parser, language)
                if require_dfg_alignment:
                    stage = 'dfg_alignment'
                    dfg_graph, dfg_tokens = get_dfg_adj(
                        cleaned['code'],
                        parser,
                        lang=language,
                        expected_tokens=cleaned['code_tokens'],
                    )
                    expected_shape = (
                        len(cleaned['code_tokens']),
                        len(cleaned['code_tokens']),
                    )
                    if dfg_tokens != cleaned['code_tokens']:
                        raise ValueError(
                            'Aligned DFG tokens differ from cleaned dataset tokens'
                        )
                    if dfg_graph.shape != expected_shape:
                        raise ValueError(
                            f'DFG shape {dfg_graph.shape} does not match '
                            f'cleaned token shape {expected_shape}'
                        )
                # Preserve provenance when an already-cleaned dataset is
                # validated again; this also makes preprocessing idempotent.
                cleaned.setdefault('preprocessing_source_index', source_index)
                output.write(json.dumps(cleaned) + '\n')
                output_rows += 1
                output_partitions[str(cleaned.get('partition'))] += 1
                changed_rows += stats['comments_removed'] > 0
                comments_removed += stats['comments_removed']
                comment_tokens_removed += stats['comment_tokens_removed']
                comment_types.update(stats['comment_types'])
            except Exception as exc:
                failures.append({
                    'source_index': source_index,
                    'code_file': record.get('code_file'),
                    'stage': stage,
                    'reason': f'{type(exc).__name__}: {exc}',
                })

    status = 'complete' if not failures or drop_failures else 'failed'
    manifest = {
        'status': status,
        'language': language,
        'method': 'tree_sitter_comment_spans_preserve_source_coordinates',
        'input_file': os.path.abspath(input_path),
        'input_sha256': sha256(input_path),
        'output_file': os.path.abspath(output_path),
        'input_rows': input_rows,
        'output_rows': output_rows,
        'input_partitions': dict(input_partitions),
        'output_partitions': dict(output_partitions),
        'changed_rows': changed_rows,
        'comments_removed': comments_removed,
        'comment_tokens_removed': comment_tokens_removed,
        'comment_types': dict(comment_types),
        'num_failures': len(failures),
        'failures': failures,
        'drop_failures': drop_failures,
        'require_dfg_alignment': require_dfg_alignment,
        'validations': [
            'tree_sitter_parse',
            'comment_removal',
            'ast_token_alignment',
        ] + (['dfg_token_alignment'] if require_dfg_alignment else []),
    }

    if status == 'complete':
        os.replace(temporary_path, output_path)
        manifest['output_sha256'] = sha256(output_path)
    else:
        os.remove(temporary_path)
        manifest['output_sha256'] = None
    with open(manifest_path, 'w') as handle:
        json.dump(manifest, handle, indent=2)

    if status != 'complete':
        raise ValueError(
            f'{len(failures)} records failed preprocessing; no output dataset '
            f'was published. See {manifest_path}. Use --drop_failures only when '
            'constructing an explicitly filtered candidate pool.'
        )
    print(
        f'[COMMENT PREPROCESSING COMPLETE] {language}: '
        f'{output_rows}/{input_rows} rows, {comments_removed} comments removed, '
        f'{len(failures)} failures | {manifest_path}'
    )
    return manifest


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument('--input', required=True)
    cli.add_argument('--output', required=True)
    cli.add_argument('--manifest', required=True)
    cli.add_argument('--lang', required=True, choices=SUPPORTED_LANGUAGES)
    cli.add_argument('--drop_failures', action='store_true')
    cli.add_argument(
        '--require_dfg_alignment',
        action='store_true',
        help=(
            'Only publish records whose DFG can be aligned exactly to the '
            'cleaned dataset tokens.'
        ),
    )
    args = cli.parse_args()
    preprocess_file(
        args.input,
        args.output,
        args.manifest,
        args.lang,
        drop_failures=args.drop_failures,
        require_dfg_alignment=args.require_dfg_alignment,
    )


if __name__ == '__main__':
    main()
