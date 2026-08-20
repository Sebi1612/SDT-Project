"""Build a reproducible, paper-compatible CodeSearchNet attention cohort."""

import argparse
import hashlib
import json
import os
import random
from collections import Counter

from tqdm import tqdm
from transformers import PLBartTokenizer, RobertaTokenizer

from dfg_comp import build_parser, get_dfg_adj
from preprocess_csn_comments import preprocess_record, sha256


SUPPORTED_LANGUAGES = ('java', 'go', 'javascript')
MODEL_TOKENIZERS = {
    'codebert': ('microsoft/codebert-base', 'roberta'),
    'graphcodebert': ('microsoft/graphcodebert-base', 'roberta'),
    'unixcoder': ('microsoft/unixcoder-base', 'roberta'),
    'plbart': ('uclanlp/plbart-base', 'plbart'),
}


class RecordRejected(ValueError):
    def __init__(self, stage, reason):
        super().__init__(reason)
        self.stage = stage


def tokens_merge_exactly(tokenized_tokens, code_tokens, special_char='Ġ'):
    """Mirror the repository's attention merge without allocating attention."""
    expected = [token.replace(' ', '') for token in code_tokens]
    expected_index = 0
    merged = ''
    for token in tokenized_tokens:
        while token.startswith(special_char):
            token = token[len(special_char):]
        if expected_index >= len(expected):
            return False
        merged += token
        if merged == expected[expected_index]:
            expected_index += 1
            merged = ''
    return expected_index == len(expected) and not merged


def load_tokenizers():
    tokenizers = {}
    for model, (version, tokenizer_type) in MODEL_TOKENIZERS.items():
        tokenizer_class = (
            PLBartTokenizer if tokenizer_type == 'plbart' else RobertaTokenizer
        )
        tokenizers[model] = tokenizer_class.from_pretrained(
            version,
            local_files_only=True,
        )
    return tokenizers


def validate_and_clean(record, parser, language, tokenizers):
    if record.get('partition') != 'test':
        raise RecordRejected('partition', 'record is not from the test split')
    if not record.get('code_tokens'):
        raise RecordRejected('tokens', 'record has no code tokens')

    # The Go export in the local aggregate CSN file retains standalone newline
    # tokens, while the paper's token-to-AST alignment treats whitespace as
    # source layout rather than graph nodes.
    normalized = dict(record)
    normalized['code_tokens'] = [
        token for token in record['code_tokens'] if token.strip()
    ]
    whitespace_tokens_removed = (
        len(record['code_tokens']) - len(normalized['code_tokens'])
    )

    try:
        cleaned, comment_stats = preprocess_record(
            normalized, parser, language
        )
    except Exception as exc:
        raise RecordRejected(
            'comment_preprocessing_and_ast_alignment',
            f'{type(exc).__name__}: {exc}',
        ) from exc

    try:
        dfg, dfg_tokens = get_dfg_adj(
            cleaned['code'],
            parser,
            lang=language,
            expected_tokens=cleaned['code_tokens'],
        )
        expected_shape = (len(cleaned['code_tokens']),) * 2
        if dfg_tokens != cleaned['code_tokens'] or dfg.shape != expected_shape:
            raise ValueError('DFG is not exactly aligned to the cleaned tokens')
    except Exception as exc:
        raise RecordRejected(
            'dfg_alignment', f'{type(exc).__name__}: {exc}'
        ) from exc

    joined = ' '.join(cleaned['code_tokens'])
    model_lengths = {}
    for model, tokenizer in tokenizers.items():
        model_tokens = tokenizer.tokenize(joined)
        model_lengths[model] = len(model_tokens)
        if model != 'plbart' and len(model_tokens) >= 500:
            raise RecordRejected(
                'model_length',
                f'{model} has {len(model_tokens)} subtokens; expected < 500',
            )
        special_char = '▁' if model == 'plbart' else 'Ġ'
        if not tokens_merge_exactly(
            model_tokens,
            cleaned['code_tokens'],
            special_char=special_char,
        ):
            raise RecordRejected(
                'model_token_alignment',
                f'{model} subtokens do not merge exactly to code_tokens',
            )

    comment_stats['whitespace_tokens_removed'] = whitespace_tokens_removed
    return cleaned, comment_stats, model_lengths


def prepare_cohort(
    input_path,
    output_path,
    manifest_path,
    language,
    sample_size=3000,
    seed=0,
):
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(f'Unsupported language: {language}')
    if sample_size <= 0:
        raise ValueError('--sample_size must be positive')
    if os.path.abspath(input_path) == os.path.abspath(output_path):
        raise ValueError('Refusing to overwrite the candidate dataset')

    parser = build_parser(language)
    tokenizers = load_tokenizers()
    rng = random.Random(seed)
    sample = []
    eligible_count = 0
    input_rows = 0
    test_rows = 0
    rejection_stages = Counter()
    rejection_reasons = Counter()
    rejection_examples = []
    accepted_comment_stats = Counter()
    accepted_comment_types = Counter()
    accepted_model_lengths = {
        model: [] for model in MODEL_TOKENIZERS
    }

    with open(input_path) as source:
        for source_index, line in enumerate(tqdm(source, desc=language)):
            input_rows += 1
            record = json.loads(line)
            if record.get('partition') != 'test':
                continue
            test_rows += 1
            try:
                cleaned, comment_stats, model_lengths = validate_and_clean(
                    record, parser, language, tokenizers
                )
            except RecordRejected as exc:
                rejection_stages[exc.stage] += 1
                rejection_reasons[str(exc)] += 1
                if len(rejection_examples) < 100:
                    rejection_examples.append({
                        'source_index': source_index,
                        'code_file': record.get('code_file'),
                        'stage': exc.stage,
                        'reason': str(exc),
                    })
                continue

            cleaned['preprocessing_source_index'] = source_index
            candidate = {
                'record': cleaned,
                'comment_stats': comment_stats,
                'model_lengths': model_lengths,
            }
            eligible_count += 1
            if len(sample) < sample_size:
                sample.append(candidate)
            else:
                replacement_index = rng.randrange(eligible_count)
                if replacement_index < sample_size:
                    sample[replacement_index] = candidate

    if eligible_count < sample_size:
        raise ValueError(
            f'Only {eligible_count} eligible {language} test records; '
            f'{sample_size} requested'
        )

    sample.sort(key=lambda item: item['record']['preprocessing_source_index'])
    for item in sample:
        comment_stats = item['comment_stats']
        accepted_comment_stats.update({
            'comments_removed': comment_stats['comments_removed'],
            'comment_bytes_removed': comment_stats['comment_bytes_removed'],
            'comment_tokens_removed': comment_stats['comment_tokens_removed'],
            'whitespace_tokens_removed': comment_stats[
                'whitespace_tokens_removed'
            ],
        })
        accepted_comment_types.update(comment_stats['comment_types'])
        for model, length in item['model_lengths'].items():
            accepted_model_lengths[model].append(length)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(manifest_path)), exist_ok=True)
    temporary_path = output_path + '.tmp'
    with open(temporary_path, 'w') as output:
        for item in sample:
            output.write(json.dumps(item['record']) + '\n')
    os.replace(temporary_path, output_path)

    code_hashes = [
        hashlib.sha256(item['record']['code'].encode('utf-8')).hexdigest()
        for item in sample
    ]
    manifest = {
        'status': 'complete',
        'language': language,
        'seed': seed,
        'sampling_method': 'uniform_reservoir_sampling_after_validation',
        'input_file': os.path.abspath(input_path),
        'input_sha256': sha256(input_path),
        'output_file': os.path.abspath(output_path),
        'output_sha256': sha256(output_path),
        'input_rows': input_rows,
        'test_rows': test_rows,
        'eligible_test_rows': eligible_count,
        'sample_size': len(sample),
        'output_partitions': {'test': len(sample)},
        'duplicate_cleaned_code_count': len(code_hashes) - len(set(code_hashes)),
        'rejection_stages': dict(rejection_stages),
        'rejection_reasons': dict(rejection_reasons),
        'rejection_examples': rejection_examples,
        'accepted_comment_stats': dict(accepted_comment_stats),
        'accepted_comment_types': dict(accepted_comment_types),
        'model_token_lengths': {
            model: {
                'minimum': min(lengths),
                'maximum': max(lengths),
                'mean': sum(lengths) / len(lengths),
            }
            for model, lengths in accepted_model_lengths.items()
        },
        'selected_source_indices': [
            item['record']['preprocessing_source_index'] for item in sample
        ],
        'validations': [
            'test_split',
            'tree_sitter_parse_before_and_after_comment_removal',
            'tree_sitter_comment_removal',
            'standalone_whitespace_token_removal',
            'ast_token_alignment',
            'dfg_token_alignment',
            'codebert_graphcodebert_unixcoder_length_below_500',
            'codebert_graphcodebert_unixcoder_plbart_token_merge_alignment',
        ],
    }
    temporary_manifest = manifest_path + '.tmp'
    with open(temporary_manifest, 'w') as handle:
        json.dump(manifest, handle, indent=2)
    os.replace(temporary_manifest, manifest_path)
    print(
        f'[ATTENTION COHORT COMPLETE] {language}: {len(sample)} sampled from '
        f'{eligible_count}/{test_rows} eligible test records | {manifest_path}'
    )
    return manifest


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument('--input', required=True)
    cli.add_argument('--output', required=True)
    cli.add_argument('--manifest', required=True)
    cli.add_argument('--lang', required=True, choices=SUPPORTED_LANGUAGES)
    cli.add_argument('--sample_size', type=int, default=3000)
    cli.add_argument('--seed', type=int, default=0)
    args = cli.parse_args()
    prepare_cohort(
        args.input,
        args.output,
        args.manifest,
        args.lang,
        sample_size=args.sample_size,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()
