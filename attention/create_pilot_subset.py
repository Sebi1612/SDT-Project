import argparse
import hashlib
import json
import os
import random
from collections import Counter


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def create_subset(
    source_path,
    output_path,
    manifest_path,
    language,
    size,
    seed,
    source_manifest_path=None,
):
    with open(source_path) as source:
        rows = source.readlines()

    if size > len(rows):
        raise ValueError(f'Requested {size} rows from a dataset with {len(rows)} rows')

    selected_indices = sorted(random.Random(seed).sample(range(len(rows)), size))
    selected_rows = []
    selected_source_indices = []
    selected_partitions = Counter()
    for source_index in selected_indices:
        row = json.loads(rows[source_index])
        original_source_index = row.get(
            'preprocessing_source_index', source_index
        )
        if original_source_index != source_index:
            row['pilot_candidate_index'] = source_index
        row['pilot_source_index'] = original_source_index
        row['pilot_seed'] = seed
        selected_rows.append(row)
        selected_source_indices.append(original_source_index)
        selected_partitions[str(row.get('partition'))] += 1

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'w') as output:
        for row in selected_rows:
            output.write(json.dumps(row) + '\n')

    manifest = {
        'language': language,
        'source_file': os.path.abspath(source_path),
        'source_sha256': sha256(source_path),
        'source_rows': len(rows),
        'output_file': os.path.abspath(output_path),
        'output_sha256': sha256(output_path),
        'size': size,
        'seed': seed,
        'selected_candidate_indices': selected_indices,
        'selected_source_indices': selected_source_indices,
        'partitions': dict(selected_partitions),
    }
    if source_manifest_path is not None:
        manifest['source_manifest'] = os.path.abspath(source_manifest_path)
        manifest['source_manifest_sha256'] = sha256(source_manifest_path)
    with open(manifest_path, 'w') as output:
        json.dump(manifest, output, indent=2)

    print(
        f"Created {language} pilot subset with {size} rows at {output_path} "
        f"(seed={seed}, sha256={manifest['output_sha256']})"
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--lang', required=True)
    parser.add_argument('--size', default=100, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--source_manifest')
    args = parser.parse_args()

    create_subset(
        args.source,
        args.output,
        args.manifest,
        args.lang,
        args.size,
        args.seed,
        source_manifest_path=args.source_manifest,
    )
