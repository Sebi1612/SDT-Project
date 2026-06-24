import json
import random
from tqdm import tqdm
from transformers import RobertaTokenizer, PLBartTokenizer
from unixcoder import UniXcoder

INPUT_FILE = 'exp_data/java_full.jsonl'
OUTPUT_FILE = 'exp_data/exp_0_java.jsonl'
SAMPLE_SIZE = 3000


def check_merge_token_size(tokenized_tokens, code_tokens, special_char='Ġ'):
    modified_code_tokens = [t.replace(" ", "") for t in code_tokens]
    code_idx = 0
    merged_token = ''
    for token in tokenized_tokens:
        while len(token) > 0 and token[0] == special_char:
            token = token[1:]
        merged_token += token
        if merged_token == modified_code_tokens[code_idx]:
            code_idx += 1
            merged_token = ''
    return code_idx == len(modified_code_tokens)


print("Loading tokenizers...")
cb_tokenizer  = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
gcb_tokenizer = RobertaTokenizer.from_pretrained('microsoft/graphcodebert-base')
plb_tokenizer = PLBartTokenizer.from_pretrained('uclanlp/plbart-base')
uxc_model     = UniXcoder('microsoft/unixcoder-base')

print("Filtering...")
passed = []

with open(INPUT_FILE) as f:
    lines = f.readlines()

for line in tqdm(lines):
    entry = json.loads(line)
    code_tokens = entry['code_tokens']

    if not code_tokens or not entry.get('docstring_tokens'):
        continue

    joined = " ".join(code_tokens)

    # codebert
    tokenized = cb_tokenizer.tokenize(joined)
    if len(tokenized) > 500 or not check_merge_token_size(tokenized, code_tokens):
        continue

    # graphcodebert
    tokenized = gcb_tokenizer.tokenize(joined)
    if len(tokenized) > 500 or not check_merge_token_size(tokenized, code_tokens):
        continue

    # unixcoder
    tokenized, _ = uxc_model.tokenize([joined])
    if len(tokenized[0]) > 500 or not check_merge_token_size(tokenized[0], code_tokens):
        continue

    # plbart
    tokenized = plb_tokenizer.tokenize(joined)
    if not check_merge_token_size(tokenized, code_tokens, special_char='▁'):
        continue

    passed.append(entry)

print(f"{len(passed)} entries passed filtering.")

if len(passed) < SAMPLE_SIZE:
    print(f"Warning: only {len(passed)} entries available, less than {SAMPLE_SIZE}.")
    sampled = passed
else:
    sampled = random.sample(passed, SAMPLE_SIZE)

with open(OUTPUT_FILE, 'w') as f:
    for entry in sampled:
        f.write(json.dumps(entry) + "\n")

print(f"Saved {len(sampled)} entries to {OUTPUT_FILE}.")
