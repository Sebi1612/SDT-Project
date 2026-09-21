import argparse
import hashlib
import json
import os
import pickle
import torch
import numpy as np
from tqdm import tqdm
import copy

from transformers import RobertaModel, RobertaTokenizer, T5ForConditionalGeneration, RobertaForMaskedLM
from transformers import PLBartTokenizer, PLBartForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

from unixcoder import UniXcoder

from tree_sitter import Language, Parser

from utils import load_codesearchnet
from dfg_comp import build_parser
from graph_utils import get_ast_tokens_and_prog_graphs, traverse_node

def get_model_and_tokenizer(model, model_version = None):
    valid_models = ['codebert', 'graphcodebert', 'unixcoder', 'codet5', 'plbart', 'coderl', 'codet5p_2b', 'codet5p_220', 'codet5p_770', 'codet5_musu', 'codet5_lntp', 'codegen', 'codet5p_2b_dec']
    assert model in valid_models, f'Wrong model name : {model}'

    if model == 'codebert':
        model_version = 'microsoft/codebert-base'
        model = RobertaModel.from_pretrained(model_version, output_hidden_states = True)
        tokenizer = RobertaTokenizer.from_pretrained(model_version)
        special_char='Ġ'
        return model, tokenizer, special_char

    if model == 'graphcodebert':
        model_version = 'microsoft/graphcodebert-base'
        model = RobertaForMaskedLM.from_pretrained(model_version, output_hidden_states = True)
        tokenizer = RobertaTokenizer.from_pretrained(model_version)
        special_char='Ġ'
        return model, tokenizer, special_char 


    if model == 'plbart':
        model_version = 'uclanlp/plbart-base'
        tokenizer = PLBartTokenizer.from_pretrained(model_version)
        model = PLBartForConditionalGeneration.from_pretrained(model_version, output_hidden_states = True)
        special_char = '▁'
        return model, tokenizer, special_char

    if model == 'codet5':
        model_version = 'Salesforce/codet5-base'
        model = T5ForConditionalGeneration.from_pretrained(model_version, output_hidden_states = True)
        tokenizer = RobertaTokenizer.from_pretrained(model_version)
        special_char='Ġ'
        return model, tokenizer, special_char

    if model == 'unixcoder':
        model_version = 'microsoft/unixcoder-base'
        model = UniXcoder(model_version)
        special_char='Ġ'
        return model, None, special_char

    if model == 'coderl':
        tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-large-ntp-py')
        model_version = 'coderl_weights/coderl/'
        model = T5ForConditionalGeneration.from_pretrained(model_version, output_hidden_states = True)
        special_char='Ġ'
        return model, tokenizer, special_char

    if model == 'codet5p_2b':
        model_version = "Salesforce/codet5p-2b"
        tokenizer = AutoTokenizer.from_pretrained(model_version)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_version,
                                              torch_dtype=torch.float16,
                                              trust_remote_code=True, output_hidden_states=True)

        model.config.decoder_start_token_id = tokenizer.bos_token_id
        model.config.pad_token_id = tokenizer.eos_token_id
        model.config.encoder.output_hidden_states = True
        special_char='Ġ'
        return model, tokenizer, special_char
        
        
    if model == 'codet5p_2b_dec':
        model_version = "Salesforce/codet5p-2b"
        tokenizer = AutoTokenizer.from_pretrained(model_version)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_version,
                                              torch_dtype=torch.float16,
                                              trust_remote_code=True, output_hidden_states=True)
        
        model.config.add_cross_attention=False
        model.config.output_attentions = True
        model.config.output_hidden_states = True
        model.config.decoder.add_cross_attention = False
        model.config.decoder.output_attentions = True
        model.config.decoder.output_hidden_states = True
        model.config.decoder_start_token_id = tokenizer.bos_token_id
        model.config.pad_token_id = tokenizer.eos_token_id
        
        special_char='Ġ'
        return model, tokenizer, special_char
    
    
    if model == 'codet5p_220':
        model_version = 'Salesforce/codet5p-220m'
        model = T5ForConditionalGeneration.from_pretrained(model_version, output_hidden_states = True)
        tokenizer = RobertaTokenizer.from_pretrained(model_version)
        special_char='Ġ'
        return model, tokenizer, special_char
        
    if model == 'codet5p_770':
        model_version = 'Salesforce/codet5p-770m'
        model = T5ForConditionalGeneration.from_pretrained(model_version, output_hidden_states = True)
        tokenizer = RobertaTokenizer.from_pretrained(model_version)
        special_char='Ġ'
        return model, tokenizer, special_char
        
    if model == 'codet5_musu':
        model_version = 'Salesforce/codet5-base-multi-sum'
        model = T5ForConditionalGeneration.from_pretrained(model_version, output_hidden_states = True)
        tokenizer = RobertaTokenizer.from_pretrained(model_version)
        special_char='Ġ'
        return model, tokenizer, special_char
        
    if model == 'codet5_lntp':
        model_version = 'Salesforce/codet5-large-ntp-py'
        model = T5ForConditionalGeneration.from_pretrained(model_version, output_hidden_states = True)
        tokenizer = RobertaTokenizer.from_pretrained(model_version)
        special_char='Ġ'
        return model, tokenizer, special_char
        
    if model == 'codegen':
        model_version = 'Salesforce/codegen2-3_7B'
        model = AutoModelForCausalLM.from_pretrained(model_version, output_hidden_states = True, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_version)
        special_char='Ġ'
        return model, tokenizer, special_char
        

def merge_hidden_repr(hidden_states, tokenized_tokens, code_tokens, start_index = 1, end_index=-1, special_char = 'Ġ'):
    mask = []
    code_idx = 0
    merged_token = ''
    merged_tokens = []

    modified_code_tokens = []

    for token in code_tokens:
        if not " " in token:
            modified_code_tokens.append(token)
        else:
            modified_code_tokens.append(token.replace(" ", ""))

    for i in range(len(tokenized_tokens)):
        token = tokenized_tokens[i]
        while len(token) > 0 and token[0] == special_char:
            token = token[1:]
        merged_token += token
        mask.append(code_idx)

        if merged_token == modified_code_tokens[code_idx]:
            code_idx += 1
            merged_tokens.append(merged_token)
            merged_token = ''

    if code_idx != len(modified_code_tokens):
        raise Exception(f'Tokens mismatch: \n {code_idx}, {len(modified_code_tokens)} ')

    mask = torch.tensor(mask)
    num_layers = len(hidden_states)
    
    if end_index == -1:
        all_hidden_states = torch.cat([hidden_states[n][:, start_index:-1, :] for n in range(num_layers)], dim=0)
        seq_len = len(modified_code_tokens)
        all_hidden_states = torch.stack([all_hidden_states[:, mask == m, :].mean(dim = 1) for m in range(seq_len)], dim = 1)
    else:
        all_hidden_states = torch.cat([hidden_states[n][:, start_index:, :] for n in range(num_layers)], dim=0)
        seq_len = len(modified_code_tokens)
        all_hidden_states = torch.stack([all_hidden_states[:, mask == m, :].mean(dim = 1) for m in range(seq_len)], dim = 1)
    return all_hidden_states.cpu().detach().numpy()


def _node_paths(root_node):
    """Map tree-sitter node IDs to root-to-node paths for exact distances."""
    paths = {}

    def visit(node, path):
        current = path + (node.id,)
        paths[node.id] = current
        for child in node.children:
            visit(child, current)

    visit(root_node, ())
    return paths


def _base_node_id(node_id):
    return node_id[0] if isinstance(node_id, tuple) else node_id


def aligned_ast_structure(code, code_tokens, tree_sitter_parser, lang):
    """Create token types and pairwise tree distances on artifact token nodes."""
    byte_code = code.encode('utf-8')
    tree = tree_sitter_parser.parse(byte_code)

    collected = []
    traverse_node(
        tree.root_node,
        collected,
        byte_code,
        include_comments=False,
    )
    ast_info, _, is_error = get_ast_tokens_and_prog_graphs(
        collected, code_tokens, code_tokens, byte_code, (0, 0)
    )
    if is_error or [info['token'] for info in ast_info] != code_tokens:
        raise ValueError('AST tokens do not exactly match dataset tokens')

    paths = _node_paths(tree.root_node)
    token_node_ids = []
    for info in ast_info:
        ids = {
            _base_node_id(item['id'])
            for item in collected
            if item['start_byte'] >= info['start_byte']
            and item['end_byte'] <= info['end_byte']
            and _base_node_id(item['id']) in paths
        }
        if not ids:
            base_id = _base_node_id(info['id'])
            if base_id not in paths:
                raise ValueError(f'Could not locate AST node for token {info["token"]!r}')
            ids = {base_id}
        token_node_ids.append(sorted(ids))

    def node_distance(first, second):
        a = paths[first]
        b = paths[second]
        common = 0
        for left, right in zip(a, b):
            if left != right:
                break
            common += 1
        return len(a) + len(b) - 2 * common

    size = len(ast_info)
    distances = np.zeros((size, size), dtype=np.int16)
    for row in range(size):
        for column in range(row + 1, size):
            distance = min(
                node_distance(first, second)
                for first in token_node_ids[row]
                for second in token_node_ids[column]
            )
            distances[row, column] = distance
            distances[column, row] = distance

    token_info = [
        {
            'token': info['token'],
            'type': info['type'],
            'start_byte': info['start_byte'],
            'end_byte': info['end_byte'],
        }
        for info in ast_info
    ]
    return distances, token_info


def load_codebert_hidden_model(device):
    model_version = 'microsoft/codebert-base'
    tokenizer = RobertaTokenizer.from_pretrained(model_version)
    model = RobertaModel.from_pretrained(
        model_version, output_hidden_states=True
    ).to(device)
    model.eval()
    return model, tokenizer, model_version


def merge_codebert_hidden_states(hidden_states, subtokens, code_tokens):
    """Average CodeBERT subtokens into the same lexical nodes as graph artifacts."""
    normalized = [token.replace(' ', '') for token in code_tokens]
    groups = []
    token_index = 0
    accumulated = ''
    current = []
    for subtoken_index, subtoken in enumerate(subtokens):
        piece = subtoken.lstrip('Ġ')
        accumulated += piece
        current.append(subtoken_index)
        if token_index >= len(normalized):
            raise ValueError('Tokenizer produced trailing subtokens')
        if accumulated == normalized[token_index]:
            groups.append(current)
            current = []
            accumulated = ''
            token_index += 1
    if current or token_index != len(normalized):
        raise ValueError(
            f'Subtoken alignment stopped at token {token_index}/{len(normalized)}'
        )

    # hidden_states contains embedding layer 0 plus all 12 transformer layers.
    stacked = torch.stack([state[0, 1:-1, :] for state in hidden_states])
    merged = torch.stack(
        [stacked[:, indexes, :].mean(dim=1) for indexes in groups], dim=1
    )
    return merged.detach().cpu().numpy().astype(np.float32, copy=False)


def save_codebert_embeddings(args, tree_sitter_parser):
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError(
            f'Requested --device {args.device!r}, but CUDA is unavailable'
        )
    if not args.graph_loc:
        raise ValueError('--graph_loc is required for deterministic CodeBERT extraction')

    graph_manifest_path = os.path.join(args.graph_loc, 'graph_manifest.json')
    with open(graph_manifest_path) as handle:
        graph_manifest = json.load(handle)
    if graph_manifest.get('status') != 'complete':
        raise ValueError('Graph manifest is not complete')
    if graph_manifest.get('lang') != args.lang:
        raise ValueError('Graph manifest language does not match --lang')
    artifacts = list(graph_manifest['artifacts'])
    if args.num_codes is not None:
        artifacts = artifacts[:args.num_codes]

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model, tokenizer, model_version = load_codebert_hidden_model(args.device)

    output_dir = os.path.join(args.save_dir, args.lang)
    if args.exp_name:
        output_dir = os.path.join(output_dir, args.exp_name)
    output_dir = os.path.join(output_dir, args.model)
    os.makedirs(output_dir, exist_ok=True)

    with open(graph_manifest_path, 'rb') as handle:
        graph_manifest_sha256 = hashlib.sha256(handle.read()).hexdigest()
    manifest = {
        'status': 'in_progress',
        'model': args.model,
        'model_version': model_version,
        'language': args.lang,
        'seed': args.seed,
        'device': args.device,
        'inference_mode': True,
        'graph_manifest': os.path.abspath(graph_manifest_path),
        'graph_manifest_sha256': graph_manifest_sha256,
        'num_requested': len(artifacts),
        'artifacts': [],
        'failures': [],
    }
    manifest_path = os.path.join(output_dir, 'embedding_manifest.json')

    def write_manifest():
        temporary = manifest_path + '.tmp'
        with open(temporary, 'w') as handle:
            json.dump(manifest, handle, indent=2)
        os.replace(temporary, manifest_path)

    write_manifest()
    for artifact_name in tqdm(artifacts):
        try:
            with open(os.path.join(args.graph_loc, artifact_name), 'rb') as handle:
                graph = pickle.load(handle)
            code_tokens = graph['code_tokens']
            subtokens = tokenizer.tokenize(' '.join(code_tokens))
            tokens = [tokenizer.cls_token] + subtokens + [tokenizer.sep_token]
            input_ids = torch.tensor(
                [tokenizer.convert_tokens_to_ids(tokens)], device=args.device
            )
            with torch.inference_mode():
                outputs = model(input_ids=input_ids)
            hidden_repr = merge_codebert_hidden_states(
                outputs.hidden_states, subtokens, code_tokens
            )
            tree_dist, code_token_info = aligned_ast_structure(
                graph['code'], code_tokens, tree_sitter_parser, args.lang
            )
            if hidden_repr.shape[:2] != (13, len(code_tokens)):
                raise ValueError(
                    f'Hidden shape {hidden_repr.shape}, expected '
                    f'(13, {len(code_tokens)}, 768)'
                )
            if tree_dist.shape != (len(code_tokens), len(code_tokens)):
                raise ValueError('Tree-distance shape does not match tokens')

            output_name = artifact_name
            output_path = os.path.join(output_dir, output_name)
            payload = {
                'hidden_repr': hidden_repr,
                'tree_dist': tree_dist,
                'code_token_info': code_token_info,
                'code_tokens': code_tokens,
                'code_file': graph.get('file_name'),
                'sample_index': graph.get('sample_index'),
                'source_index': graph.get('source_index'),
                'source_graph_artifact': artifact_name,
                'language': args.lang,
            }
            with open(output_path, 'wb') as handle:
                pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            manifest['artifacts'].append(output_name)
        except Exception as exc:
            manifest['failures'].append({
                'source_graph_artifact': artifact_name,
                'reason': f'{type(exc).__name__}: {exc}',
            })
        write_manifest()

    manifest['status'] = 'complete'
    manifest['num_saved'] = len(manifest['artifacts'])
    manifest['num_failures'] = len(manifest['failures'])
    write_manifest()
    print(
        f'[HIDDEN STATES COMPLETE] {args.lang} | '
        f"{manifest['num_saved']}/{len(artifacts)} artifacts | {manifest_path}"
    )
    return manifest
        

def get_lca_info(node, walk_path, curr_depth, depth, is_code_token, code_token_info, byte_code):
    if node.child_count == 0:
        token = byte_code[node.start_byte : node.end_byte].decode('utf-8')
        walk_path.append(token)
        depth.append(curr_depth)
        is_code_token.append(True)

        token_info = {}
        token_info['type'] = node.type
        token_info['token'] = token
        token_info['start_byte'] = node.start_byte
        token_info['end_byte'] = node.end_byte
        code_token_info.append(token_info)

    else:
        walk_path.append(node.type)
        depth.append(curr_depth)
        is_code_token.append(False)

    children = node.children
    if node.type == 'string':
        text = ''
        for child in children:
            text += byte_code[child.start_byte:child.end_byte].decode('utf-8')

    for child in children:
        curr_depth += 1
        get_lca_info(child, walk_path, curr_depth, depth, is_code_token, code_token_info, byte_code)
        curr_depth -= 1

        if node.child_count == 0:
            token = byte_code[node.start_byte:node.end_byte].decode('utf-8')
            walk_path.append(token)
            depth.append(curr_depth)
            is_code_token.append(True)

        else:
            walk_path.append(node.type)
            depth.append(curr_depth)
            is_code_token.append(False)


# get tree distances
def find_tree_distance(u, v, walk_path, depth):
    if u > v:
        a = v
        b = u
    else:
        a = u
        b = v

    slice = depth[a:b+1]
    min_depth = min(slice)
    index = slice.index(min_depth) + a

    tree_dist = depth[u] + depth[v] - 2*min_depth
    lca = walk_path[index]
    return lca, tree_dist


def rows_to_delete_or_merge(code_token_info, code_tokens, byte_code):
    rows_to_delete = []
    rows_to_merge = []
    exclude_token = ['"""', "'''", "\\'"]
    to_skip = []
    code_token_idx = 0

    for i, token_info in enumerate(code_token_info):
        if i in to_skip:
            continue
        token = token_info['token']
        new_token = copy.deepcopy(token_info)

        if token == ';':
            rows_to_delete.append(i)
            continue
        if token in exclude_token or token_info['type'] == 'comment':
            rows_to_delete.append(i)
            continue
        if token != code_tokens[code_token_idx]:
            if token not in code_tokens[code_token_idx]:
                rows_to_delete.append(i)
                continue

        if token == code_tokens[code_token_idx]:
            code_token_idx += 1
            continue

        merged_rows = []
        merged_rows.append(i)

        if new_token['token'] == "r'''":
            next_info = code_token_info[i+1]
            new_token['token'] += byte_code[new_token['end_byte']:next_info['start_byte']].decode('utf-8') + "'''"
            merged_rows.append(i+1)

        j = 0
        while new_token['token'] != code_tokens[code_token_idx]:
            j += 1
            next_info = code_token_info[i+j]

            new_token['token'] += byte_code[new_token['end_byte']:next_info['start_byte']].decode('utf-8') + next_info['token']

            new_token['end_byte'] = next_info['end_byte']
            merged_rows.append(i+j)
            to_skip.append(i+j)

        rows_to_merge.append(merged_rows)
        code_token_idx += 1

    return rows_to_delete, rows_to_merge


#save required informtation
def save_word_embeddings(args, save_dir):
    model, tokenizer, special_char = get_model_and_tokenizer(args.model)
    model = model.to(args.device)

    codes = load_codesearchnet(args.code_file)
    embeddings_dict = {}

    code_num = 0
    for code in tqdm(codes):
        code_tokens = code['code_tokens']
        code_file_name = code['code_file']
        
        if args.model == 'unixcoder':
            tokenized_tokens, token_ids = model.tokenize([' '.join(code_tokens)])
            source_ids = torch.tensor(token_ids).to(args.device)
            outputs, _, _ = model(source_ids)
            tokenized_tokens = tokenized_tokens[0]

        elif args.model == 'codegen':    
            tokenized_tokens = tokenizer.tokenize(' '.join(code_tokens))
            tokens = tokenized_tokens 
            token_idx = tokenizer.convert_tokens_to_ids(tokens)
            inputs = torch.tensor(token_idx).unsqueeze(0)
            inputs = inputs.to(args.device)
            outputs = model(inputs)
            
        elif args.model == 'codet5p_2b':    
            tokenized_tokens = tokenizer.tokenize(' '.join(code_tokens))
            tokens = tokenized_tokens 
            token_idx = tokenizer.convert_tokens_to_ids(tokens)
            inputs = torch.tensor(token_idx).unsqueeze(0)
            inputs = inputs.to(args.device)
            label = ' '.join(code['docstring_tokens'])
            labels = tokenizer(label, return_tensors='pt').input_ids.to(args.device)
            outputs = model(input_ids=inputs, labels=labels)
            
        elif args.model == 'codet5p_2b_dec':    
            tokenized_tokens = tokenizer.tokenize(' '.join(code_tokens))
            tokens = tokenized_tokens 
            token_idx = tokenizer.convert_tokens_to_ids(tokens)
            inputs = torch.tensor(token_idx).unsqueeze(0)
            inputs = inputs.to(args.device)
            label = ' '.join(code['code_tokens'])
            labels = tokenizer(label, return_tensors='pt').input_ids.to(args.device)
            outputs = model(input_ids=inputs, labels=labels)
              
        else:
            tokenized_tokens = tokenizer.tokenize(' '.join(code_tokens))
            tokens = [tokenizer.cls_token] + tokenized_tokens + [tokenizer.sep_token]
            token_idx = tokenizer.convert_tokens_to_ids(tokens)
            inputs = torch.tensor(token_idx).unsqueeze(0)
            inputs = inputs.to(args.device)
            

            if args.model in ['codet5', 'codet5_large', 'coderl', 'codet5p_220', 'codet5p_770']:
                label = ' '.join(code['docstring_tokens'])
                labels = tokenizer(label, return_tensors='pt').input_ids.to(args.device)
                outputs = model(input_ids=inputs, labels=labels)
            else:
                outputs = model(inputs)

        if args.model in ['plbart', 'codet5', 'codet5_large', 'codet5p_220', 'codet5p_770']:
            all_hidden_states = merge_hidden_repr(outputs.encoder_hidden_states, tokenized_tokens, code_tokens, special_char=special_char)
        elif args.model == 'coderl':
            hidden_states = outputs[-1]
            all_hidden_states = merge_hidden_repr(hidden_states, tokenized_tokens, code_tokens, special_char=special_char)
        elif args.model in ['codebert', 'graphcodebert', 'unixcoder']:
            start_index = 3 if args.model == 'unixcoder' else 1
            all_hidden_states = merge_hidden_repr(outputs.hidden_states, tokenized_tokens, code_tokens, start_index=start_index,  special_char=special_char)
        
        elif args.model in ['codet5p_2b'] :
            start_index = 0
            try:
                all_hidden_states = merge_hidden_repr(outputs.encoder_hidden_states, tokenized_tokens, code_tokens, start_index=start_index, end_index=None, special_char=special_char)
            except:
                all_hidden_states = None
         
        elif args.model=='codet5p_2b_dec':
            start_index = 0
            try:
                all_hidden_states = merge_hidden_repr(outputs.decoder_hidden_states, tokenized_tokens, code_tokens, start_index=start_index, end_index=None, special_char=special_char)
            except:
                all_hidden_states = None
        elif args.model == 'codegen':
            start_index = 0
            try:
                all_hidden_states = merge_hidden_repr(outputs.hidden_states, tokenized_tokens, code_tokens, start_index=start_index, end_index=None, special_char=special_char)
            except Exception as e:
                print(e)
                all_hidden_states = None
            
        
            
        if all_hidden_states is not None:
            walk_path = []
            curr_depth = 0
            depth = []
            is_code_token = []
            code_token_info = []

            code_string = code['code']
            byte_code = bytes(code_string, 'utf-8')
            tree = parser.parse(byte_code)

            get_lca_info(tree.root_node, walk_path, curr_depth, depth, is_code_token, code_token_info, byte_code)
            n = sum(is_code_token)
            tree_dist_matrix = np.ndarray((n, n))
            lca_matrix = np.ndarray((n, n), dtype='str')

            ast_terminal_tokens = []
            for u, is_terminal in enumerate(is_code_token):
                if is_terminal:
                    ast_terminal_tokens.append(walk_path[u])
                    row_idx = sum(is_code_token[:u])
                    dist_matrix_row = []
                    lca_matrix_row = []

                    for v, is_terminal_again in enumerate(is_code_token):
                        if is_terminal_again:
                            row_node = walk_path[u]
                            col_node = walk_path[v]

                            lca, tree_dist = find_tree_distance(u, v, walk_path, depth)
                            dist_matrix_row.append(tree_dist)
                            lca_matrix_row.append(lca)

                    assert len(dist_matrix_row) == n
                    assert len(lca_matrix_row) == n
                    tree_dist_matrix[row_idx] = dist_matrix_row
                    lca_matrix[row_idx] = lca_matrix_row
            try:
                rows_to_delete, rows_to_merge = rows_to_delete_or_merge(code_token_info, code_tokens, byte_code)

                for rows in rows_to_merge:
                    merging_rows = tree_dist_matrix[rows]
                    minm = merging_rows.min(axis = 0)
                    tree_dist_matrix[rows[0]] = minm

                    for other_rows in rows[1:]:
                        rows_to_delete.append(other_rows)

                tree_dist_matrix = np.delete(tree_dist_matrix, rows_to_delete, 0)
                tree_dist_matrix = np.delete(tree_dist_matrix, rows_to_delete, 1)

                code_token_info_new = []
                for i, info in enumerate(code_token_info):
                    if i not in rows_to_delete:
                        code_token_info_new.append(info)

                assert len(code_token_info_new) == len(code_tokens)
                assert tree_dist_matrix.shape[0] == len(code_tokens)
                assert tree_dist_matrix.shape[1] == len(code_tokens)
                assert all_hidden_states.shape[1] == tree_dist_matrix.shape[0]

                embeddings_dict[code_num] = {}
                embeddings_dict[code_num]['hidden_repr'] = all_hidden_states
                embeddings_dict[code_num]['tree_dist'] = tree_dist_matrix
                embeddings_dict[code_num]['code_token_info'] = code_token_info_new
                embeddings_dict[code_num]['code_file'] = code_file_name

                code_num += 1
            except:
                print('There was an issue when trying to get delete and merge rows.')
        file_name = args.model + '.pkl'

        with open(os.path.join(save_dir, file_name), 'wb') as f:
            pickle.dump(embeddings_dict, f)


if __name__ == '__main__':
    cli = argparse.ArgumentParser()
    cli.add_argument('--model', required=True)
    cli.add_argument('--code_file', default='attention/exp_data/exp_0.jsonl')
    cli.add_argument('--graph_loc')
    cli.add_argument('--num_codes', default=None, type=int)
    cli.add_argument('--save_dir', default='structural_probe')
    cli.add_argument('--exp_name')
    cli.add_argument('--device', default='cuda')
    cli.add_argument('--seed', default=0, type=int)
    cli.add_argument(
        '--lang',
        default='python',
        choices=['python', 'java', 'go', 'javascript'],
    )
    args = cli.parse_args()

    tree_sitter_parser = build_parser(args.lang)
    if args.model == 'codebert':
        save_codebert_embeddings(args, tree_sitter_parser)
    else:
        legacy_dir = args.save_dir
        os.makedirs(legacy_dir, exist_ok=True)
        if args.exp_name:
            legacy_dir = os.path.join(legacy_dir, args.exp_name)
            os.makedirs(legacy_dir, exist_ok=True)
        globals()['parser'] = tree_sitter_parser
        save_word_embeddings(args, legacy_dir)
