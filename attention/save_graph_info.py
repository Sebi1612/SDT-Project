# pyright: reportMissingImports=false, reportMissingTypeStubs=false
import os
import argparse
import hashlib
import importlib.metadata
import json
import pickle
import random
import torch
import transformers
from tqdm import tqdm
from tree_sitter import Language, Parser

from utils import load_codesearchnet
from get_attention import (
    get_attention_codeT5,
    get_attention_codeT5p,
    get_attention_codeT5p_2b,
    get_attention_codeT5p_2b_dec,
    get_attention_codegen,
    get_attention_codebert,
    get_attention_graphcodebert,
    get_attention_plbart,
    get_attention_uniXcoder,
    load_codebert_attention_model,
)
from graph_utils import get_ast_tokens_and_prog_graphs, tokens_to_graph, traverse_node
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM


def build_parser(lang, grammar_repo):
    if not os.path.exists(grammar_repo):
        raise FileNotFoundError(
            f"Tree-sitter grammar repository not found: {grammar_repo}. "
            f"Clone the grammar repo for {lang} and pass it with --grammar_repo."
        )

    language_library = os.path.join('build', f'my-languages-{lang}.so')
    language_libraries = [language_library]
    if lang == 'python':
        # This repository's established Python pipeline uses an ABI-14
        # grammar. The checked-out tree-sitter-python grammar currently emits
        # ABI 15, which tree_sitter 0.20 cannot load.
        language_libraries.insert(
            0, os.path.join('attention', 'build', 'my-languages.so')
        )

    load_errors = []
    for candidate in language_libraries:
        if not os.path.exists(candidate):
            continue
        try:
            language = Language(candidate, lang)
            parser = Parser()
            parser.set_language(language)
            return parser, candidate
        except ValueError as exc:
            load_errors.append(f'{candidate}: {exc}')

    if not os.path.exists(language_library):
        Language.build_library(language_library, [grammar_repo])

    try:
        language = Language(language_library, lang)
        parser = Parser()
        parser.set_language(language)
        return parser, language_library
    except ValueError as exc:
        load_errors.append(f'{language_library}: {exc}')
        raise RuntimeError(
            f'No compatible tree-sitter library found for {lang}: '
            + '; '.join(load_errors)
        ) from exc

def save_attention_and_ast(args, parser):
    print(f'saving attention maps for {args.model} in directory {args.save_dir}')

    function_map = {
    	"codebert": get_attention_codebert,
    	"graphcodebert": get_attention_graphcodebert,
    	"codet5": get_attention_codeT5,
    	"plbart": get_attention_plbart,
    	"unixcoder": get_attention_uniXcoder,
    	"codet5_large": get_attention_codeT5,
    	"coderl": get_attention_codeT5,
    	"coderl_train_critic": get_attention_codeT5,
    	"coderl_infer_critic": get_attention_codeT5,
    	"codet5p_220": get_attention_codeT5p,
    	"codet5p_770": get_attention_codeT5p,
    	"codet5_musu": get_attention_codeT5p,
    	"codet5_lntp": get_attention_codeT5p,
    	"codet5p_2b": get_attention_codeT5p_2b,
        "codegen": get_attention_codegen,
        "codet5p_2b_dec": get_attention_codeT5p_2b_dec
    }

    if args.model not in function_map.keys():
        raise Exception(f'Wrong model name: {args.model}')

    non_default_models = ["codet5_large", "coderl", "coderl_train_critic", "coderl_infer_critic", "codet5p_770", "codet5_musu", "codet5_lntp"]
    model_version = {
    	"codet5_large" : "Salesforce/codet5-large-ntp-py",
    	"coderl" : "coderl_weights/coderl/",
    	"codet5p_770" : "Salesforce/codet5p-770m",
    	"codet5_musu": "Salesforce/codet5-base-multi-sum",
    	"codet5_lntp": "Salesforce/codet5-large-ntp-py",
    	"codet5p_2b": "Salesforce/codet5p-2b",
        "codegen": "Salesforce/codegen2-3_7B",
        "codet5p_2b_dec": "Salesforce/codet5p-2b"

    }

    function_args = {'device': args.device}

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    codes = load_codesearchnet(args.code_file, args.num_codes)
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError(
            f"Requested --device {args.device!r}, but CUDA is not available. "
            "Use --device cpu or run on a CUDA-enabled host."
        )

    if args.model == 'codebert':
        model, tokenizer = load_codebert_attention_model(
            args.device,
            random=args.random,
        )
        function_args['model'] = model
        function_args['tokenizer'] = tokenizer
        function_args['device'] = args.device
    codet5_big_models = ['codet5p_2b']
    if args.model in codet5_big_models:
        tokenizer = AutoTokenizer.from_pretrained(model_version['codet5p_2b'])
        model = AutoModelForSeq2SeqLM.from_pretrained(model_version['codet5p_2b'],
                                              torch_dtype=torch.float16,
                                              trust_remote_code=True, output_attentions=True)

        model.config.decoder_start_token_id = tokenizer.bos_token_id
        model.config.pad_token_id = tokenizer.eos_token_id
        model.config.encoder.output_attentions = True
        model.config.encoder.output_hidden_states = True
        device = args.device
        model.to(device)

    codet5_dec = ['codet5p_2b_dec']
    if args.model in codet5_dec:
        tokenizer = AutoTokenizer.from_pretrained(model_version['codet5p_2b_dec'])
        model = AutoModelForSeq2SeqLM.from_pretrained(model_version['codet5p_2b_dec'],
                                              torch_dtype=torch.float16,
                                              trust_remote_code=True)

        model.config.add_cross_attention=False
        model.config.output_attentions = True
        model.config.output_hidden_states = True
        model.config.decoder.add_cross_attention = False
        model.config.decoder.output_attentions = True
        model.config.decoder.output_hidden_states = True
        model.config.decoder_start_token_id = tokenizer.bos_token_id
        model.config.pad_token_id = tokenizer.eos_token_id
        device = args.device
        model.to(device)

    if args.model == 'codegen':
        tokenizer = AutoTokenizer.from_pretrained(model_version['codegen'])
        model = AutoModelForCausalLM.from_pretrained(model_version['codegen'], output_attentions = True, trust_remote_code=True)
        device = args.device
        model.to(device)

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    if args.exp_name is not None:
        save_dir = os.path.join(save_dir, args.exp_name)
        os.makedirs(save_dir, exist_ok=True)

    save_dir = os.path.join(save_dir, args.model)
    os.makedirs(save_dir, exist_ok=True)

    # The manifest makes a graph directory represent the latest run even when it
    # still contains artifacts from an earlier run.
    with open(args.code_file, 'rb') as code_file_handle:
        code_file_sha256 = hashlib.sha256(code_file_handle.read()).hexdigest()
    with open(args.parser_library, 'rb') as parser_library_handle:
        parser_library_sha256 = hashlib.sha256(
            parser_library_handle.read()
        ).hexdigest()

    manifest = {
        'status': 'in_progress',
        'code_file': os.path.abspath(args.code_file),
        'code_file_sha256': code_file_sha256,
        'lang': args.lang,
        'model': args.model,
        'model_version': 'microsoft/codebert-base' if args.model == 'codebert' else None,
        'random_model': args.random,
        'seed': args.seed,
        'device': args.device,
        'torch_version': torch.__version__,
        'transformers_version': transformers.__version__,
        'tree_sitter_version': importlib.metadata.version('tree-sitter'),
        'tree_sitter_language_library': os.path.abspath(args.parser_library),
        'tree_sitter_language_library_sha256': parser_library_sha256,
        'grammar_repo': os.path.abspath(args.grammar_repo),
        'inference_mode': True,
        'requested_num_codes': args.num_codes,
        'selected_num_codes': len(codes),
        'artifacts': [],
        'failures': [],
    }

    manifest_path = os.path.join(save_dir, 'graph_manifest.json')

    def write_manifest():
        temporary_path = manifest_path + '.tmp'
        with open(temporary_path, 'w') as manifest_file:
            json.dump(manifest, manifest_file, indent=2)
        os.replace(temporary_path, manifest_path)

    write_manifest()

    for sample_index, code in enumerate(tqdm(codes)):
        filename = code['code_file']
        code_string = code['code']
        byte_code = bytes(code_string, 'utf-8')
        tree = parser.parse(byte_code)
        if tree.root_node.has_error:
            reason = 'Tree-sitter reported a parse error'
            manifest['failures'].append({
                'sample_index': sample_index,
                'source_index': code.get('pilot_source_index', sample_index),
                'file_name': filename,
                'stage': 'parse',
                'reason': reason,
            })
            print(
                f"There was an issue while getting ast graph for "
                f"sample {sample_index} ({filename}): {reason}"
            )
            write_manifest()
            continue

        function_args['data'] = code
        function_args['random'] = args.random
        if args.model in non_default_models:
            function_args["model_version"] = model_version[args.model]

        elif args.model in codet5_big_models:
            function_args["model"] = model
            function_args["tokenizer"] = tokenizer

        elif args.model in codet5_dec:
            function_args["model"] = model
            function_args["tokenizer"] = tokenizer

        elif args.model == 'codegen':
            function_args["model"] = model
            function_args["tokenizer"] = tokenizer

        try:
            output = function_map[args.model](**function_args)
        except Exception as exc:
            manifest['failures'].append({
                'sample_index': sample_index,
                'source_index': code.get('pilot_source_index', sample_index),
                'file_name': filename,
                'stage': 'attention',
                'reason': f'{type(exc).__name__}: {exc}',
            })
            print(
                f"There was an issue while getting attention for sample "
                f"{sample_index} ({filename}): {type(exc).__name__}: {exc}"
            )
            write_manifest()
            continue
        if output is not None:
            attention, tokens = output[0],output[1]

            code_tokens = code['code_tokens']
            root_node = tree.root_node

            collected_tokens = []
            traverse_node(
                root_node,
                collected_tokens,
                byte_code,
                include_comments=False,
            )
            try:
                ast_info, _, is_error = get_ast_tokens_and_prog_graphs(collected_tokens, code_tokens, tokens, byte_code, (0,0))
                if is_error:
                    raise ValueError('AST tokens do not exactly match dataset tokens')

                ast_tokens = []
                for info in ast_info:
                    ast_tokens.append(info['token'])

                ast_graph = tokens_to_graph(ast_info)

                # code_file is not unique in CodeSearchNet. Prefixing the
                # position in this sampled run prevents one function from
                # overwriting another function with the same code_file value.
                artifact_name = f'{sample_index:06d}_{filename}.pkl'
                graph_file = os.path.join(save_dir, artifact_name)

                data_to_write = {
                    'file_name' : filename,
                    'sample_index': sample_index,
                    'source_index': code.get('pilot_source_index', sample_index),
                    'code': code_string,
                    'lang': args.lang,
                    'model_tokens' : tokens,
                    'code_tokens' : code_tokens,
                    'ast_tokens' : ast_tokens,
                    'model_graphs' : attention,
                    'ast_graph' : ast_graph,
                }

                with open(graph_file, 'wb') as f:
                    pickle.dump(data_to_write, f)

                manifest['artifacts'].append(artifact_name)
            except Exception as exc:
                manifest['failures'].append({
                    'sample_index': sample_index,
                    'source_index': code.get('pilot_source_index', sample_index),
                    'file_name': filename,
                    'stage': 'ast',
                    'reason': f'{type(exc).__name__}: {exc}',
                })
                print(
                    f"There was an issue while getting ast graph for "
                    f"sample {sample_index} ({filename}): "
                    f"{type(exc).__name__}: {exc}"
                )
        else:
            manifest['failures'].append({
                'sample_index': sample_index,
                'source_index': code.get('pilot_source_index', sample_index),
                'file_name': filename,
                'stage': 'attention',
                'reason': 'Attention extractor returned no output',
            })

        write_manifest()

    manifest['status'] = 'complete'
    write_manifest()

    print(
        f"Saved {len(manifest['artifacts'])}/{len(codes)} graph artifacts "
        f"and run manifest {manifest_path}"
    )






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required =True)
    parser.add_argument('--code_file', default = 'exp_data/exp_0.jsonl')
    parser.add_argument('--num_codes', default = None, type = int)
    parser.add_argument('--save_dir', default = 'graph_info')
    parser.add_argument('--exp_name', required = False)
    parser.add_argument('--random', action = 'store_true')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--lang', default = 'python', choices = ['python', 'java', 'go', 'javascript'])
    parser.add_argument('--grammar_repo', default = None)

    args = parser.parse_args()
    #args.save_dir = os.path.join(args.save_dir, args.lang)
    #if hasattr(args, 'graph_loc'):
     #   args.graph_loc = os.path.join(args.graph_loc, args.lang)

    default_grammar_repos = {
    'python': 'tree-sitter-python',
    'java': 'tree-sitter-java',
    'go': 'tree-sitter-go',
    'javascript': 'tree-sitter-javascript',
}
    grammar_repo = args.grammar_repo or default_grammar_repos[args.lang]
    parser, parser_library = build_parser(args.lang, grammar_repo)
    args.parser_library = parser_library
    args.grammar_repo = grammar_repo

    save_attention_and_ast(args, parser)
