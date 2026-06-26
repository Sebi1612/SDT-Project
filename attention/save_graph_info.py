# pyright: reportMissingImports=false, reportMissingTypeStubs=false
import os
import argparse
import pickle

import torch
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
)
from graph_utils import get_ast_tokens_and_prog_graphs, tokens_to_graph, traverse_node
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM 


def build_parser(lang, grammar_repo):
    language_library = os.path.join('build', f'my-languages-{lang}.so')
    if not os.path.exists(grammar_repo):
        raise FileNotFoundError(
            f"Tree-sitter grammar repository not found: {grammar_repo}. "
            f"Clone the grammar repo for {lang} and pass it with --grammar_repo."
        )

    if not os.path.exists(language_library):
        Language.build_library(language_library, [grammar_repo])

    language = Language(language_library, lang)
    parser = Parser()
    parser.set_language(language)
    return parser

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
    
    function_args = dict()

    codes = load_codesearchnet(args.code_file, args.num_codes)
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
        device = "cuda"
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
        device = "cuda"
        model.to(device)
        
    if args.model == 'codegen':
        tokenizer = AutoTokenizer.from_pretrained(model_version['codegen'])
        model = AutoModelForCausalLM.from_pretrained(model_version['codegen'], output_attentions = True, trust_remote_code=True) 
        device = "cuda"
        model.to(device)  
    
    for code in tqdm(codes):
        filename = code['code_file']
    
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

        output = function_map[args.model](**function_args)
        if output is not None:
            attention, tokens = output[0],output[1] 

            code_tokens = code['code_tokens']
            code_string = code['code']
            byte_code = bytes(code_string, 'utf-8')
            tree = parser.parse(byte_code)
            root_node = tree.root_node

            collected_tokens = []
            traverse_node(root_node, collected_tokens,byte_code)
            try:
                ast_info, _, is_error = get_ast_tokens_and_prog_graphs(collected_tokens, code_tokens, tokens, byte_code, (0,0))

                ast_tokens = []
                for info in ast_info:
                    ast_tokens.append(info['token'])

                ast_graph = tokens_to_graph(ast_info)

                save_dir = args.save_dir
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)

                if args.exp_name is not None:
                    save_dir = os.path.join(save_dir, args.exp_name)
                    if not os.path.exists(save_dir):
                        os.mkdir(save_dir)

                save_dir = os.path.join(save_dir, args.model)
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)

                graph_file = os.path.join(save_dir, filename+'.pkl')

                data_to_write = {
                    'file_name' : filename,
                    'model_tokens' : tokens,
                    'code_tokens' : code_tokens,
                    'ast_tokens' : ast_tokens,
                    'model_graphs' : attention,
                    'ast_graph' : ast_graph,
                }

                with open(graph_file, 'wb') as f:
                    pickle.dump(data_to_write, f) 
            except Exception:
                print('There was an issue while getting ast graph')


         
    

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required =True)
    parser.add_argument('--code_file', default = 'exp_data/exp_0.jsonl')
    parser.add_argument('--num_codes', default = None, type = int)
    parser.add_argument('--save_dir', default = 'graph_info')
    parser.add_argument('--exp_name', required = False)
    parser.add_argument('--random', action = 'store_true')
    parser.add_argument('--lang', default = 'python', choices = ['python', 'java'])
    parser.add_argument('--grammar_repo', default = None)
    
    args = parser.parse_args()

    default_grammar_repos = {
        'python': 'tree-sitter-python',
        'java': 'tree-sitter-java',
    }
    grammar_repo = args.grammar_repo or default_grammar_repos[args.lang]
    parser = build_parser(args.lang, grammar_repo)
    
    save_attention_and_ast(args, parser)
    
    
    
    
        

    


