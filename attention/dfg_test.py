import os
from tree_sitter import Language, Parser

# Import your existing DFG extraction function
from dfg_comp import get_dfg_adj

def build_java_parser():
    lang = 'java'
    grammar_repo = f'tree-sitter-{lang}'
    language_library = os.path.join('build', f'my-languages-{lang}.so')
    
    if not os.path.exists('build'):
        os.mkdir('build')
    if not os.path.exists(language_library):
        Language.build_library(language_library, [grammar_repo])
        
    language = Language(language_library, lang)
    parser = Parser()
    parser.set_language(language)
    return parser

def main():
    print("Building Java Parser...")
    parser = build_java_parser()
    
    # The ultimate trivial test case
    code_string = "public void test() { int a = 5; int b = a; }"
    
    print("\n--- Running DFG Sanity Check ---")
    print(f"Code: {code_string}\n")
    
    try:
        # Extract the DFG
        dfg_adj, dfg_tokens = get_dfg_adj(code_string, parser, lang='java')
        
        print("✅ Extraction Successful!")
        print(f"   -> Tokens found: {len(dfg_tokens)}")
        print(f"   -> Matrix shape: {dfg_adj.shape}")
        print(f"   -> Total edges (data flows): {dfg_adj.sum()}")
        print(f"   -> Token list: {dfg_tokens}")
        
    except Exception as e:
        print(f"❌ Extraction failed: {e}")

if __name__ == "__main__":
    main()