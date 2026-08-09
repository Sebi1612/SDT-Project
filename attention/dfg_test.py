from dfg_comp import build_parser, get_dfg_adj

def main():
    test_cases = {
        'java': "public void test() { int a = 5; int b = a; }",
        'python': "def test():\n    a = 5\n    b = a",
        'go': "func test() { a := 5; b := a }",
        'javascript': "function test() { let a = 5; let b = a; }"
    }

    failures = []
    for lang, code_string in test_cases.items():
        print(f"\n--- Running DFG Sanity Check for {lang.upper()} ---")
        try:
            parser = build_parser(lang)
            # Note: For now, Go and JS will fall back to basic tokenization
            # if explicit DFG logic isn't mapped, but the parser will build!
            dfg_adj, dfg_tokens = get_dfg_adj(code_string, parser, lang=lang)
            print("✅ Extraction Successful!")
            print(f"   -> Tokens found: {len(dfg_tokens)}")
            print(f"   -> Matrix shape: {dfg_adj.shape}")
            print(f"   -> Total edges: {dfg_adj.sum()}")
        except Exception as e:
            print(f"❌ Extraction failed for {lang}: {type(e).__name__} - {e}")
            failures.append(lang)

    if failures:
        raise SystemExit(
            f"DFG sanity checks failed for: {', '.join(failures)}"
        )

if __name__ == "__main__":
    main()
