"""Small controlled-label and real-cohort DFG density audit for four languages."""

import json
from pathlib import Path

import numpy as np

from dfg_comp import build_parser, get_dfg_adj


REPO = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).with_name("dfg_manual_audit.json")
CORPUS = {
    "python": REPO / "attention" / "exp_data" / "exp_0.jsonl",
    "java": REPO / "attention" / "exp_data" / "final_3000" / "java.jsonl",
    "go": REPO / "attention" / "exp_data" / "final_3000" / "go.jsonl",
    "javascript": REPO / "attention" / "exp_data" / "final_3000" / "javascript.jsonl",
}
CONTROL = {
    "python": "def f(x):\n    y = x\n    z = y + x\n    return z",
    "java": "int f(int x) { int y = x; int z = y + x; return z; }",
    "go": "func f(x int) int { y := x; z := y + x; return z }",
    "javascript": "function f(x) { let y = x; let z = y + x; return z; }",
}


def occurrence(tokens, name, index):
    return [i for i, token in enumerate(tokens) if token == name][index]


def controlled_case(language, parser):
    binary, tokens = get_dfg_adj(CONTROL[language], parser, lang=language)
    typed, typed_tokens = get_dfg_adj(CONTROL[language], parser, lang=language, typed=True)
    if typed_tokens != tokens or not np.array_equal(binary, np.abs(typed)):
        raise AssertionError(f"Binary and signed DFG disagree: {language}")
    required_edges = [("y", 0, "x", 1), ("z", 0, "y", 1), ("z", 0, "x", 2)]
    checked = []
    for source, source_index, dependency, dependency_index in required_edges:
        row = occurrence(tokens, source, source_index)
        column = occurrence(tokens, dependency, dependency_index)
        label = int(typed[row, column])
        if not label:
            raise AssertionError(f"Missing expected edge {source}->{dependency}: {language}")
        checked.append({
            "source": f"{source}[{source_index}]",
            "dependency": f"{dependency}[{dependency_index}]",
            "native_typed_label": label,
        })
    rows, columns = np.nonzero(binary)
    all_edges = [
        f"{tokens[row]}@{row} -> {tokens[column]}@{column} ({int(typed[row, column]):+d})"
        for row, column in zip(rows, columns)
    ]
    return {
        "code": CONTROL[language],
        "num_tokens": len(tokens),
        "num_edges": len(rows),
        "checked_expected_edges": checked,
        "all_native_typed_edges": all_edges,
    }


def sample_density(language, parser, sample_size=24):
    with CORPUS[language].open() as handle:
        records = [json.loads(line) for line in handle]
    if len(records) != 3000:
        raise ValueError(f"Expected 3000 cohort records: {language}")
    indices = sorted(int(i) for i in np.random.default_rng(2026).choice(len(records), sample_size, replace=False))
    samples = []
    failures = []
    for index in indices:
        record = records[index]
        try:
            # These are the actual cleaned cohort tokens used by the project.
            binary, tokens = get_dfg_adj(
                record["code"], parser, lang=language,
                expected_tokens=record["code_tokens"], typed=False,
            )
            signed, signed_tokens = get_dfg_adj(
                record["code"], parser, lang=language,
                expected_tokens=record["code_tokens"], typed=True,
            )
            if tokens != record["code_tokens"] or signed_tokens != tokens:
                raise AssertionError("Token alignment mismatch")
            if not np.array_equal(binary, np.abs(signed)):
                raise AssertionError("Binary/typed edge support differs")
            n = len(tokens)
            edges = int(np.count_nonzero(binary))
            samples.append({
                "cohort_row": int(index),
                "function_name": record.get("func_name"),
                "num_tokens": n,
                "num_edges": edges,
                "edges_per_token": edges / n,
                "directed_pair_density": edges / (n * (n - 1)) if n > 1 else 0,
                "positive_native_labels": int(np.count_nonzero(signed > 0)),
                "negative_native_labels": int(np.count_nonzero(signed < 0)),
            })
        except Exception as exc:
            failures.append({"cohort_row": int(index), "reason": f"{type(exc).__name__}: {exc}"})
    if not samples:
        raise ValueError(f"No real cohort DFGs could be extracted: {language}")
    return {
        "selected_rows": indices,
        "successful_rows": len(samples),
        "failed_rows": len(failures),
        "failures": failures,
        "median_tokens": float(np.median([sample["num_tokens"] for sample in samples])),
        "median_edges_per_token": float(np.median([sample["edges_per_token"] for sample in samples])),
        "mean_edges_per_token": float(np.mean([sample["edges_per_token"] for sample in samples])),
        "median_directed_pair_density": float(np.median([sample["directed_pair_density"] for sample in samples])),
        "empty_graphs": sum(sample["num_edges"] == 0 for sample in samples),
        "samples": samples,
    }


def main():
    result = {
        "method": "Controlled function fixture plus 24 fixed-seed rows from each project's 3000-program attention cohort, using exact dataset token alignment.",
        "scope_warning": "Small, non-representative manual audit; density differences may reflect corpus code length/construct mix and cannot by themselves establish semantic parity.",
        "languages": {},
    }
    for language in CORPUS:
        parser = build_parser(language)
        control = controlled_case(language, parser)
        real = sample_density(language, parser)
        result["languages"][language] = {"controlled_case": control, "real_cohort_sample": real}
        print(f"{language}: fixture edges={control['num_edges']}, "
              f"sample pass={real['successful_rows']}/24, "
              f"median edges/token={real['median_edges_per_token']:.3f}, "
              f"median density={real['median_directed_pair_density']:.4f}")
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
