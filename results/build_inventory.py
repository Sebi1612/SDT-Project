"""Rebuild the final result inventory, manifest, and checksums.

The package intentionally does not duplicate multi-gigabyte feature matrices.
RESULT_INVENTORY.csv points to the authoritative artifacts in the repository.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
VALIDATION = HERE / "validation"
LANGUAGES = ("python", "java", "javascript", "go")
ATTENTION_MODELS = (
    "codebert", "graphcodebert", "unixcoder", "plbart", "codet5", "codet5p_220"
)
STRUCTURAL_MODELS = ("codebert", "graphcodebert", "codet5")
PROBE_TASKS = ("distance", "distance_id", "siblings", "siblings_id", "dfg")
FIELDS = (
    "result_id", "analysis", "language", "model", "task", "layers",
    "cohort_programs", "status", "provenance", "primary_output",
    "validation", "production_script", "paper_comparability", "caveats",
)


def relative(path: Path) -> str:
    return path.resolve().relative_to(REPO.resolve()).as_posix()


def attention_root(language: str, model: str) -> Path:
    if model == "codebert":
        return REPO / "analysis_results" / "attention_final_3000" / language
    return (
        REPO / "analysis_results" / "final_3000_multimodel" / language
        / model / "attention"
    )


def model_root(language: str, model: str) -> Path:
    if model == "codebert":
        return REPO / "analysis_results" / "attention_final_3000" / language
    return REPO / "analysis_results" / "final_3000_multimodel" / language / model


def add(rows: list[dict[str, str]], **values: object) -> None:
    row = {field: "" for field in FIELDS}
    row.update({key: str(value) for key, value in values.items()})
    primary = REPO / row["primary_output"] if row["primary_output"] else None
    validation = REPO / row["validation"] if row["validation"] else None
    if primary is not None and not primary.exists():
        raise FileNotFoundError(primary)
    if validation is not None and not validation.exists():
        raise FileNotFoundError(validation)
    rows.append(row)


def build_inventory() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for language in LANGUAGES:
        for model in ATTENTION_MODELS:
            root = attention_root(language, model)
            validation = ""
            if model != "codebert":
                validation = relative(model_root(language, model) / "final_validation.json")
            elif language == "python":
                validation = relative(root / "python_final_validation.json")
            add(
                rows,
                result_id=f"attention-{language}-{model}", analysis="attention_overlap",
                language=language, model=model, task="AST+DFG",
                layers="0-5" if model == "plbart" else "0-11",
                cohort_programs=3000, status="complete",
                provenance="final-3000 recomputation",
                primary_output=relative(root / "section_3_2_summary.json"),
                validation=validation,
                production_script="attention/run_staged_model_analysis.py",
                paper_comparability="same threshold/head-selection protocol",
                caveats=("PLBART Python reference discrepancy" if model == "plbart" else
                         "language-specific AST/DFG extraction"),
            )

    # Stored Python outputs released by the original paper repository.
    for model in ATTENTION_MODELS:
        ast = REPO / "attention" / "graph_comparision" / "ast" / "exp_0" / f"{model}_layer_0.json"
        dfg = REPO / "attention" / "graph_comparision" / "dfg" / "exp_0" / f"{model}_layer_0.json"
        if not ast.exists() or not dfg.exists():
            raise FileNotFoundError(f"Missing original reference for {model}")
        add(
            rows,
            result_id=f"paper-reference-attention-python-{model}",
            analysis="attention_overlap_reference", language="python", model=model,
            task="AST+DFG", layers="0-5" if model == "plbart" else "0-11",
            cohort_programs=3000, status="complete", provenance="original paper repository",
            primary_output=relative(ast.parent.parent.parent),
            production_script="attention/graph_comp.py; attention/dfg_comp.py",
            paper_comparability="paper source", caveats="aggregate output; legacy files omit coverage metadata",
        )

    for language in LANGUAGES:
        for model in STRUCTURAL_MODELS:
            if language == "python":
                primary = (
                    REPO / "attention" / "graph_comparision" / "similarity"
                    / "exp_0" / model
                )
                validation = ""
                provenance = "original paper repository"
                caveat = "legacy Python output; NetworkX/runtime provenance not recorded"
            else:
                validation_path = (
                    REPO / "analysis_results" / "ged_final_3000" / "validation"
                    / language / model / "ged_validation.json"
                )
                info = json.loads(validation_path.read_text())
                primary = Path(info["results_directory"])
                validation = relative(validation_path)
                provenance = "multilingual final-3000 run"
                caveat = "legacy GED mode; library version differs from stored Python reference"
            add(
                rows,
                result_id=f"ged-{language}-{model}", analysis="graph_edit_distance",
                language=language, model=model, task="AST+AST-no-identifiers+DFG",
                layers="0-11", cohort_programs=3000, status="complete",
                provenance=provenance, primary_output=relative(primary),
                validation=validation, production_script="attention/run_requested_ged_queue.py",
                paper_comparability="paper-compatible legacy GED mode", caveats=caveat,
            )

    for language in LANGUAGES:
        for model in STRUCTURAL_MODELS:
            for task in PROBE_TASKS:
                if language == "python":
                    primary = REPO / "DirectProbe" / "results" / task / model
                    provenance = "original paper repository"
                    validation = ""
                    caveat = "original feature matrices are not published; pair-level 80:20 split"
                else:
                    primary = (
                        REPO / "DirectProbe" / "final_3000" / "results"
                        / language / task / model
                    )
                    provenance = "multilingual final-3000 run"
                    validation = relative(
                        REPO / "DirectProbe" / "final_3000"
                        / "requested_probing_parallel_queue_manifest.json"
                    )
                    caveat = "pair-level split; source programs overlap between train and test"
                add(
                    rows,
                    result_id=f"probe-{language}-{model}-{task}", analysis="DirectProbe",
                    language=language, model=model, task=task, layers="5,9,12",
                    cohort_programs=3000, status="complete", provenance=provenance,
                    primary_output=relative(primary), validation=validation,
                    production_script="DirectProbe/run_requested_probe_queue.py",
                    paper_comparability="same tasks, pair construction, and layer convention",
                    caveats=caveat,
                )

    for language in LANGUAGES:
        for model in ATTENTION_MODELS:
            if model == "codebert":
                tsne_root = (
                    REPO / "analysis_results" / language / "tsne" / "final_3000"
                )
                caveat = (
                    "qualitative; main paper-style comparison; independent projections "
                    "and different programs across languages"
                )
            else:
                tsne_root = model_root(language, model) / "hidden_tsne"
                caveat = (
                    "qualitative supplemental output; not used in the main CodeBERT-only "
                    "paper-style figure"
                )
            for task, subdir, manifest in (
                ("token types", "token_types", "token_type_tsne_manifest.json"),
                ("AST/hidden distances", "distances", "distance_tsne_manifest.json"),
            ):
                path = tsne_root / subdir / manifest
                add(
                    rows,
                    result_id=f"tsne-{language}-{model}-{subdir}", analysis="t-SNE",
                    language=language, model=model, task=task, layers="5",
                    cohort_programs=3000, status="complete",
                    provenance="final-3000 recomputation",
                    primary_output=relative(path),
                    production_script="attention/hidden_tsne.py",
                    paper_comparability="paper visualization design",
                    caveats=caveat,
                )

    for name, output in (
        ("program-disjoint-probe-control", "probe_split_audit.json"),
        ("PLBART-reference-diagnosis", "plbart_reference_audit.json"),
        ("manual-DFG-density-audit", "dfg_manual_audit.json"),
    ):
        path = VALIDATION / output
        add(
            rows, result_id=f"validation-{name}", analysis="validation audit",
            language="all", model="selected", task=name, layers="12 or applicable",
            cohort_programs="see audit", status="complete", provenance="project validation",
            primary_output=relative(path),
            validation=relative(VALIDATION / "README.md"),
            production_script=relative(path.with_suffix(".py")) if path.with_suffix(".py").exists() else "",
            paper_comparability="diagnostic only",
            caveats="see validation_audit/README.md for interpretation boundaries",
        )
    return rows


def write_checksums() -> None:
    paths = sorted(
        path for path in HERE.rglob("*")
        if path.is_file() and path.name not in {"checksums.sha256"}
        and "__pycache__" not in path.parts
    )
    lines = []
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(HERE).as_posix()}")
    (HERE / "checksums.sha256").write_text("\n".join(lines) + "\n")


def git_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()


def main() -> None:
    rows = build_inventory()
    with (HERE / "RESULT_INVENTORY.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["analysis"]] = counts.get(row["analysis"], 0) + 1
    manifest = {
        "status": "complete",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(),
        "inventory_rows": len(rows),
        "inventory_counts": counts,
        "design": "final result index; authoritative artifacts remain at inventory paths",
    }
    (HERE / "PACKAGE_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    write_checksums()
    print(f"Built final result index with {len(rows)} result groups")


if __name__ == "__main__":
    main()
