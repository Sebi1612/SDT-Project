"""Validate the retained final artifact without rerunning expensive analyses."""

from __future__ import annotations

import csv
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
LANGUAGES = ("python", "java", "javascript", "go")
MODELS = ("codebert", "graphcodebert", "unixcoder", "plbart", "codet5", "codet5p_220")
PROBE_MODELS = ("codebert", "graphcodebert", "codet5")
PROBE_TASKS = ("distance", "distance_id", "siblings", "siblings_id", "dfg")
PROBE_LAYERS = (5, 9, 12)
REQUIRED_PROBE_OUTPUTS = ("clusters.txt", "prediction.txt", "dis.txt", "log.txt")


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def attention_root(language: str, model: str) -> Path:
    if model == "codebert":
        return REPO / "analysis_results" / "attention_final_3000" / language
    return (
        REPO / "analysis_results" / "final_3000_multimodel" / language
        / model / "attention"
    )


def main() -> None:
    cohort_paths = {
        "python": REPO / "attention" / "exp_data" / "exp_0.jsonl",
        **{
            language: REPO / "attention" / "exp_data" / "final_3000" / f"{language}.jsonl"
            for language in LANGUAGES[1:]
        },
    }
    for language, path in cohort_paths.items():
        require(sum(1 for _ in path.open()) == 3000, f"invalid cohort: {language}")

    attention_summaries = []
    for language in LANGUAGES:
        for model in MODELS:
            path = attention_root(language, model) / "section_3_2_summary.json"
            summary = load(path)
            require(summary.get("status") == "complete", f"incomplete attention: {path}")
            require(summary.get("expected_programs") == 3000, f"wrong coverage: {path}")
            attention_summaries.append(path)

    ged_validations = list(
        (REPO / "analysis_results" / "ged_final_3000" / "validation").glob(
            "*/*/ged_validation.json"
        )
    )
    require(len(ged_validations) == 9, "expected nine multilingual GED validations")
    for path in ged_validations:
        record = load(path)
        require(record.get("status") == "complete" and record.get("valid"), f"invalid GED: {path}")
        require(record.get("layers") == list(range(12)), f"wrong GED layers: {path}")
        result = Path(record["results_directory"])
        require(result.is_dir(), f"missing GED result directory: {result}")
        require(
            len(list(result.glob("layer_*_threshold_0.05.json"))) == 12,
            f"missing GED layer files: {result}",
        )

    dp_root = REPO / "DirectProbe" / "final_3000"
    datasets = []
    result_configs = []
    for language in LANGUAGES[1:]:
        for model in PROBE_MODELS:
            for task in PROBE_TASKS:
                dataset = dp_root / "data" / language / task / model / "dataset_manifest.json"
                record = load(dataset)
                require(record.get("status") == "complete", f"incomplete probe data: {dataset}")
                datasets.append(dataset)
                for layer in PROBE_LAYERS:
                    config = dp_root / "config_files" / language / task / f"config_{model}_{layer}.ini"
                    result = dp_root / "results" / language / task / model / str(layer)
                    require(config.is_file(), f"missing probe config: {config}")
                    require(
                        all((result / name).is_file() for name in REQUIRED_PROBE_OUTPUTS),
                        f"missing probe result: {result}",
                    )
                    result_configs.append(config)

    tsne_manifests = list((REPO / "analysis_results").glob("*/tsne/final_3000/*/*tsne_manifest.json"))
    tsne_manifests += list(
        (REPO / "analysis_results" / "final_3000_multimodel").glob(
            "*/*/hidden_tsne/*/*tsne_manifest.json"
        )
    )
    require(len(tsne_manifests) == 48, "expected 48 t-SNE result groups")

    with (HERE / "RESULT_INVENTORY.csv").open(newline="") as handle:
        inventory = list(csv.DictReader(handle))
    require(len(inventory) == 153, "expected 153 inventory rows")
    for row in inventory:
        primary = REPO / row["primary_output"]
        require(primary.exists(), f"missing inventory output: {primary}")
        if row["validation"]:
            validation = REPO / row["validation"]
            require(validation.exists(), f"missing inventory validation: {validation}")

    report = HERE / "report" / "four_language_results.pdf"
    require(report.is_file() and report.stat().st_size > 0, "missing final PDF")
    print(
        "archive complete: "
        f"{len(cohort_paths)} cohorts, {len(attention_summaries)} attention summaries, "
        f"{len(ged_validations)} GED cohorts, {len(datasets)} probe datasets, "
        f"{len(result_configs)} probe configurations, {len(tsne_manifests)} t-SNE groups, "
        f"{len(inventory)} inventory rows"
    )


if __name__ == "__main__":
    main()
