"""Cheap, program-disjoint linear-probe control on retained layer-12 features.

This is deliberately NOT a rerun of the Gurobi-based DirectProbe classifier.
The same fixed Ridge classifier is evaluated on the original pair split and a
program-disjoint split, so the difference estimates split sensitivity.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler


REPO = Path(__file__).resolve().parents[2]
LANGUAGES = ("java", "go", "javascript")
MODELS = ("codebert", "graphcodebert", "codet5")
TASKS = ("distance", "distance_id", "siblings", "siblings_id", "dfg")


def score(x, y, train, test):
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x[train])
    x_test = scaler.transform(x[test])
    classifier = RidgeClassifier(alpha=1.0, solver="lsqr")
    classifier.fit(x_train, y[train])
    predicted = classifier.predict(x_test)
    return {
        "accuracy": float(accuracy_score(y[test], predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y[test], predicted)),
        "test_label_counts": {label: int(np.sum(y[test] == label)) for label in sorted(set(y))},
    }


def audit_one(language, model, task, layer):
    root = REPO / "DirectProbe" / "final_3000" / "data" / language / task / model
    manifest = json.loads((root / "dataset_manifest.json").read_text())
    records = manifest["pairs"]
    y = np.asarray([record["label"] for record in records])
    groups = np.asarray([record["artifact"] for record in records])
    original_train = np.asarray([i for i, record in enumerate(records) if record["split"] == "train"])
    original_test = np.asarray([i for i, record in enumerate(records) if record["split"] == "test"])
    train_path = root / "embeddings" / "layers" / "train" / f"{layer}.txt"
    test_path = root / "embeddings" / "layers" / "test" / f"{layer}.txt"
    x_train = np.loadtxt(train_path, dtype=np.float32, ndmin=2)
    x_test = np.loadtxt(test_path, dtype=np.float32, ndmin=2)
    if len(x_train) != len(original_train) or len(x_test) != len(original_test):
        raise ValueError(f"Feature/manifest row count mismatch: {root}")
    x = np.empty((len(records), x_train.shape[1]), dtype=np.float32)
    x[original_train] = x_train
    x[original_test] = x_test
    if not np.isfinite(x).all():
        raise ValueError(f"Nonfinite features: {root}")
    shared = set(groups[original_train]) & set(groups[original_test])

    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0).split(x, y, groups))
    labels = sorted(set(y))
    def fold_cost(fold):
        train, test = fold
        if set(y[train]) != set(labels) or set(y[test]) != set(labels):
            return float("inf")
        target = np.asarray([np.mean(y == label) for label in labels])
        actual = np.asarray([np.mean(y[test] == label) for label in labels])
        return abs(len(test) / len(y) - .2) + np.max(abs(actual - target))
    valid_folds = [fold for fold in folds if fold_cost(fold) != float("inf")]
    if not valid_folds:
        raise ValueError(f"No group fold contained all labels: {root}")
    group_train, group_test = min(valid_folds, key=fold_cost)
    if set(groups[group_train]) & set(groups[group_test]):
        raise AssertionError("Program-disjoint split leaked an artifact")
    pair_score = score(x, y, original_train, original_test)
    group_score = score(x, y, group_train, group_test)
    group_fold_scores = [score(x, y, train, test) for train, test in valid_folds]
    group_mean_accuracy = float(np.mean([fold["accuracy"] for fold in group_fold_scores]))
    group_mean_balanced_accuracy = float(np.mean([fold["balanced_accuracy"] for fold in group_fold_scores]))
    return {
        "language": language,
        "model": model,
        "task": task,
        "layer": layer,
        "num_pairs": len(records),
        "num_programs": len(set(groups)),
        "pair_split_train_programs": len(set(groups[original_train])),
        "pair_split_test_programs": len(set(groups[original_test])),
        "pair_split_shared_programs": len(shared),
        "pair_split_shared_test_fraction": len(shared) / len(set(groups[original_test])),
        "group_split_train_programs": len(set(groups[group_train])),
        "group_split_test_programs": len(set(groups[group_test])),
        "group_split_test_pairs": len(group_test),
        "pair_split": pair_score,
        "program_disjoint_split": group_score,
        "program_disjoint_five_fold": {
            "num_valid_folds": len(valid_folds),
            "accuracy_mean": group_mean_accuracy,
            "accuracy_std": float(np.std([fold["accuracy"] for fold in group_fold_scores])),
            "balanced_accuracy_mean": group_mean_balanced_accuracy,
            "balanced_accuracy_std": float(np.std([fold["balanced_accuracy"] for fold in group_fold_scores])),
            "fold_scores": group_fold_scores,
        },
        "accuracy_delta_disjoint_minus_pair": group_score["accuracy"] - pair_score["accuracy"],
        "balanced_accuracy_delta_disjoint_minus_pair": group_score["balanced_accuracy"] - pair_score["balanced_accuracy"],
        "accuracy_delta_five_fold_minus_pair": group_mean_accuracy - pair_score["accuracy"],
        "balanced_accuracy_delta_five_fold_minus_pair": group_mean_balanced_accuracy - pair_score["balanced_accuracy"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--languages", nargs="+", choices=LANGUAGES, default=list(LANGUAGES))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("probe_split_audit.json"))
    args = parser.parse_args()
    results = []
    for language in args.languages:
        for model in args.models:
            for task in args.tasks:
                result = audit_one(language, model, task, args.layer)
                print(f"{language}/{model}/{task}: pair={result['pair_split']['accuracy']:.3f}, "
                      f"disjoint={result['program_disjoint_split']['accuracy']:.3f}, "
                      f"shared-test-programs={result['pair_split_shared_test_fraction']:.1%}", flush=True)
                results.append(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "method": "StandardScaler + RidgeClassifier(alpha=1, solver=lsqr), fixed across both splits",
        "warning": "Diagnostic linear probe, not the paper's Gurobi DirectProbe; pair and program splits have different test rows.",
        "results": results,
    }, indent=2) + "\n")
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "language", "model", "task", "layer", "num_pairs", "num_programs",
            "pair_split_shared_test_fraction", "pair_accuracy", "program_disjoint_accuracy", "five_fold_disjoint_accuracy_mean",
            "pair_balanced_accuracy", "program_disjoint_balanced_accuracy", "five_fold_disjoint_balanced_accuracy_mean",
            "accuracy_delta_disjoint_minus_pair", "balanced_accuracy_delta_disjoint_minus_pair",
            "accuracy_delta_five_fold_minus_pair", "balanced_accuracy_delta_five_fold_minus_pair",
        ])
        writer.writeheader()
        for result in results:
            writer.writerow({
                **{key: result[key] for key in ("language", "model", "task", "layer", "num_pairs", "num_programs", "pair_split_shared_test_fraction", "accuracy_delta_disjoint_minus_pair", "balanced_accuracy_delta_disjoint_minus_pair", "accuracy_delta_five_fold_minus_pair", "balanced_accuracy_delta_five_fold_minus_pair")},
                "pair_accuracy": result["pair_split"]["accuracy"],
                "program_disjoint_accuracy": result["program_disjoint_split"]["accuracy"],
                "five_fold_disjoint_accuracy_mean": result["program_disjoint_five_fold"]["accuracy_mean"],
                "pair_balanced_accuracy": result["pair_split"]["balanced_accuracy"],
                "program_disjoint_balanced_accuracy": result["program_disjoint_split"]["balanced_accuracy"],
                "five_fold_disjoint_balanced_accuracy_mean": result["program_disjoint_five_fold"]["balanced_accuracy_mean"],
            })


if __name__ == "__main__":
    main()
