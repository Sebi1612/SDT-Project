"""Build four-language comparative figures.

Run with the project's attention environment; no model inference is performed here.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
ANALYSIS = REPO / "analysis_results"
OUTPUT = HERE / "report"
FIGURES = OUTPUT / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)
LANGUAGES = ("python", "java", "javascript", "go")
LANG_LABELS = ("Python", "Java", "JavaScript", "Go")
MODELS = ("codebert", "graphcodebert", "unixcoder", "plbart", "codet5", "codet5p_220")
MODEL_LABELS = ("CodeBERT", "GraphCodeBERT", "UniXcoder", "PLBART", "CodeT5", "CodeT5+220M")
PROBE_MODELS = ("codebert", "graphcodebert", "codet5")
GED_MODELS = PROBE_MODELS  # only these three models were in the multilingual GED schedule
GRAPHS = ("ast", "dfg")
GED_GRAPHS = ("ast", "ast_wo_identifiers", "dfg")
GED_LABELS = ("Complete AST", "Non-identifier AST", "DFG")
COLORS = plt.get_cmap("tab10").colors


def attention_root(language: str, model: str) -> Path:
    if model == "codebert":
        return ANALYSIS / "attention_final_3000" / language
    return ANALYSIS / "final_3000_multimodel" / language / model / "attention"


def read_overlap(language: str, model: str) -> dict[str, dict[str, list[float]]]:
    result: dict[str, dict[str, list[float]]] = {graph: {metric: [] for metric in ("recall", "precision")} for graph in GRAPHS}
    if language == "python":
        root = REPO / "attention" / "graph_comparision"
        for graph in GRAPHS:
            for layer in range(6 if model == "plbart" else 12):
                path = root / graph / "exp_0" / f"{model}_layer_{layer}.json"
                record = json.loads(path.read_text())
                head = max(record["fscore"], key=lambda h: (float(record["fscore"][h]["0.05"]), -int(h)))
                for metric in ("recall", "precision"):
                    result[graph][metric].append(float(record[metric][head]["0.05"]))
        return result
    path = attention_root(language, model) / "section_3_2_summary_overlap.csv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) in (12, 24), (path, len(rows))
    for graph in GRAPHS:
        selected = sorted((row for row in rows if row["graph"] == graph), key=lambda row: int(row["layer_index"]))
        assert len(selected) in (6, 12), (path, graph, len(selected))
        for metric in ("recall", "precision"):
            result[graph][metric] = [float(row[metric]) for row in selected]
    return result


def read_ged(language: str, model: str) -> dict[str, list[float]]:
    if language == "python":
        root = REPO / "attention" / "graph_comparision" / "similarity" / "exp_0" / model
    else:
        root = attention_root(language, model) / "similarity_legacy" / model
    result = {graph: [] for graph in GED_GRAPHS}
    for layer in range(6 if model == "plbart" else 12):
        path = root / f"layer_{layer}_threshold_0.05.json"
        record = json.loads(path.read_text())
        for graph in GED_GRAPHS:
            values = [float(value) for value in record[graph].values()]
            assert values and all(np.isfinite(values)), (path, graph)
            result[graph].append(min(values))
    return result


def probe_root(language: str, task: str, model: str, layer: int = 12) -> Path:
    if language == "python":
        return REPO / "DirectProbe" / "results" / task / model / str(layer)
    return REPO / "DirectProbe" / "final_3000" / "results" / language / task / model / str(layer)


def read_probe(language: str, task: str, model: str, layer: int = 12) -> dict:
    root = probe_root(language, task, model, layer)
    clusters = {int(line) for line in (root / "clusters.txt").read_text().splitlines() if line.strip()}
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for line in (root / "prediction.txt").read_text().splitlines():
        fields = line.split("\t")
        assert len(fields) > 1, (root, line[:80])
        truth = re.sub(r"\.0$", "", fields[0])
        # A ranked prediction entry is "clusterID-label,distance"; labels may be negative.
        first = fields[1].split(",", 1)[0]
        predicted = re.sub(r"\.0$", "", first.split("-", 1)[1])
        counts[truth][0] += truth == predicted
        counts[truth][1] += 1
    expected = 5 if task.startswith("distance") else 2 if task.startswith("siblings") else 3
    assert len(counts) == expected, (root, counts)
    expected_test = 1300 if task.startswith("distance") else 600 if task.startswith("siblings") else 900
    assert sum(value[1] for value in counts.values()) == expected_test, root
    return {
        "clusters": len(clusters),
        "accuracy": sum(value[0] for value in counts.values()) / expected_test,
        "per_label": {label: value[0] / value[1] for label, value in counts.items()},
        "test_counts": {label: value[1] for label, value in counts.items()},
    }


OVERLAP = {(lang, model): read_overlap(lang, model) for lang in LANGUAGES for model in MODELS}
GED = {(lang, model): read_ged(lang, model) for lang in LANGUAGES for model in GED_MODELS}
PROBE_LAYERS = {(lang, model, task, layer): read_probe(lang, task, model, layer)
                for lang in LANGUAGES for model in PROBE_MODELS
                for task in ("distance", "distance_id", "siblings", "siblings_id", "dfg")
                for layer in (5, 9, 12)}


def plot_overlap(metric: str, outfile: str) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(13.8, 11.5), sharex=True, sharey=True, constrained_layout=True)
    for row, lang in enumerate(LANGUAGES):
        for col, graph in enumerate(GRAPHS):
            ax = axes[row, col]
            for index, model in enumerate(MODELS):
                values = OVERLAP[(lang, model)][graph][metric]
                ax.plot(range(1, len(values) + 1), values, color=COLORS[index], lw=1.5,
                        marker="o", markersize=2.1, label=MODEL_LABELS[index])
            ax.set_title(f"{LANG_LABELS[row]} · {'AST' if graph == 'ast' else 'DFG'}")
            ax.set_xlim(1, 12)
            ax.set_ylim(0, 0.9 if metric == "recall" else 0.7)
            ax.grid(alpha=0.22)
            if row == 3:
                ax.set_xlabel("Layer (one-based)")
            if col == 0:
                ax.set_ylabel(metric.capitalize())
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    fig.savefig(FIGURES / outfile, bbox_inches="tight")
    plt.close(fig)


def plot_ged() -> None:
    fig, axes = plt.subplots(4, 3, figsize=(13.0, 11.5), sharey=True, constrained_layout=True)
    graph_colors = ("#bd4b30", "#3478aa", "#2b9d70")
    for row, lang in enumerate(LANGUAGES):
        for col, model in enumerate(GED_MODELS):
            ax = axes[row, col]
            for graph, label, color in zip(GED_GRAPHS, GED_LABELS, graph_colors):
                values = GED[(lang, model)][graph]
                ax.plot(range(1, len(values) + 1), values, lw=1.45, marker="o", markersize=2,
                        color=color, label=label)
            ax.set_title(f"{LANG_LABELS[row]} · {dict(zip(MODELS, MODEL_LABELS))[model]}", fontsize=9)
            ax.set_xlim(1, 12)
            ax.set_xticks((1, 6, 12))
            ax.set_ylim(0, 5.0)
            ax.grid(alpha=0.2)
            if row == 3:
                ax.set_xlabel("Layer")
            if col == 0:
                ax.set_ylabel("GED per node")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    fig.savefig(FIGURES / "figure5_ged_four_languages.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_tsne(kind: str, filename: str) -> None:
    if kind == "token_types":
        fig, axes = plt.subplots(2, 2, figsize=(12.7, 10.5), constrained_layout=True)
    else:
        fig, axes = plt.subplots(4, 1, figsize=(12.7, 16.5), constrained_layout=True)
    for ax, lang, label in zip(axes.flat, LANGUAGES, LANG_LABELS):
        if kind == "token_types":
            source = ANALYSIS / lang / "tsne" / "final_3000" / "token_types" / "token_types_layer_5_perplexity_50.png"
        else:
            source = ANALYSIS / lang / "tsne" / "final_3000" / "distances" / "distance_layer_5_perplexity_5.png"
        assert source.is_file(), source
        ax.imshow(plt.imread(source))
        ax.set_title(f"CodeBERT · {label}")
        ax.axis("off")
    fig.savefig(FIGURES / filename, bbox_inches="tight")
    plt.close(fig)


def plot_probe_layers() -> None:
    tasks = ("distance", "distance_id", "siblings", "siblings_id", "dfg")
    titles = ("AST distance\nkeyword--all", "AST distance\nkeyword--identifier",
              "Siblings\nkeyword--all", "Siblings\nkeyword--identifier", "DFG edges")
    fig, axes = plt.subplots(4, 5, figsize=(17.0, 10.5), sharex=True, sharey=True,
                             constrained_layout=True)
    for row, lang in enumerate(LANGUAGES):
        for col, (task, title) in enumerate(zip(tasks, titles)):
            ax = axes[row, col]
            for model, color in zip(PROBE_MODELS, COLORS):
                values = [PROBE_LAYERS[(lang, model, task, layer)]["accuracy"] for layer in (5, 9, 12)]
                ax.plot((5, 9, 12), values, color=color, lw=1.55, marker="o", markersize=3,
                        label=dict(zip(MODELS, MODEL_LABELS))[model])
            ax.set_title(f"{LANG_LABELS[row]} · {title}", fontsize=9)
            ax.set_ylim(0.4, 1.0)
            ax.set_xticks((5, 9, 12))
            ax.grid(alpha=0.2)
            if row == 3:
                ax.set_xlabel("Layer")
            if col == 0:
                ax.set_ylabel("Overall test accuracy")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    fig.savefig(FIGURES / "appendix_probe_layers_four_languages.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    cli = argparse.ArgumentParser()
    cli.add_argument("--skip-tsne", action="store_true")
    args = cli.parse_args()
    plot_overlap("recall", "figure4_recall_four_languages.pdf")
    plot_overlap("precision", "figure7_precision_four_languages.pdf")
    plot_ged()
    if not args.skip_tsne:
        plot_tsne("token_types", "figure10_token_tsne_four_languages.pdf")
        plot_tsne("distances", "figure11_distance_tsne_four_languages.pdf")
    plot_probe_layers()
    print(f"Built {4 if args.skip_tsne else 6} comparative figures in {FIGURES}")


if __name__ == "__main__":
    main()
