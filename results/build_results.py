"""Build a four-language counterpart to the figures and tables in paper.pdf §4.

Run with the project's attention environment; no model inference is performed here.
"""

from __future__ import annotations

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
TABLES = OUTPUT / "tables"
FIGURES.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)
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
    path = attention_root(language, model) / "section_3_2_summary_overlap.csv"
    result: dict[str, dict[str, list[float]]] = {graph: {metric: [] for metric in ("recall", "precision")} for graph in GRAPHS}
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
        # §4 / Figure 5 reference outputs, not a final-3000 Python GED rerun.
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
PROBES = {(lang, model, task): read_probe(lang, task, model)
          for lang in LANGUAGES for model in PROBE_MODELS
          for task in ("distance", "distance_id", "siblings", "siblings_id", "dfg")}
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
        # A four-row layout makes the two AST/hidden panels for each language
        # legible when the combined figure is placed on an A4 page.
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


def probe_table(tasks: tuple[str, ...], labels: tuple[tuple[str, ...], ...], filename: str,
                caption: str, label: str) -> None:
    lines = ["\\begin{table}[p]", "\\centering", "\\small"]
    lines.append("\\resizebox{\\textwidth}{!}{%")
    columns = 3 + max(len(label_set) for label_set in labels)
    lines.append("\\begin{tabular}{llr" + "r" * (columns - 3) + "}")
    lines.append("\\toprule")
    for task, label_set in zip(tasks, labels):
        task_title = {"distance": "Keyword--all AST distance", "distance_id": "Keyword--identifier AST distance",
                      "siblings": "Keyword--all siblings", "siblings_id": "Keyword--identifier siblings",
                      "dfg": "Identifier--identifier DFG"}[task]
        lines.append("\\multicolumn{" + str(columns) + "}{l}{" + task_title + "} \\\\")
        lines.append("Language & Model & Clusters & " + " & ".join(label_set) + " \\\\")
        lines.append("\\midrule")
        for lang, lang_label in zip(LANGUAGES, LANG_LABELS):
            for model in PROBE_MODELS:
                result = PROBES[(lang, model, task)]
                if task.startswith("distance"):
                    label_keys = ("2", "3", "4", "5", "6")
                elif task.startswith("siblings"):
                    label_keys = ("0", "1") if lang == "python" else ("NotSibling", "Sibling")
                else:
                    label_keys = ("0", "1", "-1") if lang == "python" else ("NoEdge", "ComesFrom", "ComputedFrom")
                values = [f"{result['per_label'][key]:.2f}" for key in label_keys]
                model_label = {"codebert": "CodeBERT", "graphcodebert": "GraphCodeBERT", "codet5": "CodeT5"}[model]
                lines.append(f"{lang_label} & {model_label} & {result['clusters']} & " + " & ".join(values) + " \\\\")
            lines.append("\\addlinespace")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines += ["\\end{tabular}%", "}", f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\end{table}"]
    (TABLES / filename).write_text("\n".join(lines) + "\n")


def summarize() -> dict:
    record = {"attention": {}, "ged": {}, "probing": {}, "probing_layers": {}, "tsne": {}}
    for lang in LANGUAGES:
        record["attention"][lang] = {}
        record["ged"][lang] = {}
        record["probing"][lang] = {}
        record["probing_layers"][lang] = {}
        tsne_manifest = json.loads((ANALYSIS / lang / "tsne" / "final_3000" / "distances"
                                    / "distance_tsne_manifest.json").read_text())
        record["tsne"][lang] = {"selected_program_spearman_ast_vs_hidden_distance":
                                float(tsne_manifest["spearman_ast_vs_hidden_distance"]["5"]["spearman_r"])}
        for graph in GRAPHS:
            values = [max(OVERLAP[(lang, model)][graph]["recall"]) for model in MODELS]
            record["attention"][lang][f"{graph}_peak_recall_median_models"] = float(np.median(values))
        for graph in GED_GRAPHS:
            values = [min(GED[(lang, model)][graph]) for model in GED_MODELS]
            record["ged"][lang][f"{graph}_best_layer_median_models"] = float(np.median(values))
        for task in ("distance", "distance_id", "siblings", "siblings_id", "dfg"):
            values = [PROBES[(lang, model, task)]["accuracy"] for model in PROBE_MODELS]
            record["probing"][lang][f"{task}_median_models_accuracy"] = float(np.median(values))
            record["probing"][lang][f"{task}_median_models_clusters"] = float(np.median(
                [PROBES[(lang, model, task)]["clusters"] for model in PROBE_MODELS]))
            record["probing_layers"][lang][task] = {str(layer): float(np.median(
                [PROBE_LAYERS[(lang, model, task, layer)]["accuracy"] for model in PROBE_MODELS]))
                for layer in (5, 9, 12)}
    return record


def main() -> None:
    plot_overlap("recall", "figure4_recall_four_languages.pdf")
    plot_overlap("precision", "figure7_precision_four_languages.pdf")
    plot_ged()
    plot_tsne("token_types", "figure10_token_tsne_four_languages.pdf")
    plot_tsne("distances", "figure11_distance_tsne_four_languages.pdf")
    plot_probe_layers()
    probe_table(("distance", "distance_id"), (("2", "3", "4", "5", "6"),) * 2,
                "table1_distance.tex", "Four-language DirectProbe AST tree-distance results at layer 12. Values are per-label test accuracy; Python is the paper-repository reference and the other languages are final-3000 runs.", "tab:distance")
    probe_table(("siblings", "siblings_id"), (("Not siblings", "Siblings"),) * 2,
                "table2_siblings.tex", "Four-language DirectProbe AST-sibling results at layer 12. Values are per-label test accuracy.", "tab:siblings")
    probe_table(("dfg",), (("No edge", "Comes from", "Computed from"),),
                "table3_dfg.tex", "Four-language DirectProbe DFG-edge results at layer 12. Values are per-label test accuracy.", "tab:dfg")
    (OUTPUT / "derived_summary.json").write_text(json.dumps(summarize(), indent=2) + "\n")
    print(f"Built six comparative figures, three probing tables, and {OUTPUT / 'derived_summary.json'}")


if __name__ == "__main__":
    main()
