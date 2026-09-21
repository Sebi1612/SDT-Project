"""Diagnose PLBART paper-reference mismatch using all retained head curves."""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


REPO = Path(__file__).resolve().parents[2]
RECOMPUTED = REPO / "analysis_results" / "final_3000_multimodel" / "python" / "plbart" / "attention"
REFERENCE = REPO / "attention" / "graph_comparision"
OUTPUT = Path(__file__).with_name("plbart_reference_audit.json")
METRICS = ("fscore", "recall", "precision")
THRESHOLDS = ("0", "0.05", "0.1", "0.15", "0.2", "0.25", "0.3", "0.35", "0.4")


def head_vectors(data):
    heads = sorted(data["fscore"], key=int)
    return np.asarray([
        [float(data[metric][head][threshold]) for metric in METRICS for threshold in THRESHOLDS]
        for head in heads
    ])


def best_head(data):
    return int(max(data["fscore"], key=lambda head: data["fscore"][head]["0.05"]))


def run():
    results = []
    for graph in ("ast", "dfg"):
        for layer in range(6):
            candidate_path = RECOMPUTED / graph / f"plbart_layer_{layer}.json"
            reference_path = REFERENCE / graph / "exp_0" / f"plbart_layer_{layer}.json"
            candidate_data = json.loads(candidate_path.read_text())
            reference_data = json.loads(reference_path.read_text())
            candidate = head_vectors(candidate_data)
            reference = head_vectors(reference_data)
            if candidate.shape != reference.shape:
                raise ValueError(f"Head-curve shape mismatch: {graph} L{layer}")
            cost = np.mean(np.abs(candidate[:, None, :] - reference[None, :, :]), axis=2)
            assigned_candidate, assigned_reference = linear_sum_assignment(cost)
            direct = np.diag(cost)
            assigned = cost[assigned_candidate, assigned_reference]
            mapping = {int(i): int(j) for i, j in zip(assigned_candidate, assigned_reference)}
            candidate_best = best_head(candidate_data)
            reference_best = best_head(reference_data)
            result = {
                "graph": graph,
                "layer_index": layer,
                "candidate_num_evaluated": candidate_data.get("num_evaluated", candidate_data.get("num_aligned")),
                "reference_num_evaluated": reference_data.get("num_evaluated"),
                "candidate_best_head": candidate_best,
                "reference_best_head": reference_best,
                "candidate_best_maps_to_reference_head": mapping[candidate_best],
                "direct_head_curve_mean_absolute_difference": float(np.mean(direct)),
                "permuted_head_curve_mean_absolute_difference": float(np.mean(assigned)),
                "direct_head_curve_median_absolute_difference": float(np.median(direct)),
                "direct_head_curve_max_absolute_difference": float(np.max(direct)),
                "reference_head_mapping": mapping,
                "head_curve_mean_absolute_differences": [float(v) for v in direct],
                "permuted_head_curve_mean_absolute_differences": [float(v) for v in assigned],
            }
            print(f"{graph} L{layer+1}: direct={np.mean(direct):.4f}, "
                  f"permuted={np.mean(assigned):.4f}, best={candidate_best}->{reference_best}, "
                  f"candidate={result['candidate_num_evaluated']}")
            results.append(result)
    OUTPUT.write_text(json.dumps({
        "method": "Headwise F-score/recall/precision across nine thresholds; Hungarian minimum-MAD head matching",
        "reference_caveat": "Legacy result JSONs do not record evaluated program counts or checkpoint revisions.",
        "results": results,
    }, indent=2) + "\n")


if __name__ == "__main__":
    run()
