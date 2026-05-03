"""
Extended NAICS evaluation metrics.

compute_naics_metrics(y_true, y_scores, class_labels)
    y_true        : list[str]  — true NAICS codes as strings (e.g. "31", "311", "311812")
    y_scores      : (N, C) ndarray — predicted class probabilities (from predict_proba)
    class_labels  : list[str]  — ordered class names matching columns of y_scores

Returns a dict with:
    acc@1, acc@3, acc@5          — top-k accuracy
    hier@2, hier@3               — top-1 prediction matches true code at 2- / 3-digit prefix
"""

import numpy as np


def _top_k_accuracy(y_true_idx: np.ndarray, y_scores: np.ndarray, k: int) -> float:
    """Fraction of samples where the true class is in the top-k scored classes."""
    # argsort descending: take the last k columns after ascending sort
    top_k = np.argpartition(y_scores, -k, axis=1)[:, -k:]
    hits = np.any(top_k == y_true_idx[:, None], axis=1)
    return float(hits.mean())


def _hierarchical_accuracy(
    y_true: list[str],
    top1_labels: list[str],
    prefix_len: int,
) -> float:
    """Fraction of samples where true and predicted codes share the same prefix."""
    matches = sum(
        t[:prefix_len] == p[:prefix_len]
        for t, p in zip(y_true, top1_labels)
    )
    return matches / len(y_true)


def compute_naics_metrics(
    y_true: list[str],
    y_scores: np.ndarray,
    class_labels: list[str],
) -> dict:
    label_to_idx = {lbl: i for i, lbl in enumerate(class_labels)}
    y_true_idx = np.array([label_to_idx[t] for t in y_true])

    top1_idx = np.argmax(y_scores, axis=1)
    top1_labels = [class_labels[i] for i in top1_idx]

    metrics = {
        "acc@1": _top_k_accuracy(y_true_idx, y_scores, k=1),
        "acc@3": _top_k_accuracy(y_true_idx, y_scores, k=3),
        "acc@5": _top_k_accuracy(y_true_idx, y_scores, k=5),
        "hier@2": _hierarchical_accuracy(y_true, top1_labels, prefix_len=2),
        "hier@3": _hierarchical_accuracy(y_true, top1_labels, prefix_len=3),
    }
    return metrics


def print_metrics(metrics: dict) -> None:
    print("\n" + "=" * 60)
    print("  EXTENDED METRICS")
    print("=" * 60)
    print(f"  Top-1 accuracy : {metrics['acc@1']:.4f}")
    print(f"  Top-3 accuracy : {metrics['acc@3']:.4f}")
    print(f"  Top-5 accuracy : {metrics['acc@5']:.4f}")
    print(f"  Hier acc @2    : {metrics['hier@2']:.4f}  (2-digit prefix match)")
    print(f"  Hier acc @3    : {metrics['hier@3']:.4f}  (3-digit prefix match)")
    print()
