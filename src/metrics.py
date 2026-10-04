"""Evaluation metrics for a heavily imbalanced, cost-sensitive problem.

With ~0.07% fraud, accuracy and ROC-AUC look excellent for almost any model,
so the headline numbers are about the top of the ranking: PR-AUC, recall at a
precision floor, recall when only a fixed share of traffic can be reviewed,
and the money lost at a cost-optimal threshold.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def recall_at_precision(y: np.ndarray, score: np.ndarray, target: float) -> dict:
    """Highest recall achievable while keeping precision >= target."""
    precision, recall, thresholds = precision_recall_curve(y, score)
    ok = precision[:-1] >= target  # the last PR point has no threshold
    if not ok.any():
        return {"recall": 0.0, "threshold": None}
    best = np.argmax(np.where(ok, recall[:-1], -1))
    return {"recall": float(recall[best]), "threshold": float(thresholds[best])}


def budget_metrics(y: np.ndarray, score: np.ndarray, fraction: float) -> dict:
    """Recall and precision if analysts can review only the top `fraction` of transactions."""
    k = max(1, int(np.ceil(fraction * len(y))))
    # Break ties at random: a coarse score (e.g. a 0/1 rule) must not get credit
    # for whatever row order the file happens to have.
    tie_breaker = np.random.default_rng(0).random(len(y))
    top = np.lexsort((tie_breaker, -score))[:k]
    caught = int(y[top].sum())
    return {
        "alerts": k,
        "recall": caught / max(1, int(y.sum())),
        "precision": caught / k,
    }


def cost_curve(y: np.ndarray, score: np.ndarray, amount: np.ndarray, review_cost: float):
    """Total cost for every possible number of alerts k (top-k by score).

    cost(k) = review_cost * k  +  fraud amount among transactions not alerted.
    Returns the score thresholds (descending), the alert counts and the costs.
    """
    order = np.argsort(-score, kind="stable")
    fraud_amount = (amount * y)[order]
    caught = np.concatenate([[0.0], np.cumsum(fraud_amount)])
    k = np.arange(len(y) + 1)
    cost = review_cost * k + (fraud_amount.sum() - caught)
    thresholds = np.concatenate([[np.inf], score[order]])
    return thresholds, k, cost


def cost_at_threshold(y, score, amount, threshold: float, review_cost: float) -> dict:
    alert = score >= threshold
    missed = float((amount * y * ~alert).sum())
    reviews = int(alert.sum())
    return {
        "threshold": float(threshold),
        "alerts": reviews,
        "review_cost": review_cost * reviews,
        "missed_fraud_amount": missed,
        "total_cost": review_cost * reviews + missed,
        "recall": float((alert & (y == 1)).sum() / max(1, y.sum())),
        "precision": float((alert & (y == 1)).sum() / max(1, reviews)),
        "fraud_amount_caught_share": 1 - missed / max(1.0, float((amount * y).sum())),
    }


def ranking_metrics(y: np.ndarray, score: np.ndarray, eval_cfg: dict) -> dict:
    out = {
        "pr_auc": float(average_precision_score(y, score)),
        "roc_auc": float(roc_auc_score(y, score)),
        "base_rate": float(y.mean()),
    }
    target = eval_cfg["precision_target"]
    out[f"recall_at_precision_{target}"] = recall_at_precision(y, score, target)["recall"]
    for frac in eval_cfg["alert_budgets"]:
        b = budget_metrics(y, score, frac)
        out[f"recall_at_top_{frac * 100:g}pct"] = b["recall"]
        out[f"precision_at_top_{frac * 100:g}pct"] = b["precision"]
    return out
