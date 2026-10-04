"""Stage 8: final evaluation on the held-out test window.

Everything that needs a decision (the alert threshold) is decided on the
validation window and only then applied to test, exactly as it would be in
production. Outputs: results/metrics.json, figures in results/figures/.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import precision_recall_curve

from .common import (FEATURE_ROLES, FIGURES_DIR, LABEL, RESULTS_DIR, base_parser, feature_list, get_paths,
                     load_config, load_split, role_feature_set, setup_mlflow, stage_timer, write_json)
from .metrics import cost_at_threshold, cost_curve, ranking_metrics

# Models compared on the test window: label shown in reports -> saved score name.
# "primary" = authorisation-time features, "strict" = no balance columns at all.
MODELS = {
    "XGBoost tuned, primary": "xgb_tuned_primary",
    "XGBoost tuned, strict": "xgb_tuned_strict",
    "Logistic regression, primary": "logreg_primary",
    "Logistic regression, strict": "logreg_strict",
    "XGBoost incl. post-txn balances (leaky)": "xgb_all_incl_post_txn_balance",
    "Rule: isFlaggedFraud": "rule_isFlaggedFraud",
}
PRIMARY = "XGBoost tuned, primary"
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#eda100", "#e87ba4"]
INK, INK_2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
        "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK_2, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 2, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "legend.frameon": False,
    })


def load_scores(paths, name: str, split: str, txn_ids: pd.Series) -> np.ndarray:
    scores = pd.read_parquet(paths.scores / f"{name}__{split}.parquet").set_index("txn_id")["score"]
    return scores.loc[txn_ids].to_numpy()


def choose_threshold(y, score, amount, review_cost: float) -> float:
    """Threshold that minimises review cost + missed fraud amount."""
    thresholds, _, cost = cost_curve(y, score, amount, review_cost)
    return float(thresholds[int(np.argmin(cost))])


def plot_pr_curves(y, scores: dict, path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 6.0))
    for (label, score), color in zip(scores.items(), COLORS):
        if label.startswith("Rule"):
            p, r = (y[score == 1].mean() if (score == 1).any() else 0), y[score == 1].sum() / y.sum()
            ax.plot([r], [p], "o", ms=8, color=color, label=f"{label} (single point)")
            continue
        precision, recall, _ = precision_recall_curve(y, score)
        style = "--" if "leaky" in label else "-"
        ax.plot(recall, precision, style, color=color, label=label)
    ax.set_xlabel("Recall (share of fraud caught)")
    ax.set_ylabel("Precision (share of alerts that are fraud)")
    ax.set_title("Precision-recall on the test window (days 14-16)")
    ax.set_xlim(0, 1.01)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_cost_curve(y, amount, review_cost, curves: dict, path) -> None:
    """curves: label -> (test scores, threshold chosen on validation)."""
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    # Label offsets per curve, chosen so the labels sit in empty parts of the plot.
    label_offsets = [((10, -4), "left"), ((-6, -26), "right")]
    for (label, (score, threshold)), color, (offset, align) in zip(curves.items(), COLORS, label_offsets):
        _, k, cost = cost_curve(y, score, amount, review_cost)
        chosen = cost_at_threshold(y, score, amount, threshold, review_cost)
        x, c = chosen["alerts"] / len(y) * 100, chosen["total_cost"] / 1e6
        ax.plot(k[1:] / len(y) * 100, cost[1:] / 1e6, color=color, label=label)
        ax.plot([x], [c], "o", ms=8, color=color, mec="#fcfcfb", mew=2)
        ax.annotate(f"{chosen['alerts']:,} alerts, {chosen['recall']:.0%} recall,\ncost {c:,.1f}M", (x, c),
                    textcoords="offset points", xytext=offset, ha=align, fontsize=8, color=INK_2)
    ax.axhline(cost[0] / 1e6, color=MUTED, lw=1, ls=":", label="No model (all fraud missed)")
    ax.set_xscale("log")
    ax.set_yscale("log")  # optimal costs differ by orders of magnitude between models
    ax.set_xlabel("Share of test transactions sent for review (%, log scale)")
    ax.set_ylabel("Total cost (million units, log scale)")
    ax.set_title(f"Missed fraud + review cost ({review_cost:g} per alert), test window\n"
                 "dots: threshold chosen on the validation window")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def shap_analysis(booster, test: pd.DataFrame, features, score, threshold, n_sample, seed) -> dict:
    """Global SHAP summary on a fraud-enriched sample plus three local explanations."""
    rng = np.random.default_rng(seed)
    fraud_idx = np.flatnonzero(test[LABEL].to_numpy() == 1)
    legit_idx = rng.choice(np.flatnonzero(test[LABEL].to_numpy() == 0), n_sample - len(fraud_idx), replace=False)
    idx = np.sort(np.concatenate([fraud_idx, legit_idx]))
    X = test.iloc[idx][features]
    # Exact TreeSHAP computed by XGBoost itself (on the GPU); last column is the bias term.
    contribs = booster.predict(xgb.DMatrix(X), pred_contribs=True,
                               iteration_range=(0, booster.best_iteration + 1))
    values, base = contribs[:, :-1], contribs[:, -1]

    plt.figure()
    shap.summary_plot(values, X, max_display=15, show=False, plot_size=(7.5, 5.5))
    plt.title(f"SHAP values, test window ({len(fraud_idx)} frauds + {len(legit_idx):,} random legitimate)",
              fontsize=10)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_summary.png", dpi=150)
    plt.close()

    importance = pd.Series(np.abs(values).mean(axis=0), index=features).sort_values(ascending=False)
    importance.round(5).rename("mean_abs_shap").to_csv(RESULTS_DIR / "shap_importance.csv", index_label="feature")

    # Local explanations: the most confident catch, the fraud the model found
    # hardest (lowest score), and the most confident false alarm.
    y, s = test[LABEL].to_numpy()[idx], score[idx]
    cases = {
        "caught_fraud": int(np.argmax(np.where(y == 1, s, -np.inf))),
        "lowest_scored_fraud": int(np.argmin(np.where(y == 1, s, np.inf))),
        "false_alarm": int(np.argmax(np.where(y == 0, s, -np.inf))),
    }
    local = {}
    for name, i in cases.items():
        expl = shap.Explanation(values=values[i], base_values=base[i], data=X.iloc[i].to_numpy(),
                                feature_names=features)
        plt.figure()
        shap.plots.waterfall(expl, max_display=10, show=False)
        plt.title(f"{name.replace('_', ' ')}: score {s[i]:.3f}, threshold {threshold:.3f}", fontsize=10)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"shap_local_{name}.png", dpi=150)
        plt.close()
        top = np.argsort(-np.abs(values[i]))[:5]
        row = test.iloc[idx[i]]
        local[name] = {
            "txn_id": int(row["txn_id"]), "type": str(row["type"]), "amount": float(row["amount"]),
            "score": float(s[i]), "is_fraud": int(y[i]),
            "top_contributions_log_odds": {features[j]: round(float(values[i, j]), 3) for j in top},
        }
    return {"sample_size": len(idx), "frauds_in_sample": len(fraud_idx),
            "top_features": importance.head(10).round(4).to_dict(), "local_explanations": local}


def main() -> None:
    args = base_parser(__doc__).parse_args()
    cfg = load_config(args.config)
    paths = get_paths(cfg)
    ecfg = cfg["evaluation"]
    review_cost = ecfg["review_cost"]
    apply_style()

    with stage_timer("evaluate"):
        features = feature_list(cfg, role_feature_set(cfg, "primary"))  # superset of strict
        cols = ["txn_id", "type", "amount", LABEL, *features]
        val, test, late = (load_split(cfg, s, cols) for s in ("val", "test", "late_period"))
        y_val, y_test = val[LABEL].to_numpy(), test[LABEL].to_numpy()
        amt_val, amt_test = val["amount"].to_numpy(), test["amount"].to_numpy()

        report = {"feature_sets": {r: role_feature_set(cfg, r) for r in FEATURE_ROLES},
                  "review_cost_per_alert": review_cost,
                  "test_window": {"rows": len(test), "fraud": int(y_test.sum()),
                                  "fraud_amount": float((amt_test * y_test).sum())},
                  "models": {}}
        test_scores = {}
        for label, name in MODELS.items():
            s_val = load_scores(paths, name, "val", val["txn_id"])
            s_test = load_scores(paths, name, "test", test["txn_id"])
            test_scores[label] = s_test
            # The rule is binary: its only sensible threshold is "flagged".
            thr = 1.0 if name.startswith("rule") else choose_threshold(y_val, s_val, amt_val, review_cost)
            report["models"][label] = {
                "score_name": name,
                **ranking_metrics(y_test, s_test, ecfg),
                "cost_at_validation_threshold": cost_at_threshold(y_test, s_test, amt_test, thr, review_cost),
            }
        report["no_model_cost"] = report["test_window"]["fraud_amount"]

        # Sensitivity of the operating point to the assumed review cost.
        tuned_val = load_scores(paths, "xgb_tuned_primary", "val", val["txn_id"])
        tuned_test = test_scores[PRIMARY]
        report["review_cost_sensitivity"] = {
            str(c): cost_at_threshold(y_test, tuned_test, amt_test,
                                      choose_threshold(y_val, tuned_val, amt_val, c), c)
            for c in (10, 100, 1_000, 10_000)
        }

        # Late period: legitimate traffic collapses in the simulator, fraud does not.
        report["late_period_stress_test"] = {"rows": len(late), "fraud": int(late[LABEL].sum())}
        boosters = {}
        for role in FEATURE_ROLES:
            booster = boosters[role] = xgb.Booster()
            booster.load_model(paths.models / f"xgb_tuned_{role}.json")
            role_features = feature_list(cfg, role_feature_set(cfg, role))
            late_score = booster.predict(xgb.DMatrix(late[role_features]),
                                         iteration_range=(0, booster.best_iteration + 1))
            report["late_period_stress_test"][f"xgb_tuned_{role}"] = ranking_metrics(
                late[LABEL].to_numpy(), late_score, ecfg)

        booster = boosters["primary"]
        threshold = report["models"][PRIMARY]["cost_at_validation_threshold"]["threshold"]
        plot_pr_curves(y_test, test_scores, FIGURES_DIR / "pr_curves.png")
        strict = "XGBoost tuned, strict"
        plot_cost_curve(y_test, amt_test, review_cost, {
            PRIMARY: (tuned_test, threshold),
            strict: (test_scores[strict],
                     report["models"][strict]["cost_at_validation_threshold"]["threshold"]),
        }, FIGURES_DIR / "threshold_cost.png")
        report["shap"] = shap_analysis(booster, test, features, tuned_test, threshold, ecfg["shap_sample"], cfg["seed"])

    write_json(RESULTS_DIR / "metrics.json", report)

    mlflow = setup_mlflow(cfg)
    with mlflow.start_run(run_name="evaluate_test_window"):
        mlflow.set_tags({"stage": "evaluation", "model": "xgboost", "feature_set": role_feature_set(cfg, "primary")})
        tuned = report["models"][PRIMARY]
        mlflow.log_metrics({f"test_{k}": v for k, v in tuned.items() if isinstance(v, float)})
        mlflow.log_metrics({f"test_cost_{k}": v for k, v in tuned["cost_at_validation_threshold"].items()})
        for png in FIGURES_DIR.glob("*.png"):
            mlflow.log_artifact(str(png), artifact_path="figures")
        mlflow.log_artifact(str(RESULTS_DIR / "metrics.json"))

    summary = {label: {k: round(m[k], 4) for k in ("pr_auc", "roc_auc", "recall_at_precision_0.9",
                                                   "recall_at_top_0.1pct", "recall_at_top_0.5pct")}
               for label, m in report["models"].items()}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
