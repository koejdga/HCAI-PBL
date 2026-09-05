"""Model training, candidate evaluation, Rashomon set calculation, and odds ratios."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .constants import (
    C_OPTIONS,
    DEFAULT_THETA,
    MAX_LEAF_OPTIONS,
)
from .dataset import (
    build_preprocessor,
    format_feature_name,
    split_penguin_data,
)


def train_decision_tree(penguins, max_leaf_nodes=5, split_data=None):
    # Reusing the same split makes the candidate models directly comparable.
    if split_data is None:
        split_data = split_penguin_data(penguins)

    X_train, X_test, y_train, y_test = split_data

    pipeline = Pipeline(
        [
            ("preprocess", build_preprocessor()),
            (
                "model",
                DecisionTreeClassifier(
                    max_leaf_nodes=max_leaf_nodes,
                    random_state=42,
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    return {
        "pipeline": pipeline,
        "accuracy": accuracy_score(y_test, predictions),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
    }


def train_tree_candidates(penguins, lambda_value, theta_value=DEFAULT_THETA, return_rashomon=False):
    """Select the tree maximizing accuracy minus complexity penalty, and calculate Rashomon set."""
    split_data = split_penguin_data(penguins)
    candidates = []
    maximum_leaves = max(MAX_LEAF_OPTIONS)

    for max_leaf_nodes in MAX_LEAF_OPTIONS:
        result = train_decision_tree(
            penguins,
            max_leaf_nodes=max_leaf_nodes,
            split_data=split_data,
        )

        tree = result["pipeline"].named_steps["model"]
        leaf_count = tree.get_n_leaves()

        normalized_complexity = leaf_count / maximum_leaves
        selection_score = result["accuracy"] - lambda_value * normalized_complexity

        candidates.append(
            {
                **result,
                "max_leaf_nodes": max_leaf_nodes,
                "leaf_count": leaf_count,
                "selection_score": selection_score,
            }
        )

    # Rashomon Set calculation (Lecture 4 Slides 11-21)
    best_accuracy = max(candidate["accuracy"] for candidate in candidates)
    rashomon_count = 0
    simplest_rashomon_model = None

    for candidate in candidates:
        is_in_rashomon = candidate["accuracy"] >= (best_accuracy - theta_value)
        candidate["in_rashomon"] = is_in_rashomon
        if is_in_rashomon:
            rashomon_count += 1
            if simplest_rashomon_model is None or candidate["leaf_count"] < simplest_rashomon_model["leaf_count"]:
                simplest_rashomon_model = candidate

    rashomon_ratio = round((rashomon_count / len(candidates)) * 100, 1)

    selected = max(
        candidates,
        key=lambda candidate: candidate["selection_score"],
    )
    rashomon_info = {
        "best_accuracy": best_accuracy,
        "theta": theta_value,
        "count": rashomon_count,
        "total": len(candidates),
        "ratio_percent": rashomon_ratio,
        "simplest_param": f"Max Leaves = {simplest_rashomon_model['max_leaf_nodes']} ({simplest_rashomon_model['leaf_count']} actual leaves)" if simplest_rashomon_model else "",
    }
    if return_rashomon:
        return selected, candidates, rashomon_info
    return selected, candidates


def train_logistic_regression(split_data, C=1.0):
    X_train, X_test, y_train, y_test = split_data

    pipeline = Pipeline(
        [
            ("preprocess", build_preprocessor()),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=C,
                    l1_ratio=1.0,
                    solver="saga",
                    random_state=42,
                    max_iter=5000,
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    return {
        "pipeline": pipeline,
        "accuracy": accuracy_score(y_test, predictions),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
    }


def compute_odds_ratios(pipeline):
    """Compute odds ratios and plain-language interpretations for logistic regression (Lecture 2 Slides 58-62)."""
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocess"]
    feature_names = [format_feature_name(n) for n in preprocessor.get_feature_names_out()]
    classes = list(model.classes_)
    coefs = model.coef_

    rows = []
    for f_idx, f_name in enumerate(feature_names):
        betas = {classes[c]: round(float(coefs[c, f_idx]), 3) for c in range(len(classes))}
        ors = {classes[c]: round(float(np.exp(coefs[c, f_idx])), 3) for c in range(len(classes))}
        
        active = any(abs(b) > 0.001 for b in betas.values())
        
        max_class = max(betas.keys(), key=lambda k: betas[k])
        if active:
            or_val = ors[max_class]
            if or_val >= 1.05:
                interpretation = f"A unit increase multiplies odds of {max_class} by {or_val:.2f}x"
            elif or_val <= 0.95:
                interpretation = f"A unit increase decreases odds of {max_class} by {or_val:.2f}x"
            else:
                interpretation = "Near neutral effect"
        else:
            interpretation = "Eliminated by L1 regularization (Weight = 0)"

        rows.append({
            "feature": f_name,
            "active": active,
            "betas": betas,
            "odds_ratios": ors,
            "interpretation": interpretation,
        })

    return rows


def train_regression_candidates(penguins, lambda_value, theta_value=DEFAULT_THETA, return_rashomon=False):
    split_data = split_penguin_data(penguins)
    candidates = []

    for C in C_OPTIONS:
        result = train_logistic_regression(split_data, C=C)
        model = result["pipeline"].named_steps["model"]

        non_zero_weights = np.count_nonzero(model.coef_)
        total_weights = model.coef_.size

        normalized_complexity = non_zero_weights / total_weights
        selection_score = result["accuracy"] - lambda_value * normalized_complexity

        candidates.append(
            {
                **result,
                "C": C,
                "non_zero_weights": non_zero_weights,
                "total_weights": total_weights,
                "selection_score": selection_score,
            }
        )

    # Rashomon Set calculation (Lecture 4 Slides 11-21)
    best_accuracy = max(candidate["accuracy"] for candidate in candidates)
    rashomon_count = 0
    simplest_rashomon_model = None

    for candidate in candidates:
        is_in_rashomon = candidate["accuracy"] >= (best_accuracy - theta_value)
        candidate["in_rashomon"] = is_in_rashomon
        if is_in_rashomon:
            rashomon_count += 1
            if simplest_rashomon_model is None or candidate["non_zero_weights"] < simplest_rashomon_model["non_zero_weights"]:
                simplest_rashomon_model = candidate

    rashomon_ratio = round((rashomon_count / len(candidates)) * 100, 1)

    selected = max(
        candidates,
        key=lambda candidate: candidate["selection_score"],
    )
    rashomon_info = {
        "best_accuracy": best_accuracy,
        "theta": theta_value,
        "count": rashomon_count,
        "total": len(candidates),
        "ratio_percent": rashomon_ratio,
        "simplest_param": f"C = {simplest_rashomon_model['C']} ({simplest_rashomon_model['non_zero_weights']} active weights)" if simplest_rashomon_model else "",
    }
    if return_rashomon:
        return selected, candidates, rashomon_info
    return selected, candidates
