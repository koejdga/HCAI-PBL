"""Feature effect calculation (PDP, M-Plot, and ALE)."""

import numpy as np
from sklearn.linear_model import LogisticRegression

from .constants import FEATURE_COLUMNS


def compute_pdp(pipeline, penguins, feature_name, grid_size=30):
    """Compute partial dependence values for each species without a PDP library."""
    feature_values = penguins[feature_name]
    grid = np.linspace(feature_values.min(), feature_values.max(), grid_size)
    curves = {class_name: [] for class_name in pipeline.classes_}

    for value in grid:
        modified = penguins[FEATURE_COLUMNS].copy()
        modified[feature_name] = value
        probabilities = pipeline.predict_proba(modified)
        average_probabilities = probabilities.mean(axis=0)

        for class_index, class_name in enumerate(pipeline.classes_):
            curves[class_name].append(float(average_probabilities[class_index]))

    return {
        "feature": feature_name,
        "x_values": [float(value) for value in grid],
        "curves": curves,
    }


def compute_m_plot(pipeline, penguins, feature_name, bins=10):
    """Compute M-Plot values (conditional expectation over observed neighbors) (Lecture 3 Slide 65)."""
    feature_values = penguins[feature_name]
    quantiles = np.linspace(0, 1, bins + 1)
    bin_edges = np.unique(np.quantile(feature_values, quantiles))

    x_values = []
    curves = {class_name: [] for class_name in pipeline.classes_}

    for lower, upper in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (feature_values >= lower) & (feature_values <= upper)
        bin_rows = penguins.loc[in_bin, FEATURE_COLUMNS]
        if bin_rows.empty:
            continue
        mid_x = float((lower + upper) / 2)
        x_values.append(mid_x)
        probs = pipeline.predict_proba(bin_rows).mean(axis=0)
        for idx, class_name in enumerate(pipeline.classes_):
            curves[class_name].append(float(probs[idx]))

    return {
        "feature": feature_name,
        "x_values": x_values,
        "curves": curves,
    }


def compute_ale(pipeline, penguins, feature_name, bins=10):
    """Compute accumulated local effects for each species without an ALE library (Lecture 3 Slide 66)."""
    feature_values = penguins[feature_name]
    quantiles = np.linspace(0, 1, bins + 1)
    bin_edges = np.unique(np.quantile(feature_values, quantiles))

    if len(bin_edges) < 2:
        return {
            "feature": feature_name,
            "x_values": [float(feature_values.iloc[0])],
            "curves": {class_name: [0.0] for class_name in pipeline.classes_},
        }

    model = pipeline.named_steps["model"]
    is_logistic = isinstance(model, LogisticRegression)

    local_effects = []
    x_values = []

    for lower, upper in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (feature_values >= lower) & (feature_values <= upper)
        bin_rows = penguins.loc[in_bin, FEATURE_COLUMNS].copy()

        if bin_rows.empty:
            local_effects.append(np.zeros(len(pipeline.classes_)))
        else:
            if is_logistic:
                # Exact analytical partial derivative for softmax
                preprocessor = pipeline.named_steps["preprocess"]
                scaler = pipeline.named_steps["scaler"]
                feat_names = list(preprocessor.get_feature_names_out())
                feat_idx = feat_names.index(f"numeric__{feature_name}")
                
                scale = scaler.scale_[feat_idx]
                weights = model.coef_[:, feat_idx] / scale

                probs = pipeline.predict_proba(bin_rows)
                weighted_sum = (probs * weights).sum(axis=1, keepdims=True)
                derivatives = probs * (weights - weighted_sum)
                mean_derivative = derivatives.mean(axis=0)
                delta_x = upper - lower
                local_effects.append(mean_derivative * delta_x)
            else:
                lower_rows = bin_rows.copy()
                upper_rows = bin_rows.copy()
                lower_rows[feature_name] = lower
                upper_rows[feature_name] = upper
                probability_delta = (
                    pipeline.predict_proba(upper_rows) - pipeline.predict_proba(lower_rows)
                )
                local_effects.append(probability_delta.mean(axis=0))

        x_values.append(float((lower + upper) / 2))

    accumulated = np.cumsum(np.vstack(local_effects), axis=0)
    centered = accumulated - accumulated.mean(axis=0)

    curves = {
        class_name: [float(value) for value in centered[:, class_index]]
        for class_index, class_name in enumerate(pipeline.classes_)
    }
    return {
        "feature": feature_name,
        "x_values": x_values,
        "curves": curves,
        "is_exact_analytical": is_logistic,
    }
