"""Matplotlib visualizations for feature effects, decision trees, and regression weights."""

import os
from django.conf import settings

mpl_cache = os.path.join(settings.BASE_DIR, ".matplotlib_cache")
os.makedirs(mpl_cache, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", mpl_cache)

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
from sklearn.tree import plot_tree

from .constants import CLASS_NAMES, DISPLAY_COLUMN_LABELS
from .dataset import format_feature_name


def save_feature_effect_plot(effect_data, plot_kind, feature_name, model_type, lambda_value):
    output_dir = os.path.join(settings.MEDIA_ROOT, "project2")
    os.makedirs(output_dir, exist_ok=True)

    safe_lambda = f"{lambda_value:.2f}".replace(".", "_")
    image_name = f"{plot_kind}_{model_type}_{feature_name}_{safe_lambda}.png"
    image_path = os.path.join(output_dir, image_name)
    if os.path.exists(image_path):
        return settings.MEDIA_URL + f"project2/{image_name}"

    figure, axis = plt.subplots(figsize=(9, 5))
    class_colors = {"Adelie": "#ea580c", "Chinstrap": "#16a34a", "Gentoo": "#2563eb"}
    for class_name in CLASS_NAMES:
        if class_name in effect_data["curves"]:
            axis.plot(
                effect_data["x_values"],
                effect_data["curves"][class_name],
                marker="o",
                linewidth=2.2,
                color=class_colors.get(class_name, None),
                label=class_name,
            )

    feature_label = DISPLAY_COLUMN_LABELS[feature_name]
    if plot_kind == "pdp":
        axis.set_title(f"Partial Dependence Plot (PDP) for {feature_label}")
        axis.set_ylabel("Marginal Predicted Probability")
    elif plot_kind == "mplot":
        axis.set_title(f"M-Plot (Conditional Distribution) for {feature_label}")
        axis.set_ylabel("Conditional Predicted Probability")
    else:
        axis.set_title(f"Accumulated Local Effects (ALE) for {feature_label}")
        axis.set_ylabel("Centered Probability Effect")

    axis.set_xlabel(feature_label)
    axis.grid(True, alpha=0.25)
    axis.legend(title="Species")
    figure.tight_layout()
    figure.savefig(image_path, dpi=150, bbox_inches="tight")
    plt.close(figure)

    return settings.MEDIA_URL + f"project2/{image_name}"


def save_tree_plot(pipeline, leaf_count, create_if_missing=True):
    """Save fallback/zoomable matplotlib decision tree image."""
    output_dir = os.path.join(settings.MEDIA_ROOT, "project2")
    os.makedirs(output_dir, exist_ok=True)

    image_name = "decision_tree.png"
    image_path = os.path.join(output_dir, image_name)

    if os.path.exists(image_path):
        return settings.MEDIA_URL + f"project2/{image_name}"
    if not create_if_missing:
        return None

    preprocessor = pipeline.named_steps["preprocess"]
    trained_tree = pipeline.named_steps["model"]
    feature_names = [
        format_feature_name(name) for name in preprocessor.get_feature_names_out()
    ]

    figure, axis = plt.subplots(figsize=(22, 12))
    plot_tree(
        trained_tree,
        feature_names=feature_names,
        class_names=trained_tree.classes_,
        filled=True,
        rounded=True,
        fontsize=9,
        ax=axis,
    )
    figure.tight_layout()
    figure.savefig(image_path, dpi=150, bbox_inches="tight")
    plt.close(figure)

    return settings.MEDIA_URL + f"project2/{image_name}"


def save_regression_weight_plot(pipeline, C, lambda_value):
    """Generate Weight Plot (Lecture 2 Slide 50) showing active vs zeroed coefficients across classes."""
    output_dir = os.path.join(settings.MEDIA_ROOT, "project2")
    os.makedirs(output_dir, exist_ok=True)

    safe_lambda = f"{lambda_value:.2f}".replace(".", "_")
    image_name = f"weight_plot_C_{C}_{safe_lambda}.png"
    image_path = os.path.join(output_dir, image_name)
    if os.path.exists(image_path):
        return settings.MEDIA_URL + f"project2/{image_name}"

    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocess"]
    feature_names = [format_feature_name(n) for n in preprocessor.get_feature_names_out()]

    classes = model.classes_
    coefs = model.coef_

    n_features = len(feature_names)
    y_pos = np.arange(n_features)
    bar_height = 0.25

    figure, axis = plt.subplots(figsize=(10, 8))
    colors = ["#ea580c", "#16a34a", "#2563eb"]

    for idx, (c_name, color) in enumerate(zip(classes, colors)):
        offset = (idx - 1) * bar_height
        axis.barh(
            y_pos + offset,
            coefs[idx],
            height=bar_height,
            color=color,
            alpha=0.85,
            label=f"{c_name}",
        )

    axis.axvline(0, color="black", linestyle="--", linewidth=1.2, alpha=0.7)
    axis.set_yticks(y_pos)
    axis.set_yticklabels(feature_names, fontsize=10)
    axis.set_xlabel("Learned Coefficient (β)", fontsize=11)
    axis.set_title(f"Logistic Regression Weight Plot (C = {C}, L1 Regularized)", fontsize=13, fontweight="bold")
    axis.grid(True, axis="x", alpha=0.3)
    axis.legend(title="Species Class", loc="lower right")
    axis.invert_yaxis()
    figure.tight_layout()
    figure.savefig(image_path, dpi=150, bbox_inches="tight")
    plt.close(figure)

    return settings.MEDIA_URL + f"project2/{image_name}"
