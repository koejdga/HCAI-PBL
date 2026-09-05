import os
import uuid
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from django.conf import settings
from .dataset import numeric_values, categorical_values, top_categories

def plot_target_distribution(ax, target, target_column):
    if target_column["type"] == "numeric":
        target_values = numeric_values(target_column["values"])
        target_values = target_values[~np.isnan(target_values)]
        ax.hist(target_values, bins=min(20, max(5, len(target_values))), color="#2563eb", edgecolor="white")
        ax.set_xlabel(target)
        ax.set_ylabel("Count")
    else:
        values = categorical_values(target_column["values"])
        categories, counts = np.unique(values, return_counts=True)
        order = np.argsort(counts)[::-1][:12]
        ax.bar(categories[order], counts[order], color="#2563eb")
        ax.set_xlabel(target)
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=45)
    ax.set_title(f"Distribution of {target}")

def plot_feature_against_target(ax, feature, feature_column, target, target_column):
    feature_type = feature_column["type"]
    target_type = target_column["type"]
    if feature_type == "numeric" and target_type == "numeric":
        x = numeric_values(feature_column["values"])
        y = numeric_values(target_column["values"])
        mask = ~np.isnan(x) & ~np.isnan(y)
        ax.scatter(x[mask], y[mask], color="#16a34a", alpha=0.75)
        ax.set_xlabel(feature)
        ax.set_ylabel(target)
    elif feature_type == "categorical" and target_type == "numeric":
        y = numeric_values(target_column["values"])
        feature_values = np.array(categorical_values(feature_column["values"]))
        categories = top_categories(feature_column["values"])
        groups = [y[(feature_values == cat) & ~np.isnan(y)] for cat in categories]
        ax.boxplot(groups, tick_labels=categories)
        ax.set_xlabel(feature)
        ax.set_ylabel(target)
        ax.tick_params(axis="x", rotation=45)
    elif feature_type == "numeric" and target_type == "categorical":
        x = numeric_values(feature_column["values"])
        target_values = np.array(categorical_values(target_column["values"]))
        categories = top_categories(target_column["values"])
        groups = [x[(target_values == cat) & ~np.isnan(x)] for cat in categories]
        ax.boxplot(groups, tick_labels=categories)
        ax.set_xlabel(target)
        ax.set_ylabel(feature)
        ax.tick_params(axis="x", rotation=45)
    else:
        feature_values = np.array(categorical_values(feature_column["values"]))
        target_values = np.array(categorical_values(target_column["values"]))
        feature_categories = top_categories(feature_column["values"], limit=8)
        target_categories = top_categories(target_column["values"], limit=8)
        counts = np.zeros((len(target_categories), len(feature_categories)))
        for r_idx, target_cat in enumerate(target_categories):
            for c_idx, feat_cat in enumerate(feature_categories):
                counts[r_idx, c_idx] = np.sum((target_values == target_cat) & (feature_values == feat_cat))
        im = ax.imshow(counts, cmap="Blues")
        ax.set_xticks(range(len(feature_categories)), feature_categories, rotation=45, ha="right")
        ax.set_yticks(range(len(target_categories)), target_categories)
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xlabel(feature)
        ax.set_ylabel(target)
    ax.set_title(f"{feature} vs {target}")

def scatter_axis_values(column):
    if column["type"] == "numeric":
        return numeric_values(column["values"]), None
    values = categorical_values(column["values"])
    categories = sorted(set(values))
    encoded_values = np.array([categories.index(v) for v in values], dtype=float)
    return encoded_values, categories

def plot_feature_scatter(ax, dataset, x_feature, y_feature):
    features = dataset["features"]
    columns = dataset["columns"]
    target = dataset["target"]
    target_column = columns[target]
    if x_feature not in features or y_feature not in features:
        raise ValueError("Please select two valid feature columns.")
    
    x, x_categories = scatter_axis_values(columns[x_feature])
    y, y_categories = scatter_axis_values(columns[y_feature])
    mask = ~np.isnan(x) & ~np.isnan(y)
    
    if target_column["type"] == "categorical":
        target_values = np.array(categorical_values(target_column["values"]))
        categories = top_categories(target_column["values"], limit=8)
        for cat in categories:
            cat_mask = mask & (target_values == cat)
            ax.scatter(x[cat_mask], y[cat_mask], alpha=0.75, label=cat)
        ax.legend(title=target, fontsize=8, title_fontsize=9)
    else:
        colors = numeric_values(target_column["values"])
        color_mask = mask & ~np.isnan(colors)
        scatter = ax.scatter(x[color_mask], y[color_mask], c=colors[color_mask], cmap="viridis", alpha=0.75)
        ax.figure.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label=target)
        
    ax.set_xlabel(x_feature)
    ax.set_ylabel(y_feature)
    if x_categories:
        ax.set_xticks(range(len(x_categories)), x_categories, rotation=45, ha="right")
    if y_categories:
        ax.set_yticks(range(len(y_categories)), y_categories)
    ax.set_title(f"{x_feature} vs {y_feature}")

def save_single_plot(output_dir, filename_prefix, title, plot_function):
    filename = f"{filename_prefix}_{uuid.uuid4().hex}.png"
    image_path = os.path.join(output_dir, filename)
    fig, ax = plt.subplots(figsize=(7, 5))
    plot_function(ax)
    fig.tight_layout()
    fig.savefig(image_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return {"title": title, "url": settings.MEDIA_URL + f"project1/{filename}"}

def save_overview_visualizations(dataset):
    output_dir = os.path.join(settings.MEDIA_ROOT, "project1")
    os.makedirs(output_dir, exist_ok=True)
    target = dataset["target"]
    target_column = dataset["columns"][target]
    return [
        save_single_plot(
            output_dir,
            "target_distribution",
            f"Distribution of {target}",
            lambda ax: plot_target_distribution(ax, target, target_column)
        )
    ]

def save_feature_target_visualization(dataset, feature):
    output_dir = os.path.join(settings.MEDIA_ROOT, "project1")
    os.makedirs(output_dir, exist_ok=True)
    target = dataset["target"]
    columns = dataset["columns"]
    plot = save_single_plot(
        output_dir,
        "feature_target",
        f"{feature} vs {target}",
        lambda ax: plot_feature_against_target(ax, feature, columns[feature], target, columns[target])
    )
    plot["key"] = feature_target_key(feature)
    plot["feature"] = feature
    return plot

def save_scatter_visualization(dataset, x_feature, y_feature):
    output_dir = os.path.join(settings.MEDIA_ROOT, "project1")
    os.makedirs(output_dir, exist_ok=True)
    plot = save_single_plot(
        output_dir,
        "feature_scatter",
        f"{x_feature} vs {y_feature}",
        lambda ax: plot_feature_scatter(ax, dataset, x_feature, y_feature)
    )
    plot["key"] = scatter_key(x_feature, y_feature)
    plot["x_feature"] = x_feature
    plot["y_feature"] = y_feature
    return plot

def plot_univariate_distribution(ax, feature_name, column):
    from .dataset import MISSING_MARKERS
    from collections import Counter
    vals = column["values"]
    non_missing = [v for v in vals if v not in MISSING_MARKERS and v != ""]
    if column["type"] == "numeric":
        raw_vals = []
        for v in non_missing:
            try:
                raw_vals.append(float(v))
            except ValueError:
                pass
        if raw_vals:
            ax.hist(raw_vals, bins=15, edgecolor="black", color="#008da2", alpha=0.7)
            ax.set_xlabel(feature_name)
            ax.set_ylabel("Count")
        else:
            ax.text(0.5, 0.5, "No numeric data", ha="center", va="center")
    else:
        counts = Counter(non_missing)
        if counts:
            labels, values = zip(*counts.most_common(10))
            str_labels = [str(lbl) for lbl in labels]
            ax.bar(str_labels, values, color="#0c243c", alpha=0.7, edgecolor="black")
            ax.set_ylabel("Count")
            ax.tick_params(axis="x", rotation=30)
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
    ax.set_title(f"Distribution of {feature_name}")

def save_univariate_visualization(dataset, feature):
    output_dir = os.path.join(settings.MEDIA_ROOT, "project1")
    os.makedirs(output_dir, exist_ok=True)
    columns = dataset["columns"]
    plot = save_single_plot(
        output_dir,
        "univariate",
        f"Distribution of {feature}",
        lambda ax: plot_univariate_distribution(ax, feature, columns[feature])
    )
    plot["key"] = univariate_key(feature)
    plot["feature"] = feature
    return plot

def univariate_key(feature):
    return f"univariate:{feature}"

def feature_target_key(feature):
    return f"feature:{feature}"

def scatter_key(x_feature, y_feature):
    return f"scatter:{x_feature}:{y_feature}"

def build_saved_visualizations(dataset, feature_specs, scatter_specs, univariate_specs=None, highlight_key=None):
    if univariate_specs is None:
        univariate_specs = []
    feature_target_plots = []
    scatter_plots = []
    univariate_plots = []
    
    for feat in feature_specs:
        plot = save_feature_target_visualization(dataset, feat)
        plot["highlight"] = plot["key"] == highlight_key
        feature_target_plots.append(plot)
        
    for spec in scatter_specs:
        plot = save_scatter_visualization(dataset, spec["x"], spec["y"])
        plot["highlight"] = plot["key"] == highlight_key
        scatter_plots.append(plot)
        
    for feat in univariate_specs:
        plot = save_univariate_visualization(dataset, feat)
        plot["highlight"] = plot["key"] == highlight_key
        univariate_plots.append(plot)
        
    return feature_target_plots, scatter_plots, univariate_plots
