import csv
import io
import os
import uuid
import numpy as np

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
try:
    from sklearn.compose import ColumnTransformer
    from sklearn.dummy import DummyClassifier, DummyRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
except ImportError:
    ColumnTransformer = None


from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import redirect, render

from .forms import CSVUploadForm


MISSING_MARKERS = {"", "na", "n/a", "nan", "null", "none", "?"}
TASK_OPTIONS = [
    {"value": "classification", "label": "Classification"},
    {"value": "regression", "label": "Regression"},
]


def index(request):
    return upload_csv(request)


def normalize_cell(value):
    value = value.strip()
    if value.lower() in MISSING_MARKERS:
        return ""
    return value


def parse_csv_dataset(uploaded_file):
    decoded_file = uploaded_file.read().decode("utf-8-sig")
    reader = csv.reader(io.StringIO(decoded_file))
    rows = list(reader)

    if len(rows) < 2:
        raise ValueError("The CSV must contain a header row and at least one data row.")

    column_names = [column.strip() for column in rows[0]]
    if len(column_names) < 2:
        raise ValueError(
            "The CSV must contain at least one feature column and one target column."
        )
    if any(name == "" for name in column_names):
        raise ValueError("Every CSV column must have a name.")
    if len(set(column_names)) != len(column_names):
        raise ValueError("CSV column names must be unique.")

    raw_rows = []

    for row_number, row in enumerate(rows[1:], start=2):
        if not row or all(not value.strip() for value in row):
            continue

        if len(row) != len(column_names):
            raise ValueError(
                f"Row {row_number} has {len(row)} values, but the header has {len(column_names)} columns."
            )

        raw_rows.append([normalize_cell(value) for value in row])

    if not raw_rows:
        raise ValueError("No data rows found in the CSV.")

    columns = {}
    for index, name in enumerate(column_names):
        values = [row[index] for row in raw_rows]
        columns[name] = {
            "values": values,
            "type": get_column_type(values),
        }

    dataset = {
        "column_names": column_names,
        "default_target": column_names[-1],
        "columns": copy_columns(columns),
        "source_columns": copy_columns(columns),
        "row_count": len(raw_rows),
        "source_row_count": len(raw_rows),
        "drop_missing_rows": False,
        "dropped_row_count": 0,
    }
    return configure_dataset(dataset)


def get_column_type(values):
    non_empty_values = [value for value in values if value != ""]
    if not non_empty_values:
        return "categorical"

    try:
        for value in non_empty_values:
            float(value)
    except ValueError:
        return "categorical"

    return "numeric"


def numeric_values(values):
    return np.array([float(value) if value != "" else np.nan for value in values])


def categorical_values(values):
    return [value if value != "" else "(missing)" for value in values]


def top_categories(values, limit=12):
    categories, counts = np.unique(categorical_values(values), return_counts=True)
    order = np.argsort(counts)[::-1]
    return categories[order][:limit]


def encoded_column(column):
    if column["type"] == "numeric":
        values = numeric_values(column["values"])
        if np.isnan(values).any():
            fill_value = np.nanmean(values)
            values = np.nan_to_num(values, nan=fill_value)
        return values

    values = categorical_values(column["values"])
    categories = {value: index for index, value in enumerate(sorted(set(values)))}
    return np.array([categories[value] for value in values], dtype=float)


def copy_columns(columns):
    return {
        name: {
            "values": list(column["values"]),
            "type": column["type"],
        }
        for name, column in columns.items()
    }


def plot_target_distribution(ax, target, target_column):
    if target_column["type"] == "numeric":
        target_values = numeric_values(target_column["values"])
        target_values = target_values[~np.isnan(target_values)]
        ax.hist(
            target_values,
            bins=min(20, max(5, len(target_values))),
            color="#2563eb",
            edgecolor="white",
        )
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
        groups = [
            y[(feature_values == category) & ~np.isnan(y)] for category in categories
        ]
        ax.boxplot(groups, tick_labels=categories)
        ax.set_xlabel(feature)
        ax.set_ylabel(target)
        ax.tick_params(axis="x", rotation=45)

    elif feature_type == "numeric" and target_type == "categorical":
        x = numeric_values(feature_column["values"])
        target_values = np.array(categorical_values(target_column["values"]))
        categories = top_categories(target_column["values"])
        groups = [
            x[(target_values == category) & ~np.isnan(x)] for category in categories
        ]
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

        for row_index, target_category in enumerate(target_categories):
            for col_index, feature_category in enumerate(feature_categories):
                counts[row_index, col_index] = np.sum(
                    (target_values == target_category)
                    & (feature_values == feature_category)
                )

        im = ax.imshow(counts, cmap="Blues")
        ax.set_xticks(
            range(len(feature_categories)), feature_categories, rotation=45, ha="right"
        )
        ax.set_yticks(range(len(target_categories)), target_categories)
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xlabel(feature)
        ax.set_ylabel(target)

    ax.set_title(f"{feature} vs {target}")


def plot_correlation(ax, dataset):
    column_names = dataset["features"] + [dataset["target"]]
    columns = dataset["columns"]
    encoded_data = np.column_stack(
        [encoded_column(columns[name]) for name in column_names]
    )
    correlation = np.nan_to_num(np.corrcoef(encoded_data, rowvar=False))
    im = ax.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_title("Column Correlation (categorical values encoded)")
    ax.set_xticks(range(len(column_names)), column_names, rotation=45, ha="right")
    ax.set_yticks(range(len(column_names)), column_names)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def scatter_axis_values(column):
    if column["type"] == "numeric":
        return numeric_values(column["values"]), None

    values = categorical_values(column["values"])
    categories = sorted(set(values))
    encoded_values = np.array([categories.index(value) for value in values], dtype=float)
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

        for category in categories:
            category_mask = mask & (target_values == category)
            ax.scatter(
                x[category_mask],
                y[category_mask],
                alpha=0.75,
                label=category,
            )

        ax.legend(title=target, fontsize=8, title_fontsize=9)
    else:
        colors = numeric_values(target_column["values"])
        color_mask = mask & ~np.isnan(colors)
        scatter = ax.scatter(
            x[color_mask],
            y[color_mask],
            c=colors[color_mask],
            cmap="viridis",
            alpha=0.75,
        )
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

    return {
        "title": title,
        "url": settings.MEDIA_URL + f"project1/{filename}",
    }


def save_overview_visualizations(dataset):
    output_dir = os.path.join(settings.MEDIA_ROOT, "project1")
    os.makedirs(output_dir, exist_ok=True)

    target = dataset["target"]
    columns = dataset["columns"]
    target_column = columns[target]

    return [
        save_single_plot(
            output_dir,
            "target_distribution",
            f"Distribution of {target}",
            lambda ax: plot_target_distribution(ax, target, target_column),
        ),
        save_single_plot(
            output_dir,
            "correlation",
            "Column Correlation",
            lambda ax: plot_correlation(ax, dataset),
        ),
    ]


def save_feature_target_visualization(dataset, feature):
    output_dir = os.path.join(settings.MEDIA_ROOT, "project1")
    os.makedirs(output_dir, exist_ok=True)

    target = dataset["target"]
    columns = dataset["columns"]
    target_column = columns[target]
    plot = save_single_plot(
        output_dir,
        "feature_target",
        f"{feature} vs {target}",
        lambda ax: plot_feature_against_target(
            ax,
            feature,
            columns[feature],
            target,
            target_column,
        ),
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
        lambda ax: plot_feature_scatter(ax, dataset, x_feature, y_feature),
    )
    plot["key"] = scatter_key(x_feature, y_feature)
    plot["x_feature"] = x_feature
    plot["y_feature"] = y_feature
    return plot


def build_saved_visualizations(dataset, feature_specs, scatter_specs, highlight_key=None):
    feature_target_plots = []
    scatter_plots = []

    for feature in feature_specs:
        plot = save_feature_target_visualization(dataset, feature)
        plot["highlight"] = plot["key"] == highlight_key
        feature_target_plots.append(plot)

    for spec in scatter_specs:
        plot = save_scatter_visualization(dataset, spec["x"], spec["y"])
        plot["highlight"] = plot["key"] == highlight_key
        scatter_plots.append(plot)

    return feature_target_plots, scatter_plots


def feature_target_key(feature):
    return f"feature:{feature}"


def scatter_key(x_feature, y_feature):
    return f"scatter:{x_feature}:{y_feature}"


def calculate_average_target(dataset):
    target = dataset["target"]
    target_column = dataset["columns"][target]
    if target_column["type"] == "numeric":
        return float(np.nanmean(numeric_values(target_column["values"])))

    return None


def infer_task_type(dataset):
    target_column = dataset["columns"][dataset["target"]]
    if target_column["type"] == "categorical":
        return "classification"

    values = numeric_values(target_column["values"])
    values = values[~np.isnan(values)]
    unique_values = np.unique(values)
    integer_coded = np.all(np.isclose(unique_values, np.round(unique_values)))

    if (
        integer_coded
        and len(unique_values) <= 10
        and len(unique_values) <= len(values) / 2
    ):
        return "classification"

    return "regression"


def get_effective_task_type(dataset):
    return dataset.get("task_type") or infer_task_type(dataset)


def is_likely_id_column(dataset, column_name):
    lowered_name = column_name.lower().replace("-", "_").replace(" ", "_")
    if lowered_name in {"id", "rowid", "row_id", "index"} or lowered_name.endswith(
        "_id"
    ):
        return True

    values = [value for value in dataset["columns"][column_name]["values"] if value != ""]
    if len(values) != dataset["row_count"]:
        return False

    try:
        numeric_values_for_column = [float(value) for value in values]
    except ValueError:
        return False

    if len(set(numeric_values_for_column)) != dataset["row_count"]:
        return False

    ordered = sorted(numeric_values_for_column)
    one_based = [float(index) for index in range(1, dataset["row_count"] + 1)]
    zero_based = [float(index) for index in range(dataset["row_count"])]
    return ordered == one_based or ordered == zero_based


def suggested_excluded_features(dataset, target):
    return [
        column_name
        for column_name in dataset["column_names"]
        if column_name != target and is_likely_id_column(dataset, column_name)
    ]


def validate_task_type(dataset, task_type):
    valid_task_types = {option["value"] for option in TASK_OPTIONS}
    if task_type not in valid_task_types:
        return infer_task_type(dataset)

    target_column = dataset["columns"][dataset["target"]]
    if task_type == "regression" and target_column["type"] != "numeric":
        raise ValueError("Regression requires a numeric target column.")

    return task_type


def restore_source_rows(dataset):
    if "source_columns" not in dataset:
        dataset["source_columns"] = copy_columns(dataset["columns"])
        dataset["source_row_count"] = dataset["row_count"]

    dataset["columns"] = copy_columns(dataset["source_columns"])
    dataset["row_count"] = dataset["source_row_count"]
    dataset["dropped_row_count"] = 0
    return dataset


def apply_missing_row_drop(dataset):
    kept_indexes = []
    for row_index in range(dataset["source_row_count"]):
        has_missing = any(
            dataset["columns"][column_name]["values"][row_index] == ""
            for column_name in dataset["column_names"]
        )
        if not has_missing:
            kept_indexes.append(row_index)

    if not kept_indexes:
        raise ValueError("Dropping missing rows would remove the whole dataset.")

    for column in dataset["columns"].values():
        column["values"] = [column["values"][index] for index in kept_indexes]
        column["type"] = get_column_type(column["values"])

    dataset["row_count"] = len(kept_indexes)
    dataset["dropped_row_count"] = dataset["source_row_count"] - len(kept_indexes)
    return dataset


def configure_dataset(
    dataset, target=None, features=None, task_type=None, drop_missing_rows=None
):
    dataset = restore_source_rows(dataset)
    column_names = dataset["column_names"]
    target = target or dataset.get("target") or dataset.get("default_target")
    if target not in column_names:
        raise ValueError("Please select a valid target column.")

    available_features = [name for name in column_names if name != target]
    suggested_exclusions = suggested_excluded_features(dataset, target)

    if features is None:
        existing_features = dataset.get("features", [])
        if existing_features:
            selected_features = [
                feature for feature in existing_features if feature in available_features
            ]
        else:
            selected_features = [
                feature
                for feature in available_features
                if feature not in suggested_exclusions
            ]
    else:
        selected_features = [
            feature for feature in features if feature in available_features
        ]

    if not selected_features:
        raise ValueError("Please keep at least one feature column for training.")

    if drop_missing_rows is None:
        drop_missing_rows = dataset.get("drop_missing_rows", False)

    dataset["target"] = target
    dataset["features"] = selected_features
    dataset["excluded_features"] = [
        feature for feature in available_features if feature not in selected_features
    ]
    dataset["suggested_excluded_features"] = suggested_exclusions
    dataset["inferred_task_type"] = infer_task_type(dataset)
    dataset["task_type"] = validate_task_type(dataset, task_type)
    dataset["drop_missing_rows"] = bool(drop_missing_rows)
    if dataset["drop_missing_rows"]:
        dataset = apply_missing_row_drop(dataset)
    return dataset


def count_missing_for_column(dataset, column_name):
    return sum(1 for value in dataset["columns"][column_name]["values"] if value == "")


def unique_non_missing_count(dataset, column_name):
    values = [value for value in dataset["columns"][column_name]["values"] if value != ""]
    return len(set(values))


def build_column_config_options(dataset):
    options = []
    for name in dataset["column_names"]:
        options.append(
            {
                "name": name,
                "type": dataset["columns"][name]["type"],
                "is_target": name == dataset["target"],
                "is_feature": name in dataset["features"],
                "missing_count": count_missing_for_column(dataset, name),
                "unique_count": unique_non_missing_count(dataset, name),
                "likely_id": is_likely_id_column(dataset, name),
            }
        )
    return options


def training_model_options(task_type):
    if task_type == "classification":
        return [
            {
                "value": "logistic_regression",
                "label": "Logistic Regression",
                "hyperparameter": "C",
            },
            {
                "value": "knn_classifier",
                "label": "K-Nearest Neighbors",
                "hyperparameter": "n_neighbors",
            },
        ]

    return [
        {
            "value": "ridge_regression",
            "label": "Ridge Regression",
            "hyperparameter": "alpha",
        },
        {
            "value": "knn_regressor",
            "label": "K-Nearest Neighbors",
            "hyperparameter": "n_neighbors",
        },
    ]


def get_model_choice(model_options, selected_model):
    if selected_model:
        for option in model_options:
            if option["value"] == selected_model:
                return option

    return model_options[0]


def clean_numeric_value(value):
    return np.nan if value == "" else float(value)


def build_training_arrays(dataset):
    features = dataset["features"]
    if not features:
        raise ValueError("Please keep at least one feature column for training.")

    columns = dataset["columns"]
    row_count = dataset["row_count"]
    numeric_feature_indexes = []
    categorical_feature_indexes = []

    rows = []
    for row_index in range(row_count):
        row = []
        for feature_index, feature in enumerate(features):
            column = columns[feature]
            value = column["values"][row_index]
            if column["type"] == "numeric":
                row.append(clean_numeric_value(value))
                if feature_index not in numeric_feature_indexes:
                    numeric_feature_indexes.append(feature_index)
            else:
                row.append(value if value != "" else "(missing)")
                if feature_index not in categorical_feature_indexes:
                    categorical_feature_indexes.append(feature_index)
        rows.append(row)

    return rows, numeric_feature_indexes, categorical_feature_indexes


def build_target_values(dataset, task_type):
    target_column = dataset["columns"][dataset["target"]]
    if task_type == "regression":
        values = numeric_values(target_column["values"])
        if np.isnan(values).any():
            raise ValueError("Regression targets cannot contain missing values.")
        return values

    if count_missing_for_column(dataset, dataset["target"]):
        raise ValueError("Classification targets cannot contain missing values.")

    return np.array(categorical_values(target_column["values"]))


def build_preprocessor(numeric_feature_indexes, categorical_feature_indexes):
    transformers = []
    if numeric_feature_indexes:
        numeric_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric_feature_indexes))

    if categorical_feature_indexes:
        categorical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(
            ("categorical", categorical_pipeline, categorical_feature_indexes)
        )

    return ColumnTransformer(transformers)


def model_candidates(model_name, train_size=None):
    max_neighbors = max(1, train_size or 7)

    if model_name == "logistic_regression":
        return [
            ("C=0.1", LogisticRegression(C=0.1, max_iter=1000)),
            ("C=1", LogisticRegression(C=1, max_iter=1000)),
            ("C=10", LogisticRegression(C=10, max_iter=1000)),
        ]
    if model_name == "knn_classifier":
        neighbor_values = [k for k in (3, 5, 7) if k <= max_neighbors]
        if not neighbor_values:
            neighbor_values = [1]
        return [
            (f"k={k}", KNeighborsClassifier(n_neighbors=k))
            for k in neighbor_values
        ]
    if model_name == "ridge_regression":
        return [
            ("alpha=0.1", Ridge(alpha=0.1)),
            ("alpha=1", Ridge(alpha=1)),
            ("alpha=10", Ridge(alpha=10)),
        ]
    if model_name == "knn_regressor":
        neighbor_values = [k for k in (3, 5, 7) if k <= max_neighbors]
        if not neighbor_values:
            neighbor_values = [1]
        return [
            (f"k={k}", KNeighborsRegressor(n_neighbors=k))
            for k in neighbor_values
        ]

    raise ValueError("Please select a valid model.")


def build_baseline_comparison(task_type, model_score, baseline_score):
    difference = float(model_score) - float(baseline_score)
    tolerance = 0.0001

    # Keep the verdict descriptive so users do not mistake one score for proof
    # that a model is reliable or suitable for a real-world application.
    if difference > tolerance:
        status = "better"
        headline = "Model beats the baseline"
        message = (
            "The trained model found more predictive signal than the naive "
            "reference on this test split."
        )
    elif difference < -tolerance:
        status = "worse"
        headline = "Model is below the baseline"
        message = (
            "The naive reference performed better on this test split. Review "
            "the data, selected features, model, and split before relying on it."
        )
    else:
        status = "similar"
        headline = "Model is similar to the baseline"
        message = (
            "The trained model did not show a meaningful improvement over the "
            "naive reference on this test split."
        )

    if task_type == "classification":
        difference_label = f"{difference * 100:+.2f} percentage points"
    else:
        difference_label = f"{difference:+.4f} R2"

    return {
        "difference": round(difference, 4),
        "difference_label": difference_label,
        "status": status,
        "headline": headline,
        "message": message,
    }


def format_metric_value(task_type, score):
    if task_type == "classification":
        return f"{float(score) * 100:.2f}%"
    return f"{float(score):.4f}"


def train_model(dataset, model_name, test_size_percent):
    if ColumnTransformer is None:
        raise ValueError("scikit-learn is required for model training.")

    task_type = get_effective_task_type(dataset)
    model_options = training_model_options(task_type)
    model_choice = get_model_choice(model_options, model_name)
    test_size = int(test_size_percent) / 100

    if test_size < 0.1 or test_size > 0.5:
        raise ValueError("The test split must be between 10% and 50%.")

    X, numeric_indexes, categorical_indexes = build_training_arrays(dataset)
    y = build_target_values(dataset, task_type)

    stratify = y if task_type == "classification" and len(np.unique(y)) > 1 else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=42,
            stratify=stratify,
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=42,
        )

    # Evaluate a deliberately simple predictor on the same split as the ML models.
    # This gives users a fair reference for deciding whether training adds value.
    if task_type == "classification":
        baseline_name = "Most-frequent class"
        baseline_estimator = DummyClassifier(strategy="most_frequent")
    else:
        baseline_name = "Training-target mean"
        baseline_estimator = DummyRegressor(strategy="mean")

    baseline_pipeline = Pipeline(
        [
            (
                "preprocess",
                build_preprocessor(numeric_indexes, categorical_indexes),
            ),
            ("model", baseline_estimator),
        ]
    )
    baseline_pipeline.fit(X_train, y_train)
    baseline_predictions = baseline_pipeline.predict(X_test)

    # Use the same task-specific metrics as the trained models for comparison.
    if task_type == "classification":
        baseline_score = accuracy_score(y_test, baseline_predictions)
        baseline_rmse = None
    else:
        baseline_score = r2_score(y_test, baseline_predictions)
        baseline_rmse = np.sqrt(
            mean_squared_error(y_test, baseline_predictions)
        )

    rows = []
    best_row = None
    higher_is_better = True
    for parameter_label, estimator in model_candidates(
        model_choice["value"], len(X_train)
    ):
        pipeline = Pipeline(
            [
                (
                    "preprocess",
                    build_preprocessor(numeric_indexes, categorical_indexes),
                ),
                ("model", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        if task_type == "classification":
            score = accuracy_score(y_test, predictions)
            metric = "Accuracy"
        else:
            score = r2_score(y_test, predictions)
            metric = "R2 score"

        row = {
            "parameter": parameter_label,
            "score": round(float(score), 4),
            "display_score": format_metric_value(task_type, score),
        }
        if task_type == "regression":
            row["rmse"] = round(
                float(np.sqrt(mean_squared_error(y_test, predictions))), 4
            )

        rows.append(row)
        if best_row is None or row["score"] > best_row["score"]:
            best_row = row

    comparison = build_baseline_comparison(
        task_type,
        best_row["score"],
        baseline_score,
    )

    return {
        "task_type": task_type,
        "model": model_choice["label"],
        "model_value": model_choice["value"],
        "hyperparameter": model_choice["hyperparameter"],
        "metric": metric,
        "test_size_percent": int(test_size_percent),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "rows": rows,
        "best": {
            **best_row,
            "display_score": format_metric_value(task_type, best_row["score"]),
        },
        "baseline": {
            "name": baseline_name,
            "score": round(float(baseline_score), 4),
            "display_score": format_metric_value(task_type, baseline_score),
            "rmse": (
                round(float(baseline_rmse), 4)
                if baseline_rmse is not None
                else None
            ),
            "description": (
                "Always predicts the most common class from the training data."
                if task_type == "classification"
                else "Always predicts the average target value from the training data."
            ),
        },
        "comparison": comparison,
        "metric_explanation": (
            "Accuracy is the percentage of test rows predicted correctly."
            if task_type == "classification"
            else (
                "R2 measures improvement over predicting the mean. Higher is "
                "better; 1 is perfect, 0 matches the mean baseline, and a "
                "negative value is worse than that baseline."
            )
        ),
        "higher_is_better": higher_is_better,
    }


def selected_features_from_options(feature_options, selected_feature, x_feature, y_feature):
    if feature_options:
        selected_feature = selected_feature or feature_options[0]
        x_feature = x_feature or feature_options[0]
    if len(feature_options) >= 2:
        y_feature = y_feature or feature_options[1]

    return selected_feature, x_feature, y_feature


def get_dataset_summary(dataset):
    target = dataset["target"]
    target_column = dataset["columns"][target]
    return {
        "rows": dataset["row_count"],
        "source_rows": dataset.get("source_row_count", dataset["row_count"]),
        "dropped_rows": dataset.get("dropped_row_count", 0),
        "drop_missing_rows": dataset.get("drop_missing_rows", False),
        "feature_count": len(dataset["features"]),
        "target": target,
        "target_type": target_column["type"],
    }


def count_missing_values(dataset):
    return sum(
        1
        for column in dataset["columns"].values()
        for value in column["values"]
        if value == ""
    )


def get_missing_column_summary(dataset, limit=4):
    missing_columns = []
    for name in dataset["column_names"]:
        missing_count = count_missing_for_column(dataset, name)
        if missing_count:
            missing_columns.append({"name": name, "count": missing_count})

    return missing_columns[:limit]


def get_target_source_note(dataset):
    default_target = dataset.get("default_target")
    if dataset["target"] == default_target:
        return "Defaulted to the last CSV column, as described in the project brief."

    return f"User-selected target. The CSV last column is {default_target}."


def get_task_source_note(dataset, task_type):
    inferred_task_type = dataset.get("inferred_task_type") or infer_task_type(dataset)
    if task_type == inferred_task_type:
        return "Matches the app's automatic type detection."

    return f"User override. The app initially detected {inferred_task_type}."


def get_class_counts(dataset):
    target_values = [
        value for value in categorical_values(dataset["columns"][dataset["target"]]["values"])
        if value != "(missing)"
    ]
    categories, counts = np.unique(target_values, return_counts=True)
    return dict(zip(categories, counts))


def build_quality_warnings(dataset, task_type):
    warnings = []
    target_missing = count_missing_for_column(dataset, dataset["target"])
    if target_missing:
        warnings.append(
            {
                "title": "Target has missing values",
                "message": (
                    f"{target_missing} target rows are missing. Training will ask "
                    "for a target without missing labels."
                ),
            }
        )

    if dataset.get("drop_missing_rows") and dataset.get("dropped_row_count", 0):
        warnings.append(
            {
                "title": "Rows dropped for missing values",
                "message": (
                    f"{dataset['dropped_row_count']} rows were removed before "
                    "visualization and training."
                ),
            }
        )

    missing_columns = get_missing_column_summary(dataset, limit=8)
    if missing_columns:
        columns = ", ".join(
            f"{column['name']} ({column['count']})" for column in missing_columns
        )
        warnings.append(
            {
                "title": "Missing values detected",
                "message": f"Columns with missing values: {columns}.",
            }
        )

    if dataset.get("suggested_excluded_features"):
        warnings.append(
            {
                "title": "Likely ID columns excluded",
                "message": (
                    "Excluded by default: "
                    + ", ".join(dataset["suggested_excluded_features"])
                    + ". Review the feature list if these columns are meaningful."
                ),
            }
        )

    if task_type == "classification":
        class_counts = get_class_counts(dataset)
        if class_counts:
            largest_class = max(class_counts.values())
            total = sum(class_counts.values())
            if total and largest_class / total >= 0.7:
                warnings.append(
                    {
                        "title": "Class imbalance",
                        "message": (
                            "The largest class contains "
                            f"{largest_class} of {total} labeled rows."
                        ),
                    }
                )

    high_cardinality_features = [
        feature
        for feature in dataset["features"]
        if dataset["columns"][feature]["type"] == "categorical"
        and unique_non_missing_count(dataset, feature) > 20
    ]
    if high_cardinality_features:
        warnings.append(
            {
                "title": "High-cardinality categorical features",
                "message": (
                    "These features have many categories: "
                    + ", ".join(high_cardinality_features[:5])
                    + "."
                ),
            }
        )

    if dataset["row_count"] < 30:
        warnings.append(
            {
                "title": "Small dataset",
                "message": "Model scores can be unstable with fewer than 30 rows.",
            }
        )

    return warnings


def build_dataset_assumptions(dataset, task_type, test_size_percent):
    target = dataset["target"]
    target_type = dataset["columns"][target]["type"]
    column_count = len(dataset["column_names"])
    total_cells = dataset["row_count"] * column_count
    missing_count = count_missing_values(dataset)
    missing_columns = get_missing_column_summary(dataset)

    if dataset.get("drop_missing_rows"):
        missing_detail = (
            f"{missing_count} empty cells remain after dropping "
            f"{dataset.get('dropped_row_count', 0)} rows."
        )
        imputation_note = (
            "Rows containing missing values are removed before visualization and training."
        )
    elif missing_count:
        missing_detail = f"{missing_count} of {total_cells} cells are empty."
        imputation_note = (
            "During training, numeric feature gaps use median imputation and "
            "categorical feature gaps use the most frequent value."
        )
    else:
        missing_detail = "No empty cells were detected."
        imputation_note = (
            "If missing feature values appear later, numeric gaps use median "
            "imputation and categorical gaps use the most frequent value."
        )

    if task_type == "classification":
        task_detail = "The target looks categorical or compact integer-coded."
        target_note = "Classification targets are compared by class labels."
    else:
        task_detail = "The target looks continuous numeric."
        target_note = "Regression targets must be numeric and cannot be missing."

    return {
        "cards": [
            {
                "label": "Dataset",
                "value": f"{dataset['row_count']} rows",
                "detail": (
                    f"{len(dataset['features'])} selected features, "
                    f"{column_count} columns total."
                    + (
                        f" Dropped {dataset.get('dropped_row_count', 0)} of "
                        f"{dataset.get('source_row_count', dataset['row_count'])} uploaded rows."
                        if dataset.get("drop_missing_rows")
                        else ""
                    )
                ),
            },
            {
                "label": "Target",
                "value": target,
                "detail": f"{get_target_source_note(dataset)} Detected as {target_type}.",
            },
            {
                "label": "Task",
                "value": task_type.title(),
                "detail": f"{task_detail} {get_task_source_note(dataset, task_type)}",
            },
            {
                "label": "Cleaning",
                "value": (
                    "Drop rows"
                    if dataset.get("drop_missing_rows")
                    else "Keep rows"
                ),
                "detail": missing_detail,
            },
            {
                "label": "Evaluation",
                "value": f"{test_size_percent}% test split",
                "detail": "Models are scored on a held-out test set using random seed 42.",
            },
        ],
        "notes": [
            "The app defaults to the last CSV column from the assignment, but the user can change it.",
            target_note,
            imputation_note,
            "These assumptions are shown so the user can review them before trusting model scores.",
        ],
        "missing_columns": missing_columns,
        "quality_warnings": build_quality_warnings(dataset, task_type),
    }


def is_ajax_request(request):
    return (
        request.headers.get("x-requested-with") == "XMLHttpRequest"
        or request.headers.get("accept") == "application/json"
        or request.POST.get("ajax") == "1"
    )


def plot_payload(plot):
    payload = {
        "key": plot["key"],
        "title": plot["title"],
        "url": plot["url"],
    }
    if "feature" in plot:
        payload["feature"] = plot["feature"]
    if "x_feature" in plot:
        payload["x_feature"] = plot["x_feature"]
    if "y_feature" in plot:
        payload["y_feature"] = plot["y_feature"]

    return payload


def ajax_error(message, status=400):
    return JsonResponse({"status": "error", "message": message}, status=status)


def clear_project1_session(request):
    for key in (
        "project1_dataset",
        "project1_feature_specs",
        "project1_scatter_specs",
        "project1_training_result",
    ):
        request.session.pop(key, None)


def reset_project1_outputs(request):
    request.session["project1_feature_specs"] = []
    request.session["project1_scatter_specs"] = []
    request.session["project1_training_result"] = None


def normalize_dataset_columns(dataset):
    if "source_columns" not in dataset:
        dataset["source_columns"] = copy_columns(dataset["columns"])
        dataset["source_row_count"] = dataset["row_count"]

    for columns in (dataset["source_columns"], dataset["columns"]):
        for column in columns.values():
            values = [normalize_cell(str(value)) for value in column["values"]]
            column["values"] = values
            column["type"] = get_column_type(values)

    return dataset


def ensure_dataset_configuration(dataset):
    if not dataset:
        return None

    dataset = normalize_dataset_columns(dataset)
    dataset.setdefault("default_target", dataset.get("target") or dataset["column_names"][-1])
    return configure_dataset(
        dataset,
        target=dataset.get("target"),
        features=dataset.get("features"),
        task_type=dataset.get("task_type"),
    )


def upload_csv(request):
    dataset = None
    result = None
    error = None
    duplicate_message = None
    overview_visualizations = []
    feature_target_plots = []
    scatter_plots = []
    dataset_summary = None
    feature_options = []
    selected_feature = None
    selected_x_feature = None
    selected_y_feature = None
    task_type = None
    model_options = []
    selected_model = None
    test_size_percent = 20
    training_result = None
    column_config_options = []

    if request.method == "POST":
        action = request.POST.get("action", "upload")

        if action == "clear_dataset":
            clear_project1_session(request)
            return redirect("project1:index")

        if action == "upload":
            form = CSVUploadForm(request.POST, request.FILES)
            if form.is_valid():
                file = request.FILES["file"]

                try:
                    dataset = parse_csv_dataset(file)
                    request.session["project1_dataset"] = dataset
                    reset_project1_outputs(request)

                    result = calculate_average_target(dataset)
                    dataset_summary = get_dataset_summary(dataset)
                    feature_options = dataset["features"]
                    task_type = get_effective_task_type(dataset)
                    model_options = training_model_options(task_type)
                    overview_visualizations = save_overview_visualizations(dataset)
                except Exception as e:
                    error = f"Error processing file: {str(e)}"
        else:
            form = CSVUploadForm()
            dataset = ensure_dataset_configuration(
                request.session.get("project1_dataset")
            )
            feature_specs = request.session.get("project1_feature_specs", [])
            scatter_specs = request.session.get("project1_scatter_specs", [])
            highlight_key = None

            if not dataset:
                error = "Please upload a CSV file before creating a plot."
                if is_ajax_request(request):
                    return ajax_error(error)
            else:
                dataset_summary = get_dataset_summary(dataset)
                feature_options = dataset["features"]
                task_type = get_effective_task_type(dataset)
                model_options = training_model_options(task_type)
                result = calculate_average_target(dataset)
                training_result = request.session.get("project1_training_result")
                if training_result:
                    selected_model = training_result.get("model_value")
                    test_size_percent = training_result.get(
                        "test_size_percent", test_size_percent
                    )

                if action == "configure_dataset":
                    try:
                        dataset = configure_dataset(
                            dataset,
                            target=request.POST.get("target"),
                            features=request.POST.getlist("features"),
                            task_type=request.POST.get("task_type"),
                            drop_missing_rows=(
                                request.POST.get("missing_policy") == "drop"
                            ),
                        )
                        request.session["project1_dataset"] = dataset
                        reset_project1_outputs(request)
                        feature_specs = []
                        scatter_specs = []
                        training_result = None
                        selected_model = None
                        result = calculate_average_target(dataset)
                        dataset_summary = get_dataset_summary(dataset)
                        feature_options = dataset["features"]
                        task_type = get_effective_task_type(dataset)
                        model_options = training_model_options(task_type)
                    except Exception as e:
                        error = f"Error updating dataset setup: {str(e)}"
                elif action == "feature_target":
                    selected_feature = request.POST.get("feature")
                    if selected_feature not in feature_options:
                        error = "Please select a valid feature column."
                    elif selected_feature in feature_specs:
                        highlight_key = feature_target_key(selected_feature)
                        duplicate_message = (
                            f"A plot for {selected_feature} already exists."
                        )
                    else:
                        if is_ajax_request(request):
                            try:
                                plot = save_feature_target_visualization(
                                    dataset, selected_feature
                                )
                                feature_specs = feature_specs + [selected_feature]
                                request.session["project1_feature_specs"] = (
                                    feature_specs
                                )
                                return JsonResponse(
                                    {
                                        "status": "created",
                                        "plot_type": "feature_target",
                                        "plot": plot_payload(plot),
                                    }
                                )
                            except Exception as e:
                                return ajax_error(f"Error creating plot: {str(e)}")
                        feature_specs = feature_specs + [selected_feature]
                        request.session["project1_feature_specs"] = feature_specs
                elif action == "scatter":
                    selected_x_feature = request.POST.get("x_feature")
                    selected_y_feature = request.POST.get("y_feature")

                    if (
                        selected_x_feature not in feature_options
                        or selected_y_feature not in feature_options
                    ):
                        error = "Please select two valid feature columns."
                    elif selected_x_feature == selected_y_feature:
                        error = "Please select two different feature columns."
                    else:
                        new_scatter = {
                            "x": selected_x_feature,
                            "y": selected_y_feature,
                        }
                        if new_scatter in scatter_specs:
                            highlight_key = scatter_key(
                                selected_x_feature, selected_y_feature
                            )
                            duplicate_message = (
                                f"A scatter plot for {selected_x_feature} vs "
                                f"{selected_y_feature} already exists."
                            )
                        else:
                            if is_ajax_request(request):
                                try:
                                    plot = save_scatter_visualization(
                                        dataset,
                                        selected_x_feature,
                                        selected_y_feature,
                                    )
                                    scatter_specs = scatter_specs + [new_scatter]
                                    request.session["project1_scatter_specs"] = (
                                        scatter_specs
                                    )
                                    return JsonResponse(
                                        {
                                            "status": "created",
                                            "plot_type": "scatter",
                                            "plot": plot_payload(plot),
                                        }
                                    )
                                except Exception as e:
                                    return ajax_error(
                                        f"Error creating plot: {str(e)}"
                                    )
                            scatter_specs = scatter_specs + [new_scatter]
                            request.session["project1_scatter_specs"] = scatter_specs
                elif action == "remove_feature_target":
                    selected_feature = request.POST.get("feature")
                    if selected_feature not in feature_specs:
                        error = "That feature plot is not currently shown."
                    else:
                        feature_specs = [
                            feature
                            for feature in feature_specs
                            if feature != selected_feature
                        ]
                        request.session["project1_feature_specs"] = feature_specs
                        if is_ajax_request(request):
                            return JsonResponse(
                                {
                                    "status": "removed",
                                    "plot_type": "feature_target",
                                    "key": feature_target_key(selected_feature),
                                }
                            )
                elif action == "remove_scatter":
                    selected_x_feature = request.POST.get("x_feature")
                    selected_y_feature = request.POST.get("y_feature")
                    selected_scatter = {
                        "x": selected_x_feature,
                        "y": selected_y_feature,
                    }
                    if selected_scatter not in scatter_specs:
                        error = "That scatter plot is not currently shown."
                    else:
                        scatter_specs = [
                            spec for spec in scatter_specs if spec != selected_scatter
                        ]
                        request.session["project1_scatter_specs"] = scatter_specs
                        if is_ajax_request(request):
                            return JsonResponse(
                                {
                                    "status": "removed",
                                    "plot_type": "scatter",
                                    "key": scatter_key(
                                        selected_x_feature, selected_y_feature
                                    ),
                                }
                            )
                elif action == "train_model":
                    selected_model = request.POST.get("model")
                    test_size_percent = request.POST.get("test_size", "20")
                    try:
                        training_result = train_model(
                            dataset,
                            selected_model,
                            test_size_percent,
                        )
                        request.session["project1_training_result"] = training_result
                        selected_model = training_result["model_value"]
                        test_size_percent = training_result["test_size_percent"]
                    except Exception as e:
                        error = f"Error training model: {str(e)}"
                else:
                    error = "Unknown plot action."

                if is_ajax_request(request):
                    if duplicate_message:
                        return JsonResponse(
                            {
                                "status": "duplicate",
                                "message": duplicate_message,
                                "key": highlight_key,
                            }
                        )
                    if error:
                        return ajax_error(error)

                try:
                    overview_visualizations = save_overview_visualizations(dataset)
                    feature_target_plots, scatter_plots = build_saved_visualizations(
                        dataset,
                        feature_specs,
                        scatter_specs,
                    )
                except Exception as e:
                    error = f"Error creating plot: {str(e)}"
    else:
        form = CSVUploadForm()
        dataset = ensure_dataset_configuration(request.session.get("project1_dataset"))
        if dataset:
            request.session["project1_dataset"] = dataset
            dataset_summary = get_dataset_summary(dataset)
            feature_options = dataset["features"]
            task_type = get_effective_task_type(dataset)
            model_options = training_model_options(task_type)
            result = calculate_average_target(dataset)
            training_result = request.session.get("project1_training_result")
            if training_result:
                selected_model = training_result.get("model_value")
                test_size_percent = training_result.get(
                    "test_size_percent", test_size_percent
                )
            feature_specs = request.session.get("project1_feature_specs", [])
            scatter_specs = request.session.get("project1_scatter_specs", [])
            overview_visualizations = save_overview_visualizations(dataset)
            feature_target_plots, scatter_plots = build_saved_visualizations(
                dataset, feature_specs, scatter_specs
            )

    selected_feature, selected_x_feature, selected_y_feature = (
        selected_features_from_options(
            feature_options,
            selected_feature,
            selected_x_feature,
            selected_y_feature,
        )
    )
    assumption_panel = None
    if dataset_summary and dataset:
        assumption_panel = build_dataset_assumptions(
            dataset, task_type, test_size_percent
        )
        column_config_options = build_column_config_options(dataset)

    return render(
        request,
        "project1/upload.html",
        {
            "form": form,
            "result": result,
            "error": error,
            "duplicate_message": duplicate_message,
            "overview_visualizations": overview_visualizations,
            "feature_target_plots": feature_target_plots,
            "scatter_plots": scatter_plots,
            "dataset_summary": dataset_summary,
            "feature_options": feature_options,
            "selected_feature": selected_feature,
            "selected_x_feature": selected_x_feature,
            "selected_y_feature": selected_y_feature,
            "task_type": task_type,
            "model_options": model_options,
            "selected_model": selected_model or (
                model_options[0]["value"] if model_options else None
            ),
            "test_size_percent": test_size_percent,
            "training_result": training_result,
            "assumption_panel": assumption_panel,
            "column_config_options": column_config_options,
            "task_options": TASK_OPTIONS,
        },
    )
