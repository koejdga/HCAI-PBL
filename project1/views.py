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
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render

from .forms import CSVUploadForm


def index(request):
    return HttpResponse("Welcome to Project 1!")


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

    raw_rows = []

    for row_number, row in enumerate(rows[1:], start=2):
        if not row or all(not value.strip() for value in row):
            continue

        if len(row) != len(column_names):
            raise ValueError(
                f"Row {row_number} has {len(row)} values, but the header has {len(column_names)} columns."
            )

        raw_rows.append([value.strip() for value in row])

    if not raw_rows:
        raise ValueError("No data rows found in the CSV.")

    columns = {}
    for index, name in enumerate(column_names):
        values = [row[index] for row in raw_rows]
        columns[name] = {
            "values": values,
            "type": get_column_type(values),
        }

    return {
        "column_names": column_names,
        "features": column_names[:-1],
        "target": column_names[-1],
        "columns": columns,
        "row_count": len(raw_rows),
    }


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
    column_names = dataset["column_names"]
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


def train_model(dataset, model_name, test_size_percent):
    if ColumnTransformer is None:
        raise ValueError("scikit-learn is required for model training.")

    task_type = infer_task_type(dataset)
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
        }
        if task_type == "regression":
            row["rmse"] = round(
                float(np.sqrt(mean_squared_error(y_test, predictions))), 4
            )

        rows.append(row)
        if best_row is None or row["score"] > best_row["score"]:
            best_row = row

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
        "best": best_row,
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
        "feature_count": len(dataset["features"]),
        "target": target,
        "target_type": target_column["type"],
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


def upload_csv(request):
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

    if request.method == "POST":
        action = request.POST.get("action", "upload")

        if action == "upload":
            form = CSVUploadForm(request.POST, request.FILES)
            if form.is_valid():
                file = request.FILES["file"]

                try:
                    dataset = parse_csv_dataset(file)
                    request.session["project1_dataset"] = dataset
                    request.session["project1_feature_specs"] = []
                    request.session["project1_scatter_specs"] = []
                    request.session["project1_training_result"] = None

                    result = calculate_average_target(dataset)
                    dataset_summary = get_dataset_summary(dataset)
                    feature_options = dataset["features"]
                    task_type = infer_task_type(dataset)
                    model_options = training_model_options(task_type)
                    overview_visualizations = save_overview_visualizations(dataset)
                except Exception as e:
                    error = f"Error processing file: {str(e)}"
        else:
            form = CSVUploadForm()
            dataset = request.session.get("project1_dataset")
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
                task_type = infer_task_type(dataset)
                model_options = training_model_options(task_type)
                result = calculate_average_target(dataset)
                training_result = request.session.get("project1_training_result")

                if action == "feature_target":
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
        dataset = request.session.get("project1_dataset")
        if dataset:
            dataset_summary = get_dataset_summary(dataset)
            feature_options = dataset["features"]
            task_type = infer_task_type(dataset)
            model_options = training_model_options(task_type)
            result = calculate_average_target(dataset)
            training_result = request.session.get("project1_training_result")
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
        },
    )
