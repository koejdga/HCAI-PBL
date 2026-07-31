import os

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from palmerpenguins import load_penguins
from django.conf import settings
from django.core.paginator import Paginator
from django.shortcuts import render
from django.http import JsonResponse
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from urllib.parse import urlencode

TARGET_COLUMN = "species"
CATEGORICAL_FEATURES = ["island", "sex"]
NUMERIC_FEATURES = [
    "year",
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
]
FEATURE_EFFECT_NUMERIC_FEATURES = [
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES
DISPLAY_COLUMN_LABELS = {
    "island": "Island",
    "sex": "Sex",
    "year": "Year",
    "bill_length_mm": "Bill Length",
    "bill_depth_mm": "Bill Depth",
    "flipper_length_mm": "Flipper Length",
    "body_mass_g": "Body Mass",
    "species": "Species",
}

# Candidate sizes used to compare simple and more detailed trees.
MAX_LEAF_OPTIONS = [2, 3, 4, 5, 6, 8, 10, 12, 15]
C_OPTIONS = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 100.0]

DEFAULT_LAMBDA = 0.20
MIN_LAMBDA = 0.0
MAX_LAMBDA = 1.0
COUNTERFACTUAL_ATTEMPTS = [
    {"N": 2000, "variance_scale": 0.10},
    {"N": 4000, "variance_scale": 0.20},
    {"N": 8000, "variance_scale": 0.35},
]
DATASET_PAGE_SIZE = 12
CLASS_NAMES = ["Adelie", "Chinstrap", "Gentoo"]


def format_feature_name(feature_name):
    """Turn scikit-learn feature names into labels that are easier to read."""
    if feature_name.startswith("categorical__"):
        encoded_name = feature_name.removeprefix("categorical__")
        for column_name in CATEGORICAL_FEATURES:
            prefix = f"{column_name}_"
            if encoded_name.startswith(prefix):
                category = encoded_name.removeprefix(prefix)
                label = column_name.replace("_", " ").title()
                return f"{label} = {category}"

    if feature_name.startswith("numeric__"):
        feature_name = feature_name.removeprefix("numeric__")

    return feature_name.replace("_", " ").title()


def load_clean_penguins():
    """Load the course dataset and remove rows that cannot be trained on."""
    penguins = load_penguins()
    return penguins[FEATURE_COLUMNS + [TARGET_COLUMN]].dropna().copy()


def format_display_value(column, value):
    if isinstance(value, (np.integer, int)):
        formatted = int(value)
    elif isinstance(value, (np.floating, float)):
        formatted = int(value) if float(value).is_integer() else round(float(value), 1)
    else:
        formatted = value

    if column == "year" and isinstance(formatted, (int, float)):
        return int(formatted)
    if column in ["bill_length_mm", "bill_depth_mm", "flipper_length_mm"]:
        return f"{formatted} mm"
    if column == "body_mass_g":
        return f"{formatted} g"
    return formatted


def build_preprocessor():
    # Trees do not need scaled numeric values, so numeric columns pass through.
    return ColumnTransformer(
        [
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            ("numeric", "passthrough", NUMERIC_FEATURES),
        ]
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


def train_tree_candidates(penguins, lambda_value):
    """Select the tree maximizing accuracy minus complexity penalty."""
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

    selected = max(
        candidates,
        key=lambda candidate: candidate["selection_score"],
    )
    return selected, candidates


def parse_lambda(value):
    """Convert the slider value into a safe number between zero and one."""
    try:
        lambda_value = float(value)
    except (TypeError, ValueError):
        return DEFAULT_LAMBDA

    return min(max(lambda_value, MIN_LAMBDA), MAX_LAMBDA)


def parse_feature_effect_feature(value):
    if value in FEATURE_EFFECT_NUMERIC_FEATURES:
        return value
    return FEATURE_EFFECT_NUMERIC_FEATURES[0]


def split_penguin_data(penguins):
    """Create one reproducible split shared by all candidate models."""
    X = penguins[FEATURE_COLUMNS]
    y = penguins[TARGET_COLUMN]

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )


def build_dataset_page(penguins, request):
    selected_row_id = request.GET.get("selected-row")
    page_number = request.GET.get("dataset-page", 1)
    base_query = request.GET.copy()
    base_query.pop("dataset-page", None)
    base_query.pop("format", None)

    table_rows = []
    selected_row = None

    for row_id, row in penguins.iterrows():
        values = {column: row[column] for column in FEATURE_COLUMNS + [TARGET_COLUMN]}
        display_values = {
            column: format_display_value(column, value)
            for column, value in values.items()
        }
        row_data = {
            "id": str(row_id),
            "values": values,
            "display_values": display_values,
            "selected": str(row_id) == selected_row_id,
        }
        table_rows.append(row_data)

        if row_data["selected"]:
            selected_row = row_data

    if selected_row is None and table_rows:
        selected_row = table_rows[0]
        selected_row["selected"] = True
        selected_row_id = selected_row["id"]

    base_query["selected-row"] = selected_row_id
    paginator = Paginator(table_rows, DATASET_PAGE_SIZE)
    page_obj = paginator.get_page(page_number)

    def page_url(number):
        query = base_query.copy()
        query["dataset-page"] = number
        return f"?{urlencode(query, doseq=True)}#counterfactuals"

    page_range = paginator.get_elided_page_range(
        page_obj.number,
        on_each_side=2,
        on_ends=1,
    )

    page_links = [
        {
            "number": number,
            "url": page_url(number) if isinstance(number, int) else None,
            "is_current": number == page_obj.number,
            "is_ellipsis": not isinstance(number, int),
        }
        for number in page_range
    ]

    return {
        "dataset_columns": FEATURE_COLUMNS + [TARGET_COLUMN],
        "dataset_column_labels": [
            DISPLAY_COLUMN_LABELS[column]
            for column in FEATURE_COLUMNS + [TARGET_COLUMN]
        ],
        "dataset_page": page_obj,
        "dataset_page_links": page_links,
        "dataset_previous_url": page_url(page_obj.previous_page_number())
        if page_obj.has_previous()
        else None,
        "dataset_next_url": page_url(page_obj.next_page_number())
        if page_obj.has_next()
        else None,
        "selected_dataset_row": selected_row,
        "selected_row_id": selected_row_id,
        "selected_dataset_row_display_values": [
            {
                "label": DISPLAY_COLUMN_LABELS[column],
                "value": selected_row["display_values"][column],
            }
            for column in FEATURE_COLUMNS + [TARGET_COLUMN]
        ] if selected_row is not None else [],
    }


def build_counterfactual_rows(synthetic_df):
    """Build counterfactual rows in the same dict format as the dataset table."""
    rows = []
    for row_id, row in synthetic_df.iterrows():
        values = {
            **{column: row[column] for column in FEATURE_COLUMNS},
            TARGET_COLUMN: row["predicted_species"],
        }
        display_values = {
            column: format_display_value(column, value)
            for column, value in values.items()
        }
        rows.append(
            {
                "id": str(row_id),
                "values": values,
                "display_values": display_values,
                "distance": round(row["distance"], 2),
            }
        )
    return rows


def sample_local_points(df_clean, original_x, N=1000, variance_scale=0.1):
    """
    Generates N local random variations around a single input point (original_x).
    """
    numeric_features = ["year", "bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
    categorical_features = ["island", "sex"]
    
    # Create N identical copies of our base point x
    sampled_data = pd.DataFrame([original_x] * N).reset_index(drop=True)
    
    # 1. Noise for Numeric Features
    for col in numeric_features:
        # Get standard deviation of the feature from the dataset to scale our noise
        col_std = df_clean[col].std()
        
        # Generate Gaussian noise scaled by variance parameter
        noise = np.random.normal(loc=0.0, scale=col_std * variance_scale, size=N)
        
        # Add noise to the copies
        sampled_data[col] = sampled_data[col] + noise
        
    # 2. Noise for Categorical Features
    for col in categorical_features:
        # Find all unique categories available for this feature
        unique_categories = df_clean[col].unique()
        
        for i in range(N):
            # Introduce a 10% chance to flip the category to a different one
            if np.random.rand() < 0.10:
                sampled_data.at[i, col] = np.random.choice(unique_categories)
                
    return sampled_data


def calculate_mad_l1_distance(original_x, synthetic_df, penguins_df):
    """
    Ranks synthetic points by their Mean Absolute Deviation (MAD)-weighted L1 distance to original_x.
    """
    numeric_features = ["year", "bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
    
    # Calculate MAD for each numeric feature across the entire dataset to weight distances safely
    mads = {}
    for col in numeric_features:
        median = penguins_df[col].median()
        mad = np.median(np.abs(penguins_df[col] - median))
        mads[col] = mad if mad > 0 else 0.001 # Prevent division by zero

    # Calculate weighted L1 distance for each generated row
    distances = np.zeros(len(synthetic_df))
    for col in numeric_features:
        component_dist = np.abs(synthetic_df[col] - original_x[col]) / mads[col]
        distances += component_dist
        
    return distances


def find_counterfactuals(pipeline, penguins, original_x, desired_class):
    """Retry local sampling with wider noise if no counterfactual is found."""
    attempted_rows = 0
    for attempt in COUNTERFACTUAL_ATTEMPTS:
        synthetic_points = sample_local_points(
            penguins,
            original_x,
            N=attempt["N"],
            variance_scale=attempt["variance_scale"],
        )
        attempted_rows += len(synthetic_points)
        synthetic_points["predicted_species"] = pipeline.predict(synthetic_points)

        matching_points = synthetic_points[
            synthetic_points["predicted_species"] == desired_class
        ].copy()
        if matching_points.empty:
            continue

        matching_points["distance"] = calculate_mad_l1_distance(
            original_x,
            matching_points,
            penguins,
        )
        top_counterfactuals = matching_points.sort_values(by="distance").head(5)
        return build_counterfactual_rows(top_counterfactuals), attempted_rows

    return [], attempted_rows


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


def compute_ale(pipeline, penguins, feature_name, bins=10):
    """Compute accumulated local effects for each species without an ALE library."""
    feature_values = penguins[feature_name]
    quantiles = np.linspace(0, 1, bins + 1)
    bin_edges = np.unique(np.quantile(feature_values, quantiles))

    if len(bin_edges) < 2:
        return {
            "feature": feature_name,
            "x_values": [float(feature_values.iloc[0])],
            "curves": {class_name: [0.0] for class_name in pipeline.classes_},
        }

    local_effects = []
    x_values = []

    for lower, upper in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (feature_values >= lower) & (feature_values <= upper)
        bin_rows = penguins.loc[in_bin, FEATURE_COLUMNS].copy()

        if bin_rows.empty:
            local_effects.append(np.zeros(len(pipeline.classes_)))
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
    }


def save_feature_effect_plot(effect_data, plot_kind, feature_name, model_type, lambda_value):
    output_dir = os.path.join(settings.MEDIA_ROOT, "project2")
    os.makedirs(output_dir, exist_ok=True)

    safe_lambda = f"{lambda_value:.2f}".replace(".", "_")
    image_name = f"{plot_kind}_{model_type}_{feature_name}_{safe_lambda}.png"
    image_path = os.path.join(output_dir, image_name)

    figure, axis = plt.subplots(figsize=(9, 5))
    for class_name in CLASS_NAMES:
        if class_name in effect_data["curves"]:
            axis.plot(
                effect_data["x_values"],
                effect_data["curves"][class_name],
                marker="o",
                linewidth=2,
                label=class_name,
            )

    feature_label = DISPLAY_COLUMN_LABELS[feature_name]
    if plot_kind == "pdp":
        axis.set_title(f"PDP for {feature_label}")
        axis.set_ylabel("Average predicted probability")
    else:
        axis.set_title(f"ALE for {feature_label}")
        axis.set_ylabel("Centered probability effect")

    axis.set_xlabel(feature_label)
    axis.grid(True, alpha=0.25)
    axis.legend(title="Species")
    figure.tight_layout()
    figure.savefig(image_path, dpi=150, bbox_inches="tight")
    plt.close(figure)

    return settings.MEDIA_URL + f"project2/{image_name}"


def save_tree_plot(pipeline):
    output_dir = os.path.join(settings.MEDIA_ROOT, "project2")
    os.makedirs(output_dir, exist_ok=True)

    image_path = os.path.join(output_dir, "decision_tree.png")
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

    return settings.MEDIA_URL + "project2/decision_tree.png"


def train_logistic_regression(split_data, C=1.0):
    X_train, X_test, y_train, y_test = split_data

    pipeline = Pipeline(
        [
            ("preprocess", build_preprocessor()),
            ("scaler", StandardScaler()),  # Ensures fast convergence for SAGA solver
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


def train_regression_candidates(penguins, lambda_value):
    split_data = split_penguin_data(penguins)
    candidates = []

    for C in C_OPTIONS:
        result = train_logistic_regression(split_data, C=C)
        model = result["pipeline"].named_steps["model"]

        # Calculate complexity based on active (non-zero) weights
        non_zero_weights = np.count_nonzero(model.coef_)
        total_weights = model.coef_.size  # n_classes * n_features

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

    selected = max(
        candidates,
        key=lambda candidate: candidate["selection_score"],
    )
    return selected, candidates


def index(request):
    original_penguins = load_penguins()
    penguins = load_clean_penguins()

    lambda_value = parse_lambda(request.GET.get("lambda"))
    model_type = request.GET.get("model-type", "decision-tree")
    counterfactuals_desired_class = request.GET.get("desired-class", "adelie")
    feature_effect_feature = parse_feature_effect_feature(
        request.GET.get("feature-effect-feature")
    )

    regression_result, regression_candidates = train_regression_candidates(
        penguins, lambda_value
    )
    tree_result, tree_candidates = train_tree_candidates(penguins, lambda_value)

    active_result = regression_result if model_type == "logistic-regression" else tree_result
    active_pipeline = active_result["pipeline"]
    dataset_context = build_dataset_page(penguins, request)

    if dataset_context["selected_dataset_row"] is not None:
        original_x = {
            column: dataset_context["selected_dataset_row"]["values"][column]
            for column in FEATURE_COLUMNS
        }
    else:
        original_x = penguins.iloc[0][FEATURE_COLUMNS].to_dict()

    target_match_str = counterfactuals_desired_class.capitalize()
    counterfactual_rows, counterfactual_attempted_rows = find_counterfactuals(
        active_pipeline,
        penguins,
        original_x,
        target_match_str,
    )

    pdp_data = compute_pdp(active_pipeline, penguins, feature_effect_feature)
    ale_data = compute_ale(active_pipeline, penguins, feature_effect_feature)
    pdp_image_url = save_feature_effect_plot(
        pdp_data,
        "pdp",
        feature_effect_feature,
        model_type,
        lambda_value,
    )
    ale_image_url = save_feature_effect_plot(
        ale_data,
        "ale",
        feature_effect_feature,
        model_type,
        lambda_value,
    )

    if model_type == "logistic-regression":
        selected_model_description = (
            f"Selected logistic regression model: C = {regression_result['C']}, "
            f"{int(regression_result['non_zero_weights'])} non-zero weights, "
            f"selection score {round(regression_result['selection_score'], 4)}."
        )
        complexity_description = (
            "Number of non-zero coefficients in the trained logistic regression model. "
            "Fewer active weights usually means a simpler model."
        )
        model_data = {
            "train_rows": int(regression_result["train_rows"]),
            "test_rows": int(regression_result["test_rows"]),
            "accuracy_percent": float(round(regression_result["accuracy"] * 100, 2)),
            "complexity_count": int(regression_result["non_zero_weights"]),
            "complexity_label": "Non-Zero Weights",
            "complexity_description": complexity_description,
            "selection_score": float(round(regression_result["selection_score"], 4)),
            "tree_image_url": None,
            "selected_model_description": selected_model_description,
            "candidate_models": [
                {
                    "param_label": f"C = {candidate['C']}",
                    "complexity_count": int(candidate["non_zero_weights"]),
                    "accuracy_percent": float(round(candidate["accuracy"] * 100, 2)),
                    "selection_score": float(round(candidate["selection_score"], 4)),
                    "selected": candidate is regression_result,
                }
                for candidate in regression_candidates
            ],
        }
    else:
        selected_model_description = (
            f"Selected tree: maximum {int(tree_result['max_leaf_nodes'])} leaves, "
            f"{int(tree_result['leaf_count'])} actual leaves, "
            f"selection score {round(tree_result['selection_score'], 4)}."
        )
        complexity_description = (
            "Number of leaf nodes in the trained decision tree. "
            "Fewer leaves typically mean the tree is easier to inspect."
        )
        model_data = {
            "train_rows": int(tree_result["train_rows"]),
            "test_rows": int(tree_result["test_rows"]),
            "accuracy_percent": float(round(tree_result["accuracy"] * 100, 2)),
            "complexity_count": int(tree_result["leaf_count"]),
            "complexity_label": "Leaf Count",
            "complexity_description": complexity_description,
            "selection_score": float(round(tree_result["selection_score"], 4)),
            "tree_image_url": save_tree_plot(tree_result["pipeline"]),
            "selected_model_description": selected_model_description,
            "candidate_models": [
                {
                    "param_label": f"Max Leaves = {int(candidate['max_leaf_nodes'])}",
                    "complexity_count": int(candidate["leaf_count"]),
                    "accuracy_percent": float(round(candidate["accuracy"] * 100, 2)),
                    "selection_score": float(round(candidate["selection_score"], 4)),
                    "selected": candidate is tree_result,
                }
                for candidate in tree_candidates
            ],
        }

    model_data["counterfactual_rows"] = counterfactual_rows
    model_data["counterfactual_attempted_rows"] = counterfactual_attempted_rows
    model_data["dataset_columns"] = dataset_context["dataset_columns"]
    model_data["dataset_column_labels"] = dataset_context["dataset_column_labels"]
    model_data["dataset_page_rows"] = list(dataset_context["dataset_page"].object_list)
    model_data["feature_effect_feature"] = feature_effect_feature
    model_data["feature_effect_feature_label"] = DISPLAY_COLUMN_LABELS[
        feature_effect_feature
    ]
    model_data["pdp_image_url"] = pdp_image_url
    model_data["ale_image_url"] = ale_image_url

    if request.GET.get("format") == "json":
        return JsonResponse(model_data)

    context = {
        "original_row_count": len(original_penguins),
        "row_count": len(penguins),
        "removed_row_count": len(original_penguins) - len(penguins),
        "target_column": TARGET_COLUMN.title(),
        "target_classes": CLASS_NAMES,
        "categorical_features": [
            f.replace("_", " ").title() for f in CATEGORICAL_FEATURES
        ],
        "numeric_features": [f.replace("_", " ").title() for f in NUMERIC_FEATURES],
        "feature_effect_options": [
            {
                "value": feature,
                "label": DISPLAY_COLUMN_LABELS[feature],
                "selected": feature == feature_effect_feature,
            }
            for feature in FEATURE_EFFECT_NUMERIC_FEATURES
        ],
        "feature_effect_feature": feature_effect_feature,
        "feature_effect_feature_label": DISPLAY_COLUMN_LABELS[feature_effect_feature],
        "pdp_image_url": pdp_image_url,
        "ale_image_url": ale_image_url,
        "lambda_value": lambda_value,
        "model_type": model_type,
        "selected_model_description": selected_model_description,
        "complexity_description": complexity_description,
        **model_data,
        "output_classes": [
            {"value": "adelie", "label": "Adelie"},
            {"value": "gentoo", "label": "Gentoo"},
            {"value": "chinstrap", "label": "Chinstrap"},
        ],
        "desired_class": counterfactuals_desired_class,
        **dataset_context,
        "counterfactual_rows": counterfactual_rows,
        "DISPLAY_COLUMN_LABELS": DISPLAY_COLUMN_LABELS,
    }
    return render(request, "project2/index.html", context)
