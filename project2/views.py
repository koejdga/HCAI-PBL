import os

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from palmerpenguins import load_penguins
from django.conf import settings
from django.shortcuts import render
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree


TARGET_COLUMN = "species"
CATEGORICAL_FEATURES = ["island", "sex"]
NUMERIC_FEATURES = [
    "year",
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES


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


def train_decision_tree(penguins, max_leaf_nodes=5):
    X = penguins[FEATURE_COLUMNS]
    y = penguins[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

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


def save_tree_plot(pipeline):
    output_dir = os.path.join(settings.MEDIA_ROOT, "project2")
    os.makedirs(output_dir, exist_ok=True)

    image_path = os.path.join(output_dir, "decision_tree.png")
    preprocessor = pipeline.named_steps["preprocess"]
    trained_tree = pipeline.named_steps["model"]
    feature_names = [
        format_feature_name(name)
        for name in preprocessor.get_feature_names_out()
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


def index(request):
    original_penguins = load_penguins()
    penguins = load_clean_penguins()
    result = train_decision_tree(penguins)
    trained_tree = result["pipeline"].named_steps["model"]

    context = {
        "original_row_count": len(original_penguins),
        "row_count": len(penguins),
        "removed_row_count": len(original_penguins) - len(penguins),
        "train_rows": result["train_rows"],
        "test_rows": result["test_rows"],
        "accuracy_percent": round(result["accuracy"] * 100, 2),
        "leaf_count": trained_tree.get_n_leaves(),
        "tree_image_url": save_tree_plot(result["pipeline"]),
        "target_column": TARGET_COLUMN.title(),
        "target_classes": ["Adelie", "Chinstrap", "Gentoo"],
        "categorical_features": [
            feature.replace("_", " ").title()
            for feature in CATEGORICAL_FEATURES
        ],
        "numeric_features": [
            feature.replace("_", " ").title()
            for feature in NUMERIC_FEATURES
        ],
    }
    return render(request, "project2/index.html", context)
