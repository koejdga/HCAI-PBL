"""Dataset loading, preprocessing, display formatting, and parameter parsing."""

import numpy as np
import pandas as pd
from palmerpenguins import load_penguins
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from .constants import (
    CATEGORICAL_FEATURES,
    DEFAULT_LAMBDA,
    DEFAULT_THETA,
    FEATURE_COLUMNS,
    FEATURE_EFFECT_NUMERIC_FEATURES,
    MAX_LAMBDA,
    MIN_LAMBDA,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)


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


def parse_lambda(value):
    """Convert the slider value into a safe number between zero and one."""
    try:
        lambda_value = float(value)
    except (TypeError, ValueError):
        return DEFAULT_LAMBDA

    return min(max(lambda_value, MIN_LAMBDA), MAX_LAMBDA)


def parse_theta(value):
    """Convert error tolerance theta into a safe float between 0 and 0.20."""
    try:
        theta_val = float(value)
    except (TypeError, ValueError):
        return DEFAULT_THETA
    return min(max(theta_val, 0.0), 0.20)


def parse_feature_effect_feature(value):
    if value in FEATURE_EFFECT_NUMERIC_FEATURES:
        return value
    return FEATURE_EFFECT_NUMERIC_FEATURES[0]
