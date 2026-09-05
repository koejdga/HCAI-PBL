"""Counterfactual generation using local sampling, distance weighting, and visual diffs."""

import numpy as np
import pandas as pd

from .constants import (
    CATEGORICAL_FEATURES,
    COUNTERFACTUAL_ATTEMPTS,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)
from .dataset import format_display_value


def build_counterfactual_rows(synthetic_df, original_x):
    """Build counterfactual rows with visual diff tags and readable values."""
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
        diffs = {}
        for col in FEATURE_COLUMNS:
            orig_val = original_x[col]
            new_val = row[col]
            if col in NUMERIC_FEATURES:
                delta = float(new_val) - float(orig_val)
                diffs[col] = {
                    "changed": abs(delta) > 0.05,
                    "delta_text": f"{delta:+.1f}" if abs(delta) > 0.05 else "",
                }
            else:
                diffs[col] = {
                    "changed": str(new_val) != str(orig_val),
                    "delta_text": f"{orig_val} → {new_val}" if str(new_val) != str(orig_val) else "",
                }

        rows.append(
            {
                "id": str(row_id),
                "values": values,
                "display_values": display_values,
                "diffs": diffs,
                "distance": round(row["distance"], 2),
            }
        )
    return rows


def sample_local_points(df_clean, original_x, N=1000, variance_scale=0.1, locked_features=None):
    """Generates N local random variations around original_x, respecting locked immutable features."""
    if locked_features is None:
        locked_features = set()

    sampled_data = pd.DataFrame([original_x] * N).reset_index(drop=True)

    # 1. Noise for Numeric Features
    for col in NUMERIC_FEATURES:
        if col in locked_features:
            continue
        col_std = df_clean[col].std()
        noise = np.random.normal(loc=0.0, scale=col_std * variance_scale, size=N)
        sampled_data[col] = sampled_data[col] + noise

    # 2. Noise for Categorical Features
    for col in CATEGORICAL_FEATURES:
        if col in locked_features:
            continue
        unique_categories = list(df_clean[col].unique())
        flip_mask = np.random.rand(N) < 0.15
        if flip_mask.any():
            random_cats = np.random.choice(unique_categories, size=int(flip_mask.sum()))
            sampled_data.loc[flip_mask, col] = random_cats

    return sampled_data


def calculate_mad_l1_distance(original_x, synthetic_df, penguins_df):
    """Ranks synthetic points by MAD-weighted L1 distance for numeric features and categorical mismatch penalty."""
    is_single = isinstance(synthetic_df, (dict, pd.Series))
    if isinstance(synthetic_df, dict):
        synthetic_df = pd.DataFrame([synthetic_df])
    elif isinstance(synthetic_df, pd.Series):
        synthetic_df = pd.DataFrame([synthetic_df])

    mads = {}
    for col in NUMERIC_FEATURES:
        median = penguins_df[col].median()
        mad = np.median(np.abs(penguins_df[col] - median))
        mads[col] = mad if mad > 0 else 0.001

    distances = np.zeros(len(synthetic_df))
    for col in NUMERIC_FEATURES:
        distances += np.abs(synthetic_df[col] - original_x[col]) / mads[col]

    # Include categorical mismatch in distance metric (Task 4 requirement)
    for col in CATEGORICAL_FEATURES:
        cat_mismatch = (synthetic_df[col] != original_x[col]).astype(float) * 1.0
        distances += cat_mismatch

    return float(distances[0]) if is_single else distances


def find_counterfactuals(pipeline, penguins, original_x, desired_class, locked_features=None):
    """Retry local sampling with wider noise if no counterfactual is found."""
    attempted_rows = 0
    for attempt in COUNTERFACTUAL_ATTEMPTS:
        synthetic_points = sample_local_points(
            penguins,
            original_x,
            N=attempt["N"],
            variance_scale=attempt["variance_scale"],
            locked_features=locked_features,
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
        return build_counterfactual_rows(top_counterfactuals, original_x), attempted_rows

    return [], attempted_rows
