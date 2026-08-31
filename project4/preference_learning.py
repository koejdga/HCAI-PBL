from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .feature_representation import get_feature_columns, humanize_feature_name


@dataclass(frozen=True)
class Comparison:
    preferred_id: str
    other_id: str
    source: str = "pairwise"


@dataclass
class PreferenceFit:
    weights: np.ndarray
    feature_columns: list
    comparisons_used: int
    model_type: str = "Bradley-Terry via logistic regression on feature differences"
    assumptions: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    source_counts: dict = field(default_factory=dict)


@dataclass
class RecommendationResult:
    recommendations: list
    fit: PreferenceFit
    assumptions: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


def normalize_movie_id(value):
    if value is None or pd.isnull(value):
        return ""
    return str(value).replace("\xa0", " ").strip()


def _value_from(payload, keys):
    for key in keys:
        if key in payload and payload[key] not in (None, ""):
            return payload[key]
    return None


def pairwise_choices_to_comparisons(pairwise_choices):
    """
    Converts UI pairwise choices into winner-loser comparisons.

    Supported payloads:
    - {"preferred_id": "...", "other_id": "..."}
    - {"winner_id": "...", "loser_id": "..."}
    - {"selected_movie_id": "...", "rejected_movie_id": "..."}
    - {"left_id": "...", "right_id": "...", "choice": "left"|"right"|movie_id}
    """
    comparisons = []
    for choice in pairwise_choices or []:
        if isinstance(choice, Comparison):
            comparisons.append(choice)
            continue

        preferred_id = _value_from(choice, ["preferred_id", "winner_id", "selected_movie_id"])
        other_id = _value_from(choice, ["other_id", "loser_id", "rejected_movie_id"])

        if not preferred_id and {"left_id", "right_id", "choice"}.issubset(choice):
            left_id = normalize_movie_id(choice["left_id"])
            right_id = normalize_movie_id(choice["right_id"])
            selected = normalize_movie_id(choice["choice"])
            if selected == "left" or selected == left_id:
                preferred_id, other_id = left_id, right_id
            elif selected == "right" or selected == right_id:
                preferred_id, other_id = right_id, left_id

        preferred_id = normalize_movie_id(preferred_id)
        other_id = normalize_movie_id(other_id)
        if preferred_id and other_id and preferred_id != other_id:
            comparisons.append(Comparison(preferred_id, other_id, source="pairwise"))

    return comparisons


def ranking_to_comparisons(ranking, max_pairs=None):
    """
    Extends Bradley-Terry to a full ranking by adding all ordered item pairs.

    A ranking [i1, i2, i3] becomes i1 > i2, i1 > i3, and i2 > i3.
    This is the lean pairwise-decomposition form of a ranking likelihood and is
    compatible with the same utility U(x) = w^T x used for pairwise choices.
    """
    if isinstance(ranking, dict):
        ranking = ranking.get("ordered_ids") or ranking.get("ranking") or ranking.get("movie_ids") or []

    ordered_ids = []
    for item in ranking or []:
        if isinstance(item, dict):
            item_id = _value_from(item, ["movie_id", "id", "title", "movie_title"])
        else:
            item_id = item
        item_id = normalize_movie_id(item_id)
        if item_id and item_id not in ordered_ids:
            ordered_ids.append(item_id)

    comparisons = []
    for preferred_index, preferred_id in enumerate(ordered_ids):
        for other_id in ordered_ids[preferred_index + 1:]:
            comparisons.append(Comparison(preferred_id, other_id, source="ranking"))
            if max_pairs is not None and len(comparisons) >= max_pairs:
                return comparisons
    return comparisons


def collect_comparisons(pairwise_choices=None, rankings=None):
    comparisons = pairwise_choices_to_comparisons(pairwise_choices)

    if rankings:
        is_single_ranking = isinstance(rankings, dict)
        if isinstance(rankings, list) and rankings:
            is_single_ranking = is_single_ranking or not isinstance(rankings[0], (list, tuple, dict))
            is_single_ranking = is_single_ranking or (
                isinstance(rankings[0], dict)
                and any(key in rankings[0] for key in ["movie_id", "id", "title", "movie_title"])
            )

        ranking_payloads = [rankings] if is_single_ranking else rankings
        for ranking in ranking_payloads:
            comparisons.extend(ranking_to_comparisons(ranking))

    return comparisons


def _feature_table_by_id(feature_table, movie_id_column="movie_id"):
    if movie_id_column in feature_table.columns:
        working = feature_table.copy()
    elif "movie_title" in feature_table.columns:
        working = feature_table.copy()
        working[movie_id_column] = working["movie_title"]
    else:
        working = feature_table.copy()
        working[movie_id_column] = working.index.astype(str)

    working[movie_id_column] = working[movie_id_column].apply(normalize_movie_id)
    working = working[working[movie_id_column] != ""]
    return working.drop_duplicates(subset=[movie_id_column]).set_index(movie_id_column, drop=False)


def _numeric_feature_values(feature_table, feature_columns):
    values = feature_table[feature_columns].apply(pd.to_numeric, errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return values.astype(float)


def comparisons_to_training_data(feature_table, comparisons, feature_columns=None, movie_id_column="movie_id"):
    """
    Creates symmetric training examples so logistic regression receives both labels.
    """
    indexed = _feature_table_by_id(feature_table, movie_id_column=movie_id_column)
    if feature_columns is None:
        feature_columns = get_feature_columns(indexed)

    numeric_features = _numeric_feature_values(indexed, feature_columns)
    rows = []
    labels = []
    warnings = []
    source_counts = {}
    used = 0

    for comparison in comparisons or []:
        preferred_id = normalize_movie_id(comparison.preferred_id)
        other_id = normalize_movie_id(comparison.other_id)
        if preferred_id not in numeric_features.index or other_id not in numeric_features.index:
            warnings.append(f"Skipped comparison with unavailable movie: {preferred_id} > {other_id}.")
            continue

        diff = numeric_features.loc[preferred_id].to_numpy() - numeric_features.loc[other_id].to_numpy()
        if not np.isfinite(diff).all():
            warnings.append(f"Skipped comparison with non-finite feature values: {preferred_id} > {other_id}.")
            continue

        rows.append(diff)
        labels.append(1)
        rows.append(-diff)
        labels.append(0)
        used += 1
        source_counts[comparison.source] = source_counts.get(comparison.source, 0) + 1

    if not rows:
        return np.empty((0, len(feature_columns))), np.array([]), 0, warnings, source_counts

    return np.vstack(rows), np.array(labels), used, warnings, source_counts


def fit_preference_vector(feature_table, comparisons, feature_columns=None, movie_id_column="movie_id"):
    """
    Estimates w for U(x) = w^T x using a Bradley-Terry style logistic model.
    """
    indexed = _feature_table_by_id(feature_table, movie_id_column=movie_id_column)
    if feature_columns is None:
        feature_columns = get_feature_columns(indexed)

    assumptions = [
        "Pairwise choices are interpreted as winner-loser comparisons.",
        "Full rankings are decomposed into all ordered Bradley-Terry comparisons.",
        "The fitted vector is session-local and should be treated as exploratory with few interactions.",
    ]
    warnings = []

    if not feature_columns:
        warnings.append("No feature columns were available, so a zero preference vector was returned.")
        return PreferenceFit(np.array([]), [], 0, assumptions=assumptions, warnings=warnings)

    X, y, used, data_warnings, source_counts = comparisons_to_training_data(
        indexed,
        comparisons,
        feature_columns=feature_columns,
        movie_id_column=movie_id_column,
    )
    warnings.extend(data_warnings)

    if used == 0:
        warnings.append("No valid preferences were submitted, so recommendations use a neutral preference vector.")
        return PreferenceFit(
            np.zeros(len(feature_columns)),
            list(feature_columns),
            0,
            assumptions=assumptions,
            warnings=warnings,
            source_counts=source_counts,
        )

    if used < 5:
        warnings.append("Only a few preference comparisons are available; recommendations may be unstable.")

    model = LogisticRegression(
        fit_intercept=False,
        solver="liblinear",
        C=1.0,
        max_iter=1000,
        random_state=42,
    )
    try:
        model.fit(X, y)
        weights = model.coef_[0].astype(float)
    except ValueError as exc:
        warnings.append(f"Model fitting fell back to average preference differences: {exc}")
        weights = X[y == 1].mean(axis=0) if np.any(y == 1) else np.zeros(len(feature_columns))

    return PreferenceFit(
        weights=weights,
        feature_columns=list(feature_columns),
        comparisons_used=used,
        assumptions=assumptions,
        warnings=warnings,
        source_counts=source_counts,
    )


def score_movies(feature_table, fit, movie_id_column="movie_id"):
    """
    Scores each movie by utility U(x) = w^T x.
    """
    indexed = _feature_table_by_id(feature_table, movie_id_column=movie_id_column)
    feature_columns = fit.feature_columns
    if len(fit.weights) != len(feature_columns):
        raise ValueError("Preference weight length must match feature column count.")

    numeric_features = _numeric_feature_values(indexed, feature_columns)
    scored = indexed.copy()
    scored["utility_score"] = numeric_features.to_numpy().dot(fit.weights)
    return scored.reset_index(drop=True)


def explain_recommendation(row, fit, feature_labels=None, top_k=3):
    """
    Returns the top positive feature contributions for a recommended movie.
    """
    feature_labels = feature_labels or {}
    contributions = []
    for feature_name, weight in zip(fit.feature_columns, fit.weights):
        value = pd.to_numeric(row.get(feature_name, 0.0), errors="coerce")
        if pd.isnull(value):
            value = 0.0
        contribution = float(value) * float(weight)
        if contribution > 0:
            contributions.append(
                {
                    "feature": feature_name,
                    "label": feature_labels.get(feature_name, humanize_feature_name(feature_name)),
                    "value": round(float(value), 3),
                    "contribution": round(contribution, 4),
                }
            )

    contributions.sort(key=lambda item: item["contribution"], reverse=True)
    return contributions[:top_k]


def recommend_movies(
    feature_table,
    fit,
    shown_movie_ids=None,
    top_n=10,
    feature_labels=None,
    movie_id_column="movie_id",
):
    """
    Scores candidates and excludes movies already shown during elicitation.
    """
    shown_movie_ids = {normalize_movie_id(movie_id) for movie_id in (shown_movie_ids or [])}
    scored = score_movies(feature_table, fit, movie_id_column=movie_id_column)
    scored[movie_id_column] = scored[movie_id_column].apply(normalize_movie_id)

    candidates = scored[~scored[movie_id_column].isin(shown_movie_ids)].copy()
    warnings = []
    if candidates.empty:
        warnings.append("No unseen candidate movies were available for recommendation.")

    candidates = candidates.sort_values("utility_score", ascending=False).head(top_n)

    recommendations = []
    for _, row in candidates.iterrows():
        movie_id = row[movie_id_column]
        recommendations.append(
            {
                "movie_id": movie_id,
                "title": row.get("movie_title", movie_id),
                "utility_score": round(float(row["utility_score"]), 4),
                "explanations": explain_recommendation(row, fit, feature_labels=feature_labels),
            }
        )

    assumptions = [
        "Movies already shown during pairwise or ranking elicitation are excluded.",
        "Explanation chips show the largest positive feature contributions to U(x).",
        "Results depend on the sampled movies and the current session's submitted choices.",
    ]

    return RecommendationResult(
        recommendations=recommendations,
        fit=fit,
        assumptions=assumptions,
        warnings=warnings,
    )


def learn_and_recommend(
    feature_table,
    pairwise_choices=None,
    rankings=None,
    shown_movie_ids=None,
    top_n=10,
    feature_columns=None,
    feature_labels=None,
    movie_id_column="movie_id",
):
    """
    End-to-end helper for Django views: choices in, recommendations out.
    """
    comparisons = collect_comparisons(pairwise_choices=pairwise_choices, rankings=rankings)
    fit = fit_preference_vector(
        feature_table,
        comparisons,
        feature_columns=feature_columns,
        movie_id_column=movie_id_column,
    )
    return recommend_movies(
        feature_table,
        fit,
        shown_movie_ids=shown_movie_ids,
        top_n=top_n,
        feature_labels=feature_labels,
        movie_id_column=movie_id_column,
    )
