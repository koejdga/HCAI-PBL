"""Views for Project 2: Interpretable Machine Learning & Rashomon Set."""

import os
from django.conf import settings
from django.core.cache import cache
from django.http import JsonResponse
from django.shortcuts import render
from palmerpenguins import load_penguins

from .core import (
    C_OPTIONS,
    CATEGORICAL_FEATURES,
    CLASS_NAMES,
    CONTENT_MENU_ITEMS,
    COUNTERFACTUAL_ATTEMPTS,
    DATASET_PAGE_SIZE,
    DEFAULT_LAMBDA,
    DEFAULT_THETA,
    DISPLAY_COLUMN_LABELS,
    FEATURE_COLUMNS,
    FEATURE_EFFECT_NUMERIC_FEATURES,
    MAX_LAMBDA,
    MAX_LEAF_OPTIONS,
    MIN_LAMBDA,
    NUMERIC_FEATURES,
    PROJECT2_CACHE_TIMEOUT,
    TARGET_COLUMN,
    TREE_PAGE_SIZE,
    build_counterfactual_rows,
    build_dataset_page,
    build_preprocessor,
    build_tree_datapoint_page,
    calculate_mad_l1_distance,
    compute_ale,
    compute_m_plot,
    compute_odds_ratios,
    compute_pdp,
    extract_tree_rules,
    find_counterfactuals,
    format_display_value,
    format_feature_name,
    generate_tree_svg,
    load_clean_penguins,
    parse_feature_effect_feature,
    parse_lambda,
    parse_theta,
    sample_local_points,
    save_feature_effect_plot,
    save_regression_weight_plot,
    save_tree_plot,
    split_penguin_data,
    trace_penguin_path,
    train_decision_tree,
    train_logistic_regression,
    train_regression_candidates,
    train_tree_candidates,
)

__all__ = [
    "C_OPTIONS",
    "CATEGORICAL_FEATURES",
    "CLASS_NAMES",
    "CONTENT_MENU_ITEMS",
    "COUNTERFACTUAL_ATTEMPTS",
    "DATASET_PAGE_SIZE",
    "DEFAULT_LAMBDA",
    "DEFAULT_THETA",
    "DISPLAY_COLUMN_LABELS",
    "FEATURE_COLUMNS",
    "FEATURE_EFFECT_NUMERIC_FEATURES",
    "MAX_LAMBDA",
    "MAX_LEAF_OPTIONS",
    "MIN_LAMBDA",
    "NUMERIC_FEATURES",
    "PROJECT2_CACHE_TIMEOUT",
    "TARGET_COLUMN",
    "TREE_PAGE_SIZE",
    "build_counterfactual_rows",
    "build_dataset_page",
    "build_preprocessor",
    "build_tree_datapoint_page",
    "calculate_mad_l1_distance",
    "compute_ale",
    "compute_m_plot",
    "compute_odds_ratios",
    "compute_pdp",
    "extract_tree_rules",
    "find_counterfactuals",
    "format_display_value",
    "format_feature_name",
    "generate_tree_svg",
    "index",
    "load_clean_penguins",
    "parse_feature_effect_feature",
    "parse_lambda",
    "parse_theta",
    "sample_local_points",
    "save_feature_effect_plot",
    "save_regression_weight_plot",
    "save_tree_plot",
    "split_penguin_data",
    "trace_penguin_path",
    "train_decision_tree",
    "train_logistic_regression",
    "train_regression_candidates",
    "train_tree_candidates",
]


def index(request):
    original_penguins = load_penguins()
    penguins = load_clean_penguins()

    lambda_value = parse_lambda(request.GET.get("lambda"))
    theta_value = parse_theta(request.GET.get("theta"))
    model_type = request.GET.get("model-type", "decision-tree")
    update_scope = request.GET.get("update-scope", "all")
    counterfactuals_desired_class = request.GET.get("desired-class", "adelie")
    feature_effect_feature = parse_feature_effect_feature(
        request.GET.get("feature-effect-feature")
    )
    locked_features = set(request.GET.getlist("locked-features"))

    model_cache_key = f"project2:model-candidates:{model_type}:{lambda_value:.3f}:{theta_value:.3f}"
    cached_model = cache.get(model_cache_key)
    if cached_model is None:
        if model_type == "logistic-regression":
            cached_model = train_regression_candidates(
                penguins, lambda_value, theta_value, return_rashomon=True
            )
        else:
            cached_model = train_tree_candidates(
                penguins, lambda_value, theta_value, return_rashomon=True
            )
        cache.set(model_cache_key, cached_model, PROJECT2_CACHE_TIMEOUT)

    active_result, active_candidates, rashomon_info = cached_model
    active_pipeline = active_result["pipeline"]

    # 1. Dedicated Compact Datapoint Explorer Pagination for Tree
    tree_datapoint_context = build_tree_datapoint_page(penguins, request)
    traced_row_data = tree_datapoint_context["traced_row"]

    decision_trail = None
    highlighted_nodes = None
    highlighted_edges = None

    if model_type != "logistic-regression":
        if traced_row_data is not None:
            decision_trail = trace_penguin_path(
                active_pipeline, traced_row_data["values"]
            )
            highlighted_nodes = decision_trail["visited_nodes"]
            highlighted_edges = decision_trail["visited_edges"]
        tree_svg = generate_tree_svg(
            active_pipeline, highlighted_nodes, highlighted_edges
        )
        tree_rules = extract_tree_rules(active_pipeline)
    else:
        tree_svg = ""
        tree_rules = []

    # 2. Counterfactuals Pagination Context
    dataset_context = build_dataset_page(penguins, request)
    needs_counterfactuals = update_scope in {"all", "counterfactuals"}
    needs_effects = update_scope in {"all", "effects"}

    counterfactual_rows = []
    counterfactual_attempted_rows = 0
    pdp_image_url = None
    mplot_image_url = None
    ale_image_url = None

    if needs_counterfactuals:
        if dataset_context["selected_dataset_row"] is not None:
            original_x = {
                column: dataset_context["selected_dataset_row"]["values"][column]
                for column in FEATURE_COLUMNS
            }
        else:
            original_x = penguins.iloc[0][FEATURE_COLUMNS].to_dict()

        target_match_str = counterfactuals_desired_class.capitalize()
        locked_cache_suffix = "_".join(sorted(locked_features))
        counterfactual_cache_key = (
            "project2:counterfactuals:"
            f"{model_type}:{lambda_value:.3f}:{counterfactuals_desired_class}:"
            f"{dataset_context['selected_row_id']}:{locked_cache_suffix}"
        )
        cached_counterfactuals = cache.get(counterfactual_cache_key)
        if cached_counterfactuals is None:
            cached_counterfactuals = find_counterfactuals(
                active_pipeline,
                penguins,
                original_x,
                target_match_str,
                locked_features=locked_features,
            )
            cache.set(
                counterfactual_cache_key,
                cached_counterfactuals,
                PROJECT2_CACHE_TIMEOUT,
            )
        counterfactual_rows, counterfactual_attempted_rows = cached_counterfactuals

    if needs_effects:
        effects_cache_key = (
            f"project2:effects:{model_type}:{lambda_value:.3f}:{feature_effect_feature}"
        )
        cached_effects = cache.get(effects_cache_key)
        if cached_effects is None:
            cached_effects = {
                "pdp": compute_pdp(active_pipeline, penguins, feature_effect_feature),
                "mplot": compute_m_plot(active_pipeline, penguins, feature_effect_feature),
                "ale": compute_ale(active_pipeline, penguins, feature_effect_feature),
            }
            cache.set(effects_cache_key, cached_effects, PROJECT2_CACHE_TIMEOUT)

        pdp_image_url = save_feature_effect_plot(
            cached_effects["pdp"],
            "pdp",
            feature_effect_feature,
            model_type,
            lambda_value,
        )
        mplot_image_url = save_feature_effect_plot(
            cached_effects["mplot"],
            "mplot",
            feature_effect_feature,
            model_type,
            lambda_value,
        )
        ale_image_url = save_feature_effect_plot(
            cached_effects["ale"],
            "ale",
            feature_effect_feature,
            model_type,
            lambda_value,
        )

    weight_plot_url = None
    odds_ratios_data = []

    if model_type == "logistic-regression":
        regression_result = active_result
        regression_candidates = active_candidates
        selected_model_description = (
            f"Selected logistic regression model: C = {regression_result['C']}, "
            f"{int(regression_result['non_zero_weights'])} non-zero weights, "
            f"selection score {round(regression_result['selection_score'], 4)}."
        )
        complexity_description = (
            "Number of non-zero coefficients in the trained logistic regression model. "
            "Fewer active weights usually means a simpler model."
        )
        weight_plot_url = save_regression_weight_plot(
            regression_result["pipeline"],
            regression_result["C"],
            lambda_value,
        )
        odds_ratios_data = compute_odds_ratios(regression_result["pipeline"])

        model_data = {
            "train_rows": int(regression_result["train_rows"]),
            "test_rows": int(regression_result["test_rows"]),
            "accuracy_percent": float(round(regression_result["accuracy"] * 100, 2)),
            "complexity_count": int(regression_result["non_zero_weights"]),
            "complexity_label": "Non-Zero Weights",
            "complexity_description": complexity_description,
            "selection_score": float(round(regression_result["selection_score"], 4)),
            "tree_image_url": None,
            "weight_plot_url": weight_plot_url,
            "odds_ratios": odds_ratios_data,
            "selected_model_description": selected_model_description,
            "candidate_models": [
                {
                    "param_label": f"C = {candidate['C']}",
                    "complexity_count": int(candidate["non_zero_weights"]),
                    "accuracy_percent": float(round(candidate["accuracy"] * 100, 2)),
                    "selection_score": float(round(candidate["selection_score"], 4)),
                    "in_rashomon": candidate.get("in_rashomon", False),
                    "selected": candidate is regression_result,
                }
                for candidate in regression_candidates
            ],
        }
    else:
        tree_result = active_result
        tree_candidates = active_candidates
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
            "tree_image_url": save_tree_plot(
                tree_result["pipeline"],
                int(tree_result["leaf_count"]),
                create_if_missing=update_scope != "model",
            ),
            "tree_svg": tree_svg,
            "selected_model_description": selected_model_description,
            "candidate_models": [
                {
                    "param_label": f"Max Leaves = {int(candidate['max_leaf_nodes'])}",
                    "complexity_count": int(candidate["leaf_count"]),
                    "accuracy_percent": float(round(candidate["accuracy"] * 100, 2)),
                    "selection_score": float(round(candidate["selection_score"], 4)),
                    "in_rashomon": candidate.get("in_rashomon", False),
                    "selected": candidate is tree_result,
                }
                for candidate in tree_candidates
            ],
        }

    model_data["rashomon_info"] = rashomon_info
    model_data["dataset_columns"] = dataset_context["dataset_columns"]
    model_data["dataset_column_labels"] = dataset_context["dataset_column_labels"]
    model_data["dataset_page_rows"] = list(dataset_context["dataset_page"].object_list)
    model_data["tree_page_rows"] = list(tree_datapoint_context["tree_page"].object_list)
    tree_page_obj = tree_datapoint_context["tree_page"]
    model_data["tree_page_info"] = {
        "number": tree_page_obj.number,
        "num_pages": tree_page_obj.paginator.num_pages,
        "has_previous": tree_page_obj.has_previous(),
        "previous_page_number": tree_page_obj.previous_page_number()
        if tree_page_obj.has_previous()
        else None,
        "has_next": tree_page_obj.has_next(),
        "next_page_number": tree_page_obj.next_page_number()
        if tree_page_obj.has_next()
        else None,
        "page_links": tree_datapoint_context["tree_page_links"],
    }
    model_data["decision_trail"] = decision_trail
    model_data["feature_effect_feature"] = feature_effect_feature
    model_data["feature_effect_feature_label"] = DISPLAY_COLUMN_LABELS[
        feature_effect_feature
    ]
    model_data["selected_row_id"] = dataset_context["selected_row_id"]
    model_data["selected_dataset_row_display_values"] = dataset_context[
        "selected_dataset_row_display_values"
    ]

    if needs_counterfactuals:
        model_data["counterfactual_rows"] = counterfactual_rows
        model_data["counterfactual_attempted_rows"] = counterfactual_attempted_rows

    if needs_effects:
        model_data["pdp_image_url"] = pdp_image_url
        model_data["mplot_image_url"] = mplot_image_url
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
        "mplot_image_url": mplot_image_url,
        "ale_image_url": ale_image_url,
        "lambda_value": lambda_value,
        "theta_value": theta_value,
        "model_type": model_type,
        "selected_model_description": selected_model_description,
        "complexity_description": complexity_description,
        "rashomon_info": rashomon_info,
        "tree_svg": tree_svg,
        "tree_rules": tree_rules,
        "decision_trail": decision_trail,
        "weight_plot_url": weight_plot_url,
        "odds_ratios_data": odds_ratios_data,
        "locked_features": locked_features,
        **model_data,
        "output_classes": [
            {"value": "adelie", "label": "Adelie"},
            {"value": "gentoo", "label": "Gentoo"},
            {"value": "chinstrap", "label": "Chinstrap"},
        ],
        "desired_class": counterfactuals_desired_class,
        **dataset_context,
        **tree_datapoint_context,
        "counterfactual_rows": counterfactual_rows,
        "DISPLAY_COLUMN_LABELS": DISPLAY_COLUMN_LABELS,
        "content_menu_items": CONTENT_MENU_ITEMS,
    }
    return render(request, "project2/index.html", context)
