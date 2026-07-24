import copy
import os
import time
import random
from collections import OrderedDict
from django.conf import settings
from django.http import HttpResponse
from django.shortcuts import redirect, render
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from .core.learning_to_defer import L2DClassifier, L2DNeuralNetwork, L2DLinearModel
from django.http import JsonResponse
import numpy as np
from django.core.serializers.json import DjangoJSONEncoder
from django.template.loader import render_to_string

class NumpyJSONEncoder(DjangoJSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

from .core.utils import (
    CLASS_NAMES,
    CLASS_IDS,
    DEFAULT_TRAIN_SIZE,
    DEFAULT_TEST_SIZE,
    load_ag_news_dataset,
    parse_sample_size,
    parse_float,
    save_bar_plot,
    save_active_learning_scatter_plot,
    artifact_dir,
)
from .core.deferral import (
    train_baseline_classifier,
    evaluate_simulated_expert,
    evaluate_trivial_expert,
    evaluate_confidence_threshold_defer,
    evaluate_competence_aware_defer,
    class_metric_rows,
    DEFAULT_DEFER_RATE,
    DEFAULT_QUERY_BUDGET,
)
from .core.active_learning import (
    active_learning_queries,
    build_human_expert_context,
    compare_active_learning_strategies,
    CLASS_IDS,
)

REPORT_FILENAME = "project3_report.pdf"
PROJECT3_RESULT_CACHE_MAX = 8
PROJECT3_RESULT_CACHE = OrderedDict()


def _cache_key(train_size, test_size, defer_rate, query_budget, expert_configs=None):
    configs_tuple = ()
    if expert_configs:
        configs_tuple = tuple(
            (
                cfg.get("prefix"),
                cfg.get("type"),
                tuple(cfg.get("fields", [])),
                cfg.get("competence_level"),
                cfg.get("cost_presence"),
                cfg.get("cost"),
            )
            for cfg in expert_configs
        )
    return (train_size, test_size, round(float(defer_rate), 6), query_budget, configs_tuple)


def _get_cached_project3_results(cache_key):
    if cache_key in PROJECT3_RESULT_CACHE:
        PROJECT3_RESULT_CACHE.move_to_end(cache_key)
        return PROJECT3_RESULT_CACHE[cache_key]
    return None


def _store_cached_project3_results(cache_key, payload):
    PROJECT3_RESULT_CACHE[cache_key] = payload
    PROJECT3_RESULT_CACHE.move_to_end(cache_key)
    while len(PROJECT3_RESULT_CACHE) > PROJECT3_RESULT_CACHE_MAX:
        PROJECT3_RESULT_CACHE.popitem(last=False)


PROJECT3_BASE_CACHE = OrderedDict()


def _get_cached_base_context(cache_key):
    if cache_key in PROJECT3_BASE_CACHE:
        PROJECT3_BASE_CACHE.move_to_end(cache_key)
        return PROJECT3_BASE_CACHE[cache_key]
    return None


def _store_cached_base_context(cache_key, payload):
    PROJECT3_BASE_CACHE[cache_key] = payload
    PROJECT3_BASE_CACHE.move_to_end(cache_key)
    while len(PROJECT3_BASE_CACHE) > PROJECT3_RESULT_CACHE_MAX:
        PROJECT3_BASE_CACHE.popitem(last=False)


def sample_examples(examples, count=5):
    rng = random.Random(42)
    selected = rng.sample(examples, min(count, len(examples)))
    return [
        {
            "label": CLASS_NAMES[example["label"]],
            "text": example["text"],
        }
        for example in selected
    ]


def _compute_project3_results(train_examples, test_examples, defer_rate, query_budget, expert_configs=None, baseline=None):
    if baseline is None:
        baseline = train_baseline_classifier(train_examples, test_examples)

    if expert_configs:
        expert_results = [
            evaluate_expert_from_settings(test_examples, cfg) for cfg in expert_configs
        ]
    else:
        expert_results = [evaluate_simulated_expert(test_examples)]

    expert_accuracy_avg = (
        sum(r["accuracy"] for r in expert_results) / len(expert_results)
    )

    active_learning_by_strategy = {}
    scatter_plots = {}
    for strat in ["balanced_uncertainty", "uncertainty", "random"]:
        al_data = active_learning_queries(
            train_examples, baseline, query_budget, strategy=strat, expert_configs=expert_configs
        )
        active_learning_by_strategy[strat] = al_data
        scatter_plots[strat] = save_active_learning_scatter_plot(
            train_examples, al_data, filename=f"active_learning_scatter_{strat}.png"
        )

    active_learning = active_learning_by_strategy["balanced_uncertainty"]
    strategy_rows = compare_active_learning_strategies(
        train_examples, test_examples, baseline, query_budget, expert_configs=expert_configs
    )
    confidence_team = evaluate_confidence_threshold_defer(
        test_examples, baseline, expert_results, defer_rate, expert_configs=expert_configs
    )
    competence_team = evaluate_competence_aware_defer(
        test_examples, baseline, expert_results, active_learning["competence_by_class"], expert_configs=expert_configs
    )

    print("DEFERRAL TOTAL: ", competence_team["deferred_total"])

    l2d_linear_classifier = L2DClassifier(
        model_type=L2DLinearModel,
        policy_name="Learning to Defer (with linear model)",
        tfidf_max_features=5000,
        query_cost=0.0,
    )
    l2d_linear_classifier.fit(train_examples, expert_configs=expert_configs)
    l2d_linear_results = l2d_linear_classifier.predict_and_evaluate(test_examples, expert_configs=expert_configs)

    l2d_nn_classifier = L2DClassifier(
        model_type=L2DNeuralNetwork,
        policy_name="Leaning to Defer (with neural network)",
        tfidf_max_features=5000,
        query_cost=0.0,
    )
    l2d_nn_classifier.fit(train_examples, expert_configs=expert_configs)
    l2d_nn_results = l2d_nn_classifier.predict_and_evaluate(test_examples, expert_configs=expert_configs)

    from .core.deferral import get_expert_cost
    if expert_configs:
        expert_costs = [get_expert_cost(cfg) for cfg in expert_configs]
        expert_cost_avg = sum(expert_costs) / len(expert_costs)
    else:
        expert_cost_avg = 0.0

    expert_metrics = []
    configs_to_use = expert_configs if expert_configs else [{"type": "REALISTIC", "competence_level": "more-competent", "fields": [], "cost_presence": "absent", "cost": None}]
    for index, settings in enumerate(configs_to_use, start=1):
        expert_result = expert_results[index - 1]
        expert_metrics.append(
            {
                "name": f"Expert {index}",
                "type": settings.get("type", "REALISTIC"),
                "accuracy_percent": round(expert_result["accuracy"] * 100, 2),
                "settings": settings,
            }
        )

    policy_rows = [
        {
            "group": "Baselines",
            "name": "Linear classifier only",
            "accuracy_percent": round(baseline["accuracy"] * 100, 2),
            "deferred_total": 0,
            "deferral_rate_percent": 0,
            "useful_defer": 0,
            "harmful_defer": 0,
            "total_cost": 0.0,
            "net_defer": 0,
        },
        {
            "group": "",
            "name": "Expert only" if len(expert_configs) == 1 else "Experts only",
            "accuracy_percent": round(expert_accuracy_avg * 100, 2),
            "deferred_total": len(test_examples),
            "deferral_rate_percent": 100,
            "useful_defer": 0,
            "harmful_defer": 0,
            "total_cost": round(expert_cost_avg * len(test_examples), 2),
            "net_defer": 0,
        },
        {
            "group": "Post-Hoc Routing (Frozen SVM)",
            "name": confidence_team["policy_name"],
            "accuracy_percent": round(confidence_team["accuracy"] * 100, 2),
            "deferred_total": confidence_team["deferred_total"],
            "deferral_rate_percent": round(
                confidence_team["deferred_total"] / len(test_examples) * 100, 2
            ),
            "useful_defer": confidence_team["useful_defer"],
            "harmful_defer": confidence_team["harmful_defer"],
            "total_cost": confidence_team.get("total_cost", 0.0),
            "net_defer": confidence_team["useful_defer"]
            - confidence_team["harmful_defer"],
        },
        {
            "group": "",
            "name": competence_team["policy_name"],
            "accuracy_percent": round(competence_team["accuracy"] * 100, 2),
            "deferred_total": competence_team["deferred_total"],
            "deferral_rate_percent": round(
                competence_team["deferred_total"] / len(test_examples) * 100, 2
            ),
            "useful_defer": competence_team["useful_defer"],
            "harmful_defer": competence_team["harmful_defer"],
            "total_cost": competence_team.get("total_cost", 0.0),
            "net_defer": competence_team["useful_defer"]
            - competence_team["harmful_defer"],
        },
        {
            "group": "L2D (Joint Training)",
            "name": l2d_linear_results["policy_name"],
            "accuracy_percent": round(l2d_linear_results["accuracy"] * 100, 2),
            "deferred_total": l2d_linear_results["deferred_total"],
            "deferral_rate_percent": round(
                l2d_linear_results["deferred_total"] / len(test_examples) * 100, 2
            ),
            "useful_defer": l2d_linear_results["useful_defer"],
            "harmful_defer": l2d_linear_results["harmful_defer"],
            "total_cost": l2d_linear_results.get("total_cost", 0.0),
            "net_defer": l2d_linear_results["useful_defer"]
            - l2d_linear_results["harmful_defer"],
        },
        {
            "group": "",
            "name": l2d_nn_results["policy_name"],
            "accuracy_percent": round(l2d_nn_results["accuracy"] * 100, 2),
            "deferred_total": l2d_nn_results["deferred_total"],
            "deferral_rate_percent": round(
                l2d_nn_results["deferred_total"] / len(test_examples) * 100, 2
            ),
            "useful_defer": l2d_nn_results["useful_defer"],
            "harmful_defer": l2d_nn_results["harmful_defer"],
            "total_cost": l2d_nn_results.get("total_cost", 0.0),
            "net_defer": l2d_nn_results["useful_defer"]
            - l2d_nn_results["harmful_defer"],
        },
    ]

    best_team_row = max(
        [
            row
            for row in policy_rows
            if row["name"] in {"Confidence threshold", "Competence-aware"}
        ],
        key=lambda row: (row["accuracy_percent"], row["net_defer"]),
    )
    selected_team = (
        confidence_team
        if best_team_row["name"] == "Confidence threshold"
        else competence_team
    )

    policies_map = {
        "confidence": confidence_team,
        "competence": competence_team,
        "l2d_linear": l2d_linear_results,
        "l2d_nn": l2d_nn_results,
    }
    allocation_by_policy = {}
    for key, p_data in policies_map.items():
        selected_allocation = p_data.get("query_allocation", {})

        total_correct = 0
        total_queried = 0
        total_cost = 0

        # Initialize your target nested dictionary structure
        allocation_data = {}

        for e_idx in range(len(expert_results)):
            expert_key = f"expert{e_idx + 1}"
            allocation_data[expert_key] = {}
            
            total_correct = 0
            total_queried = 0
            total_cost = 0

            for cid in CLASS_IDS:
                class_name = CLASS_NAMES[cid]
                
                # Pull stats safely
                stats = selected_allocation.get(cid, {}).get(e_idx, {"correct": 0, "queried": 0})
                correct = stats["correct"]
                queried = stats["queried"]
                
                # Calculate calculations
                pct = round(correct / queried * 100, 1) if queried > 0 else 0.0
                
                # TODO: Calculate cost based on your specific requirements
                # e.g., cost = queried * expert_settings[e_idx]["cost"]
                cost = 0  
                
                # Update running totals for this specific expert
                total_correct += correct
                total_queried += queried
                total_cost += cost

                # Map directly to your requested structure
                allocation_data[expert_key][class_name] = {
                    "corrected_queried": f"{correct}/{queried}",
                    "percentage": f"{pct}%",
                    "cost": cost
                }
                
            # After checking all classes, append the calculated "Total" row to this expert
            total_pct = round(total_correct / total_queried * 100, 1) if total_queried > 0 else 0.0
            allocation_data[expert_key]["Total"] = {
                "corrected_queried": f"{total_correct}/{total_queried}",
                "percentage": f"{total_pct}%",
                "cost": total_cost
            }

        allocation_by_policy[key] = allocation_data

    def clean_label_for_plot(label):
        lbl = label.replace("Learning to Defer", "L2D").replace("Leaning to Defer", "L2D")
        lbl = lbl.replace(" (with linear model)", " (Linear)").replace(" (with neural network)", " (NN)")
        return lbl

    accuracy_data = sorted(
        [(row["name"], row["accuracy_percent"]) for row in policy_rows],
        key=lambda x: x[1],
        reverse=True
    )
    accuracy_labels = [clean_label_for_plot(x[0]) for x in accuracy_data]
    accuracy_values = [x[1] for x in accuracy_data]

    model_plot_url = save_bar_plot(
        "model_comparison.png",
        "Model and Team Accuracy",
        accuracy_labels,
        accuracy_values,
        "Accuracy (%)",
        color="#2b76ca",
    )

    has_costs = False
    if expert_configs:
        has_costs = any(
            cfg.get("cost_presence") == "present" and cfg.get("cost") is not None
            for cfg in expert_configs
        )

    cost_plot_url = None
    if has_costs:
        cost_data = sorted(
            [(row["name"], row["total_cost"]) for row in policy_rows],
            key=lambda x: x[1]
        )
        cost_labels = [clean_label_for_plot(x[0]) for x in cost_data]
        cost_values = [x[1] for x in cost_data]
        cost_plot_url = save_bar_plot(
            "cost_comparison.png",
            "Total Cost by Deferral Policy",
            cost_labels,
            cost_values,
            "Total Cost",
            color="#e05e36",
        )

    deferred_data = sorted(
        [(row["name"], row["deferred_total"]) for row in policy_rows],
        key=lambda x: x[1],
        reverse=True
    )
    deferred_labels = [clean_label_for_plot(x[0]) for x in deferred_data]
    deferred_values = [x[1] for x in deferred_data]
    deferred_plot_url = save_bar_plot(
        "deferred_comparison.png",
        "Total Deferred Cases by Policy",
        deferred_labels,
        deferred_values,
        "Deferred Total",
        color="#8e44ad",
    )

    benefit_data = sorted(
        [(row["name"], row["net_defer"]) for row in policy_rows],
        key=lambda x: x[1],
        reverse=True
    )
    benefit_labels = [clean_label_for_plot(x[0]) for x in benefit_data]
    benefit_values = [x[1] for x in benefit_data]
    benefit_plot_url = save_bar_plot(
        "benefit_comparison.png",
        "Net Benefit by Deferral Policy",
        benefit_labels,
        benefit_values,
        "Net Benefit (Useful - Harmful)",
        color="#27ae60",
    )

    return {
        "baseline_accuracy_percent": round(baseline["accuracy"] * 100, 2),
        "expert_accuracy_percent": round(expert_accuracy_avg * 100, 2),
        "team_policy_name": best_team_row["name"],
        "team_accuracy_percent": round(selected_team["accuracy"] * 100, 2),
        "team_deferred_total": selected_team["deferred_total"],
        "team_non_deferred_total": selected_team["non_deferred_total"],
        "team_useful_defer": selected_team["useful_defer"],
        "team_harmful_defer": selected_team["harmful_defer"],
        "active_learning": active_learning,
        "active_learning_by_strategy": active_learning_by_strategy,
        "strategy_rows": strategy_rows,
        "policy_rows": policy_rows,
        "allocation_rows": allocation_by_policy["confidence"],
        "allocation_by_policy": allocation_by_policy,
        "model_plot_url": model_plot_url,
        "cost_plot_url": cost_plot_url,
        "deferred_plot_url": deferred_plot_url,
        "benefit_plot_url": benefit_plot_url,
        "scatter_plot_url": scatter_plots["balanced_uncertainty"],
        "scatter_plots": scatter_plots,
        "baseline_class_rows": class_metric_rows(baseline["per_class"]),
        "expert_class_rows": build_expert_class_rows(expert_results),
        "baseline_confusion_rows": baseline["confusion"],
        "expert_confusion_rows": expert_results[0]["confusion"],
        "sample_examples": sample_examples(test_examples),
        "expert_summaries": build_expert_summaries(expert_configs, expert_results) if expert_configs else [],
        "experts": expert_metrics,
    }


def get_base_experiment_context(train_size, test_size, query_budget):
    dataset = load_ag_news_dataset(train_size, test_size)
    train_examples = dataset["train"]
    test_examples = dataset["test"]

    if query_budget is None:
        query_budget = min(len(train_examples), DEFAULT_QUERY_BUDGET)
    query_budget = min(query_budget, len(train_examples))

    cache_key = (train_size, test_size, query_budget)
    cached = _get_cached_base_context(cache_key)
    if cached is not None:
        return {
            "dataset": dataset,
            "train_examples": train_examples,
            "test_examples": test_examples,
            "baseline": cached["baseline"],
            "query_budget": query_budget,
            "sample_examples": sample_examples(test_examples),
        }

    baseline = train_baseline_classifier(train_examples, test_examples)
    
    cached_payload = {
        "baseline": baseline,
    }
    _store_cached_base_context(cache_key, cached_payload)
    
    return {
        "dataset": dataset,
        "train_examples": train_examples,
        "test_examples": test_examples,
        "baseline": baseline,
        "query_budget": query_budget,
        "sample_examples": sample_examples(test_examples),
    }


def _are_configs_equal(c1, c2):
    if len(c1) != len(c2):
        return False
    for e1, e2 in zip(c1, c2):
        if e1.get("type") != e2.get("type"):
            return False
        if sorted(e1.get("fields") or []) != sorted(e2.get("fields") or []):
            return False
        if e1.get("competence_level") != e2.get("competence_level"):
            return False
        if e1.get("cost_presence") != e2.get("cost_presence"):
            return False
        if e1.get("cost") != e2.get("cost"):
            return False
        if e1.get("query_budget") != e2.get("query_budget"):
            return False
    return True


def build_project3_results(request):
    train_size = parse_sample_size(request.GET.get("train-size"), DEFAULT_TRAIN_SIZE)
    test_size = parse_sample_size(request.GET.get("test-size"), DEFAULT_TEST_SIZE)
    defer_rate = parse_float(request.GET.get("defer-rate"), DEFAULT_DEFER_RATE)
    expert_one_query_budget = parse_sample_size(
        request.GET.get("expert-one-query-budget") or request.GET.get("query-budget"),
        DEFAULT_QUERY_BUDGET
    )
    expert_two_query_budget = parse_sample_size(
        request.GET.get("expert-two-query-budget") or request.GET.get("query-budget"),
        DEFAULT_QUERY_BUDGET
    )
    human_strategy = request.GET.get("human-expert-strategy") or "balanced_uncertainty"
    human_limit = parse_sample_size(request.GET.get("human-expert-budget"), 6)
    
    if hasattr(request, "session"):
        prev_strategy = request.session.get("project3_human_strategy")
        prev_limit = request.session.get("project3_human_limit")
        if prev_strategy != human_strategy or prev_limit != human_limit:
            request.session["project3_human_labels"] = {}
            request.session["project3_human_strategy"] = human_strategy
            request.session["project3_human_limit"] = human_limit
            request.session.modified = True
        
    query_budget = max(expert_one_query_budget, expert_two_query_budget, human_limit)
    expert_configs = parse_requested_experts(request)
    if not expert_configs:
        raise ValueError("No experts have been configured yet. Please configure the expert(s) in the form below and click 'Apply expert's accuracy'.")

    expert_configs[0]["query_budget"] = expert_one_query_budget
    if len(expert_configs) > 1:
        expert_configs[1]["query_budget"] = expert_two_query_budget

    base = get_base_experiment_context(train_size, test_size, query_budget)

    cache_key = _cache_key(train_size, test_size, defer_rate, query_budget, expert_configs)
    cached_payload = _get_cached_project3_results(cache_key)
    if cached_payload is None:
        cached_payload = _compute_project3_results(
            base["train_examples"], base["test_examples"], defer_rate, query_budget, expert_configs, baseline=base["baseline"]
        )
        _store_cached_project3_results(cache_key, cached_payload)

    human_al = cached_payload.get("active_learning_by_strategy", {}).get(human_strategy)
    if human_al is None:
        human_al = active_learning_queries(
            base["train_examples"], base["baseline"], query_budget, strategy=human_strategy, expert_configs=expert_configs
        )

    human_expert = build_human_expert_context(
        request, base["train_examples"], human_al, limit=human_limit
    )

    if hasattr(request, "session"):
        request.session["project3_last_query_params"] = request.GET.dict()
        
        # Save current config to history
        saved_list = request.session.get("project3_saved_configs", [])
        exists = False
        for item in saved_list:
            if _are_configs_equal(item.get("expert_configs", []), expert_configs):
                exists = True
                break
                
        if not exists:
            labels = []
            for idx, cfg in enumerate(expert_configs):
                exp_num = idx + 1
                exp_type = "Realistic" if cfg.get("type") == "REALISTIC" else "Trivial"
                details = []
                if cfg.get("type") == "REALISTIC":
                    if cfg.get("competence_level"):
                        details.append(cfg["competence_level"].replace("-", " "))
                    if cfg.get("cost_presence") == "present" and cfg.get("cost") is not None:
                        details.append(f"cost: {cfg['cost']}")
                else:
                    fields = cfg.get("fields") or []
                    field_names = [CLASS_NAMES[fid] for fid in fields if fid in CLASS_NAMES]
                    if field_names:
                        details.append(f"fields: {', '.join(field_names)}")
                    else:
                        details.append("all fields")
                details.append(f"budget: {cfg.get('query_budget')}")
                labels.append(f"Expert {exp_num} ({exp_type}, {', '.join(details)})")
            
            config_label = " & ".join(labels)
            config_id = f"cfg_{int(time.time())}_{len(saved_list)}"
            
            saved_list.append({
                "id": config_id,
                "label": config_label,
                "params": request.GET.dict(),
                "expert_configs": expert_configs,
            })
            request.session["project3_saved_configs"] = saved_list
        request.session.modified = True

    return {
        "train_size": "all" if train_size is None else train_size,
        "test_size": "all" if test_size is None else test_size,
        "defer_rate": defer_rate,
        "defer_rate_percent": round(defer_rate * 100, 1),
        "query_budget": query_budget,
        "expert_one_query_budget": expert_one_query_budget,
        "expert_two_query_budget": expert_two_query_budget,
        "train_rows": len(base["train_examples"]),
        "test_rows": len(base["test_examples"]),
        "full_train_rows": base["dataset"]["full_train_rows"],
        "full_test_rows": base["dataset"]["full_test_rows"],
        "dataset_source": base["dataset"]["source"],
        "class_names": CLASS_NAMES,
        **copy.deepcopy(cached_payload),
        "human_expert": human_expert,
        "human_expert_strategy": human_strategy,
        "human_expert_budget": human_limit,
        "sample_examples": base["sample_examples"],
        "report_url": "report/",
        "expert_count": len(expert_configs),
        "saved_configs": request.session.get("project3_saved_configs", []) if hasattr(request, "session") else [],
    }


def _parse_expert_field_ids(raw_fields):
    if raw_fields is None:
        return []
    if isinstance(raw_fields, (list, tuple)):
        values = raw_fields
    else:
        values = str(raw_fields).split(",")
    field_ids = []
    for value in values:
        text = str(value).strip()
        if not text or text == "all":
            continue
        try:
            field_id = int(text)
        except ValueError:
            continue
        if field_id in CLASS_IDS and field_id not in field_ids:
            field_ids.append(field_id)
    return field_ids


def parse_expert_settings(request, prefix):
    """Collect all expert form values for one expert prefix (e.g. expert-one)."""
    expert_type = (
        request.GET.get(f"{prefix}-type")
        or request.GET.get(prefix)
        or request.GET.get("expert")
        or "REALISTIC"
    ).upper()
    if expert_type not in {"TRIVIAL", "REALISTIC"}:
        expert_type = "TRIVIAL"

    fields = _parse_expert_field_ids(request.GET.get(f"{prefix}-fields", ""))
    competence_level = request.GET.get(f"{prefix}-competence-level", "") or ""
    cost_presence = request.GET.get(f"{prefix}-cost-presence", "absent") or "absent"
    if cost_presence not in {"absent", "present"}:
        cost_presence = "absent"
    cost_raw = request.GET.get(f"{prefix}-cost", "")
    try:
        cost = float(cost_raw) if cost_raw not in (None, "") else None
    except ValueError:
        cost = None

    return {
        "prefix": prefix,
        "type": expert_type,
        "fields": fields,
        "competence_level": competence_level,
        "cost_presence": cost_presence,
        "cost": cost,
    }


def parse_requested_experts(request):
    """Return settings for one or two experts from the expert-choice form."""
    has_expert_one = (
        request.GET.get("expert-one-type") is not None
        or request.GET.get("expert-one") is not None
        or request.GET.get("expert") is not None
    )
    if not has_expert_one:
        return []

    experts = [parse_expert_settings(request, "expert-one")]
    has_second = (
        request.GET.get("expert-two-type") is not None
        or request.GET.get("expert-two") is not None
    )
    if has_second:
        experts.append(parse_expert_settings(request, "expert-two"))
    return experts


def evaluate_expert_from_settings(test_examples, settings):
    """Run the current evaluate helpers for one expert config."""
    if settings["type"] == "REALISTIC":
        return evaluate_simulated_expert(
            test_examples,
            expert_fields=settings["fields"],
            competence_level=settings["competence_level"] or "more-competent",
        )

    fields = settings["fields"]
    return evaluate_trivial_expert(test_examples, fields)


def build_expert_summaries(expert_settings, expert_results):
    summaries = []
    for index, (settings, result) in enumerate(zip(expert_settings, expert_results), start=1):
        field_names = [CLASS_NAMES[fid] for fid in settings["fields"] if fid in CLASS_NAMES]
        comp_label = ""
        if settings["type"] == "REALISTIC":
            comp_level = settings["competence_level"] or "more-competent"
            comp_label = "more competent" if comp_level == "more-competent" else "less competent"
        
        avg_acc = round(result["accuracy"] * 100, 1)
        
        summaries.append({
            "name": f"Expert {index}",
            "type_label": "Realistic" if settings["type"] == "REALISTIC" else "Trivial",
            "competence_label": comp_label,
            "fields_text": ", ".join(field_names) if field_names else "none",
            "accuracy_percent": avg_acc,
            "cost": settings["cost"] if settings["cost_presence"] == "present" else None,
        })
    return summaries


def build_expert_class_rows(expert_results):
    """One table row per class, with optional second-expert metrics."""
    if not expert_results:
        return []

    per_expert_rows = [
        class_metric_rows(result["per_class"]) for result in expert_results
    ]
    rows = []
    for class_index, first_row in enumerate(per_expert_rows[0]):
        row = {
            "class_name": first_row["class_name"],
            "accuracy_percent": first_row["accuracy_percent"],
            "correct": first_row["correct"],
            "total": first_row["total"],
        }
        if len(per_expert_rows) > 1:
            second_row = per_expert_rows[1][class_index]
            row["expert_2_accuracy_percent"] = second_row["accuracy_percent"]
            row["expert_2_correct"] = second_row["correct"]
            row["expert_2_total"] = second_row["total"]
        rows.append(row)
    return rows


def build_expert_accuracy_payload(request):
    train_size = parse_sample_size(request.GET.get("train-size"), DEFAULT_TRAIN_SIZE)
    test_size = parse_sample_size(request.GET.get("test-size"), DEFAULT_TEST_SIZE)
    test_examples = load_ag_news_dataset(train_size, test_size)["test"]

    expert_settings = parse_requested_experts(request)
    expert_results = []
    expert_metrics = []

    for index, settings in enumerate(expert_settings, start=1):
        expert_result = evaluate_expert_from_settings(test_examples, settings)
        expert_results.append(expert_result)
        expert_metrics.append(
            {
                "name": f"Expert {index}",
                "type": settings["type"],
                "accuracy_percent": round(expert_result["accuracy"] * 100, 2),
                "settings": settings,
            }
        )

    expert_class_rows = build_expert_class_rows(expert_results)
    accuracy_percent = (
        round(
            sum(item["accuracy_percent"] for item in expert_metrics)
            / len(expert_metrics),
            2,
        )
        if expert_metrics
        else 0
    )

    results = build_project3_results(request)

    return {
        "accuracy_percent": accuracy_percent,
        "experts": expert_metrics,
        "expert_count": len(expert_metrics),
        "expert_settings": expert_settings,
        "expert_summaries": build_expert_summaries(expert_settings, expert_results),
        "expert_class_rows": expert_class_rows,
        "policy_rows": results.get("policy_rows", []),
        "allocation_rows": results.get("allocation_rows", []),
        "allocation_by_policy": results.get("allocation_by_policy", {}),
        "model_plot_url": results.get("model_plot_url"),
        "cost_plot_url": results.get("cost_plot_url"),
        "deferred_plot_url": results.get("deferred_plot_url"),
        "benefit_plot_url": results.get("benefit_plot_url"),
        "active_learning": results.get("active_learning"),
        "active_learning_by_strategy": results.get("active_learning_by_strategy"),
        "strategy_rows": results.get("strategy_rows", []),
        "expert_one_query_budget": results.get("expert_one_query_budget"),
        "expert_two_query_budget": results.get("expert_two_query_budget"),
        "scatter_plot_url": results.get("scatter_plot_url"),
        "scatter_plots": results.get("scatter_plots"),
        "human_expert": results.get("human_expert"),
        "human_expert_strategy": results.get("human_expert_strategy"),
        "saved_configs": results.get("saved_configs", []),
    }


def index(request):
    if request.method == "POST":
        action = request.POST.get("action")
        if action == "clear-human-labels":
            request.session["project3_human_labels"] = {}
        elif action == "clear-configs":
            request.session["project3_saved_configs"] = []
        else:
            labels = request.session.get("project3_human_labels", {}).copy()
            for key, value in request.POST.items():
                if key.startswith("human_label_") and value in {
                    str(cid) for cid in CLASS_IDS
                }:
                    labels[key.removeprefix("human_label_")] = value
            request.session["project3_human_labels"] = labels
        request.session.modified = True
        if request.headers.get("X-Requested-With") == "XMLHttpRequest" or request.GET.get("format") == "json":
            try:
                results = build_project3_results(request)
                return JsonResponse({
                    "success": True,
                    "human_expert": results.get("human_expert"),
                    "human_expert_strategy": results.get("human_expert_strategy"),
                }, encoder=NumpyJSONEncoder)
            except Exception as e:
                return JsonResponse({"error_message": str(e)}, status=400)
        query_string = request.META.get("QUERY_STRING", "")
        target = request.path + (f"?{query_string}" if query_string else "")
        return redirect(target)

    if request.GET.get("format") == "json":
        try:
            return JsonResponse(build_expert_accuracy_payload(request), encoder=NumpyJSONEncoder)
        except ValueError as e:
            return JsonResponse({"error_message": str(e)}, status=400)

    try:
        results = build_project3_results(request)
    except ValueError as e:
        train_size = parse_sample_size(request.GET.get("train-size"), DEFAULT_TRAIN_SIZE)
        test_size = parse_sample_size(request.GET.get("test-size"), DEFAULT_TEST_SIZE)
        defer_rate = parse_float(request.GET.get("defer-rate"), DEFAULT_DEFER_RATE)
        query_budget = parse_sample_size(
            request.GET.get("expert-one-query-budget") or request.GET.get("query-budget"),
            DEFAULT_QUERY_BUDGET
        )
        base = get_base_experiment_context(train_size, test_size, query_budget)
        active_learning = active_learning_queries(
            base["train_examples"], base["baseline"], base["query_budget"], expert_configs=None
        )
        human_expert = build_human_expert_context(
            request, base["train_examples"], active_learning
        )
        scatter_plot_url = save_active_learning_scatter_plot(base["train_examples"], active_learning)

        results = {
            "error_message": str(e),
            "train_size": "all" if train_size is None else train_size,
            "test_size": "all" if test_size is None else test_size,
            "defer_rate": defer_rate,
            "defer_rate_percent": round(defer_rate * 100, 1),
            "train_rows": len(base["train_examples"]),
            "test_rows": len(base["test_examples"]),
            "full_train_rows": base["dataset"]["full_train_rows"],
            "full_test_rows": base["dataset"]["full_test_rows"],
            "dataset_source": base["dataset"]["source"],
            "class_names": CLASS_NAMES,
            "baseline_class_rows": class_metric_rows(base["baseline"]["per_class"]),
            "baseline_accuracy_percent": round(base["baseline"]["accuracy"] * 100, 2),
            "expert_count": 0,
            "human_expert": human_expert,
            "sample_examples": base["sample_examples"],
            "scatter_plot_url": scatter_plot_url,
            "saved_configs": request.session.get("project3_saved_configs", []) if hasattr(request, "session") else [],
        }

    results["content_menu_items"] = [
        {"label": "Baseline Classifier", "href": "#baseline-classifier"},
        {"label": "Learning to Defer", "href": "#learning-to-defer"},
        {"label": "Active Learning", "href": "#active-learning"},
        {"label": "Human Expert", "href": "#human-expert"},
        {"label": "Test Examples", "href": "#sample-test-examples"},
        {"label": "PDF Report", "href": "#pdf-report"}
    ]

    results["expert_options"] = [
        {"label": "Trivial (either 100% or 0% correct)", "value": "TRIVIAL"},
        {"label": "Realistic (some randomness is present)", "value": "REALISTIC"},
    ]
    #  {"label": "Trivial Expert (Always correct in sports, never correct in others)", "value": "ALWAYS_CORRECT_IN_ONE_FIELD"},
    #  {"label": "Realistic Expert (Good in sports, average in others)", "value": "REALISTIC"},]
    results["competence_levels"] = [
        {"label": "less competent", "value": "less-competent"},
        {"label": "more competent", "value": "more-competent"},
    ]
    return render(request, "project3/index.html", results)

def restore_session_query_params(request):
    if hasattr(request, "session") and "project3_last_query_params" in request.session:
        params = request.session["project3_last_query_params"]
        q_dict = request.GET.copy()
        for k, v in params.items():
            if k not in q_dict:
                q_dict[k] = v
        request.GET = q_dict

from xhtml2pdf import pisa  # Pure Python converter!
import io

REPORT_FILENAME = "project3_report.pdf"

def _prepare_active_learning_strategies(cfg_results):
    active_learning_strategies = []
    scatter_plots = cfg_results.get("scatter_plots", {})
    for strat_key, url_val in list(scatter_plots.items()):
        if url_val and not url_val.startswith(settings.MEDIA_ROOT):
            scatter_plots[strat_key] = os.path.join(
                settings.MEDIA_ROOT, url_val.removeprefix(settings.MEDIA_URL)
            )
            
    for strat_key, strat_name, strat_desc in [
        ("balanced_uncertainty", "Balanced Uncertainty", "Selects low-margin classifier examples while balancing across predicted classes."),
        ("uncertainty", "Uncertainty-only", "Selects examples where the classifier's predictions have the lowest margin, without balancing across classes."),
        ("random", "Random Querying", "Selects training examples uniformly at random to query the expert.")
    ]:
        al_data = cfg_results.get("active_learning_by_strategy", {}).get(strat_key, {})
        active_learning_strategies.append({
            "key": strat_key,
            "name": strat_name,
            "description": strat_desc,
            "estimated_competence_rows": al_data.get("estimated_competence_rows", []),
            "scatter_plot_url": scatter_plots.get(strat_key),
        })
    return active_learning_strategies


def report(request):
    restore_session_query_params(request)
    results = build_project3_results(request)

    # Convert plot URLs to absolute filesystem paths for xhtml2pdf/pisa
    results_copy = results.copy()
    for key in ["model_plot_url", "deferred_plot_url", "benefit_plot_url", "cost_plot_url", "scatter_plot_url"]:
        url_val = results_copy.get(key)
        if url_val:
            results_copy[key] = os.path.join(
                settings.MEDIA_ROOT, url_val.removeprefix(settings.MEDIA_URL)
            )

    results_copy["training_diagram_path"] = os.path.join(
        settings.BASE_DIR, "project3/static/project3/images/training_active_learning_diagram.png"
    )
    results_copy["testing_diagram_path"] = os.path.join(
        settings.BASE_DIR, "project3/static/project3/images/testing_active_learning_diagram.png"
    )

    include_configs = request.GET.getlist("include_config")
    saved_list = request.session.get("project3_saved_configs", []) if hasattr(request, "session") else []
    
    selected_configs = []
    
    # If no specific configurations are selected, render just the current configuration
    if not include_configs:
        current_cfg = {
            "label": "Current Configuration",
            "expert_count": results_copy["expert_count"],
            "experts": results_copy["experts"],
            "expert_class_rows": results_copy["expert_class_rows"],
            "policy_rows": results_copy["policy_rows"],
            "allocation_rows": results_copy["allocation_rows"],
            "model_plot_url": results_copy["model_plot_url"],
            "deferred_plot_url": results_copy["deferred_plot_url"],
            "benefit_plot_url": results_copy["benefit_plot_url"],
            "cost_plot_url": results_copy.get("cost_plot_url"),
            "expert_one_query_budget": results_copy.get("expert_one_query_budget"),
            "expert_two_query_budget": results_copy.get("expert_two_query_budget"),
            "expert_summaries": results_copy.get("expert_summaries"),
            "active_learning": results_copy.get("active_learning"),
            "scatter_plot_url": results_copy["scatter_plot_url"],
            "strategy_rows": results_copy["strategy_rows"],
            "active_learning_strategies": _prepare_active_learning_strategies(results_copy),
        }
        selected_configs.append(current_cfg)
    else:
        # Rebuild results for each selected config
        for cfg_id in include_configs:
            cfg_item = None
            for item in saved_list:
                if item["id"] == cfg_id:
                    cfg_item = item
                    break
            if not cfg_item:
                continue
                
            orig_get = request.GET
            mock_get = orig_get.copy()
            for k, v in cfg_item["params"].items():
                mock_get[k] = v
            request.GET = mock_get
            
            try:
                cfg_results = build_project3_results(request)
            finally:
                request.GET = orig_get
                
            for key in ["model_plot_url", "deferred_plot_url", "benefit_plot_url", "cost_plot_url", "scatter_plot_url"]:
                url_val = cfg_results.get(key)
                if url_val:
                    cfg_results[key] = os.path.join(
                        settings.MEDIA_ROOT, url_val.removeprefix(settings.MEDIA_URL)
                    )
            
            selected_configs.append({
                "label": cfg_item["label"],
                "expert_count": cfg_results["expert_count"],
                "experts": cfg_results["experts"],
                "expert_class_rows": cfg_results["expert_class_rows"],
                "policy_rows": cfg_results["policy_rows"],
                "allocation_rows": cfg_results["allocation_rows"],
                "model_plot_url": cfg_results["model_plot_url"],
                "deferred_plot_url": cfg_results["deferred_plot_url"],
                "benefit_plot_url": cfg_results["benefit_plot_url"],
                "cost_plot_url": cfg_results.get("cost_plot_url"),
                "expert_one_query_budget": cfg_results.get("expert_one_query_budget"),
                "expert_two_query_budget": cfg_results.get("expert_two_query_budget"),
                "expert_summaries": cfg_results.get("expert_summaries"),
                "active_learning": cfg_results.get("active_learning"),
                "scatter_plot_url": cfg_results["scatter_plot_url"],
                "strategy_rows": cfg_results["strategy_rows"],
                "active_learning_strategies": _prepare_active_learning_strategies(cfg_results),
            })
            
    results_copy["selected_configs"] = selected_configs
    results_copy["class_names_list"] = ["World", "Sports", "Business", "Sci/Tech", "Total"]

    # 1. Render Django HTML template to a string
    html_string = render_to_string("project3/report_pdf.html", results_copy)

    # 2. Write PDF to an in-memory byte buffer
    pdf_buffer = io.BytesIO()
    pisa_status = pisa.CreatePDF(src=html_string, dest=pdf_buffer)

    if pisa_status.err:
        return HttpResponse("Error generating PDF report", status=500)

    # 3. Return response as downloadable PDF
    response = HttpResponse(pdf_buffer.getvalue(), content_type="application/pdf")
    response["Content-Disposition"] = f'attachment; filename="{REPORT_FILENAME}"'
    return response
