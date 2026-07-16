import copy
import os
import random
from collections import OrderedDict
from django.conf import settings
from django.http import HttpResponse
from django.shortcuts import redirect, render
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from .core.true_l2d import TrueL2DClassifier
from .core.true_version2 import TrueL2DClassifier_2
from django.http import JsonResponse

from .core.utils import (
    CLASS_NAMES,
    CLASS_IDS,
    DEFAULT_TRAIN_SIZE,
    DEFAULT_TEST_SIZE,
    load_ag_news_dataset,
    parse_sample_size,
    parse_float,
    save_bar_plot,
    artifact_dir,
)
from .core.deferral import (
    train_baseline_classifier,
    evaluate_simulated_expert,
    evaluate_trivial_expert,
    evaluate_learning_to_defer,
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


def _cache_key(train_size, test_size, defer_rate, query_budget):
    return (train_size, test_size, round(float(defer_rate), 6), query_budget)


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


def sample_examples(examples, count=5):
    rng = random.Random(42)
    selected = rng.sample(examples, min(count, len(examples)))
    return [
        {
            "label": CLASS_NAMES[example["label"]],
            "text": example["text"][:260]
            + ("..." if len(example["text"]) > 260 else ""),
        }
        for example in selected
    ]


def _compute_project3_results(train_examples, test_examples, defer_rate, query_budget):
    baseline = train_baseline_classifier(train_examples, test_examples)
    expert = evaluate_simulated_expert(test_examples)
    active_learning = active_learning_queries(train_examples, baseline, query_budget)
    strategy_rows = compare_active_learning_strategies(
        train_examples, baseline, query_budget
    )
    confidence_team = evaluate_learning_to_defer(
        test_examples, baseline, expert, defer_rate
    )
    competence_team = evaluate_competence_aware_defer(
        test_examples, baseline, expert, active_learning["competence_by_class"]
    )

    true_l2d_classifier = TrueL2DClassifier(max_features=5000, epochs=10)
    true_l2d_classifier.fit(train_examples)
    true_l2d_results = true_l2d_classifier.predict_and_evaluate(test_examples)

    true_l2d_classifier_2 = TrueL2DClassifier_2(max_features=5000, epochs=10, query_cost=0.0)
    true_l2d_classifier_2.fit(train_examples)
    true_l2d_results_2 = true_l2d_classifier_2.predict_and_evaluate(test_examples)

    true_l2d_classifier_3 = TrueL2DClassifier_2(max_features=5000, epochs=10)
    true_l2d_classifier_3.fit(train_examples)
    true_l2d_results_3 = true_l2d_classifier_3.predict_and_evaluate(test_examples)

    policy_rows = [
        {
            "name": "Classifier only",
            "accuracy_percent": round(baseline["accuracy"] * 100, 2),
            "deferred_total": 0,
            "deferral_rate_percent": 0,
            "useful_defer": 0,
            "harmful_defer": 0,
            "net_defer": 0,
        },
        {
            "name": confidence_team["policy_name"],
            "accuracy_percent": round(confidence_team["accuracy"] * 100, 2),
            "deferred_total": confidence_team["deferred_total"],
            "deferral_rate_percent": round(
                confidence_team["deferred_total"] / len(test_examples) * 100, 2
            ),
            "useful_defer": confidence_team["useful_defer"],
            "harmful_defer": confidence_team["harmful_defer"],
            "net_defer": confidence_team["useful_defer"]
            - confidence_team["harmful_defer"],
        },
        {
            "name": competence_team["policy_name"],
            "accuracy_percent": round(competence_team["accuracy"] * 100, 2),
            "deferred_total": competence_team["deferred_total"],
            "deferral_rate_percent": round(
                competence_team["deferred_total"] / len(test_examples) * 100, 2
            ),
            "useful_defer": competence_team["useful_defer"],
            "harmful_defer": competence_team["harmful_defer"],
            "net_defer": competence_team["useful_defer"]
            - competence_team["harmful_defer"],
        },
        {
            "name": "Expert only",
            "accuracy_percent": round(expert["accuracy"] * 100, 2),
            "deferred_total": len(test_examples),
            "deferral_rate_percent": 100,
            "useful_defer": 0,
            "harmful_defer": 0,
            "net_defer": 0,
        },
        {
            "name": true_l2d_results["policy_name"],
            "accuracy_percent": round(true_l2d_results["accuracy"] * 100, 2),
            "deferred_total": true_l2d_results["deferred_total"],
            "deferral_rate_percent": round(
                true_l2d_results["deferred_total"] / len(test_examples) * 100, 2
            ),
            "useful_defer": true_l2d_results["useful_defer"],
            "harmful_defer": true_l2d_results["harmful_defer"],
            "net_defer": true_l2d_results["useful_defer"]
            - true_l2d_results["harmful_defer"],
        },
        {
            "name": true_l2d_results_2["policy_name"] + "_v2",
            "accuracy_percent": round(true_l2d_results_2["accuracy"] * 100, 2),
            "deferred_total": true_l2d_results_2["deferred_total"],
            "deferral_rate_percent": round(
                true_l2d_results_2["deferred_total"] / len(test_examples) * 100, 2
            ),
            "useful_defer": true_l2d_results_2["useful_defer"],
            "harmful_defer": true_l2d_results_2["harmful_defer"],
            "net_defer": true_l2d_results_2["useful_defer"]
            - true_l2d_results_2["harmful_defer"],
        },
        {
            "name": true_l2d_results_3["policy_name"] + "_v2",
            "accuracy_percent": round(true_l2d_results_3["accuracy"] * 100, 2),
            "deferred_total": true_l2d_results_3["deferred_total"],
            "deferral_rate_percent": round(
                true_l2d_results_3["deferred_total"] / len(test_examples) * 100, 2
            ),
            "useful_defer": true_l2d_results_3["useful_defer"],
            "harmful_defer": true_l2d_results_3["harmful_defer"],
            "net_defer": true_l2d_results_3["useful_defer"]
            - true_l2d_results_3["harmful_defer"],
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

    model_plot_url = save_bar_plot(
        "model_comparison.png",
        "Model and Team Accuracy",
        [row["name"] for row in policy_rows],
        [row["accuracy_percent"] for row in policy_rows],
        "Accuracy (%)",
    )
    strategy_plot_url = save_bar_plot(
        "active_learning_comparison.png",
        "Active Learning Strategy Comparison",
        [row["label"] for row in strategy_rows],
        [row["average_competence_percent"] for row in strategy_rows],
        "Estimated expert competence (%)",
    )

    return {
        "baseline_accuracy_percent": round(baseline["accuracy"] * 100, 2),
        "expert_accuracy_percent": round(expert["accuracy"] * 100, 2),
        "team_policy_name": best_team_row["name"],
        "team_accuracy_percent": round(selected_team["accuracy"] * 100, 2),
        "team_deferred_total": selected_team["deferred_total"],
        "team_non_deferred_total": selected_team["non_deferred_total"],
        "team_useful_defer": selected_team["useful_defer"],
        "team_harmful_defer": selected_team["harmful_defer"],
        "active_learning": active_learning,
        "strategy_rows": strategy_rows,
        "policy_rows": policy_rows,
        "model_plot_url": model_plot_url,
        "strategy_plot_url": strategy_plot_url,
        "baseline_class_rows": class_metric_rows(baseline["per_class"]),
        "expert_class_rows": class_metric_rows(expert["per_class"]),
        "baseline_confusion_rows": baseline["confusion"],
        "expert_confusion_rows": expert["confusion"],
        "sample_examples": sample_examples(test_examples),
    }


def build_project3_results(request):
    train_size = parse_sample_size(request.GET.get("train-size"), DEFAULT_TRAIN_SIZE)
    test_size = parse_sample_size(request.GET.get("test-size"), DEFAULT_TEST_SIZE)
    defer_rate = parse_float(request.GET.get("defer-rate"), DEFAULT_DEFER_RATE)
    query_budget = parse_sample_size(
        request.GET.get("query-budget"), DEFAULT_QUERY_BUDGET
    )

    dataset = load_ag_news_dataset(train_size, test_size)
    train_examples = dataset["train"]
    test_examples = dataset["test"]
    if query_budget is None:
        query_budget = min(len(train_examples), DEFAULT_QUERY_BUDGET)
    query_budget = min(query_budget, len(train_examples))

    cache_key = _cache_key(train_size, test_size, defer_rate, query_budget)
    cached_payload = _get_cached_project3_results(cache_key)
    if cached_payload is None:
        cached_payload = _compute_project3_results(
            train_examples, test_examples, defer_rate, query_budget
        )
        _store_cached_project3_results(cache_key, cached_payload)

    human_expert = build_human_expert_context(request, train_examples, cached_payload["active_learning"])

    return {
        "train_size": "all" if train_size is None else train_size,
        "test_size": "all" if test_size is None else test_size,
        "defer_rate": defer_rate,
        "defer_rate_percent": round(defer_rate * 100, 1),
        "query_budget": query_budget,
        "train_rows": len(train_examples),
        "test_rows": len(test_examples),
        "full_train_rows": dataset["full_train_rows"],
        "full_test_rows": dataset["full_test_rows"],
        "dataset_source": dataset["source"],
        "class_names": list(CLASS_NAMES.values()),
        **copy.deepcopy(cached_payload),
        "human_expert": human_expert,
        "report_url": "report/",
    }


def index(request):    
    if request.method == "POST":
        action = request.POST.get("action")
        if action == "clear-human-labels":
            request.session["project3_human_labels"] = {}
        else:
            labels = request.session.get("project3_human_labels", {}).copy()
            for key, value in request.POST.items():
                if key.startswith("human_label_") and value in {
                    str(cid) for cid in CLASS_IDS
                }:
                    labels[key.removeprefix("human_label_")] = value
            request.session["project3_human_labels"] = labels
        request.session.modified = True
        query_string = request.META.get("QUERY_STRING", "")
        target = request.path + (f"?{query_string}" if query_string else "")
        return redirect(target)
    
    results = build_project3_results(request)
    results["content_menu_items"] = [{"label": "Baseline Classifier", "href": "#baseline-classifier"},
                                     {"label": "Simulated Expert", "href": "#simulated-expert"},
                                     {"label": "Policy Comparison", "href": "#policy-comparison"},
                                     {"label": "Expert Queries", "href": "#expert-queries"}]

    results["expert_options"] = [{"label": "Trivial Expert (Always correct)", "value": "AWLAYS_CORRECT"},
                                 {"label": "Trivial Expert (Always incorrect)", "value": "ALWAYS_INCORRECT"},
                                 {"label": "Trivial Expert (Always correct in sports, never correct in others)", "value": "ALWAYS_CORRECT_IN_ONE_FIELD"},
                                 {"label": "Realistic Expert (Good in sports, average in others)", "value": "REALISTIC"},]
    return render(request, "project3/index.html", results)


def report(request):
    results = build_project3_results(request)
    report_path = os.path.join(artifact_dir(), REPORT_FILENAME)

    with PdfPages(report_path) as pdf:
        figure = plt.figure(figsize=(8.27, 11.69))
        figure.suptitle("Project 3: Active Learning for Learning-to-Defer", fontsize=16)
        lines = [
            f"Dataset source: {results['dataset_source']}",
            f"Training rows used: {results['train_rows']}",
            f"Test rows used: {results['test_rows']}",
            "",
            "Main results:",
            f"Baseline classifier accuracy: {results['baseline_accuracy_percent']}%",
            f"Simulated expert accuracy: {results['expert_accuracy_percent']}%",
            f"Competence-aware team accuracy: {results['team_accuracy_percent']}%",
            f"Expert queries used: {results['query_budget']}",
            "",
            "Design choices:",
            "- Baseline: TF-IDF text representation with a linear SVM.",
            "- Simulated expert: topic-specialist expert with uneven competence.",
            "- Deferral: compare classifier-only, confidence threshold, competence-aware, and expert-only policies.",
            "- Active learning: compare balanced uncertainty, uncertainty-only, and random querying.",
        ]
        figure.text(0.08, 0.9, "\n".join(lines), va="top", fontsize=11)
        figure.text(
            0.08,
            0.18,
            "The interface stores the same experiment metrics shown in this report, "
            "including class-level expert competence and queried examples.",
            fontsize=10,
        )
        pdf.savefig(figure)
        plt.close(figure)

        for image_url in [results["model_plot_url"], results["strategy_plot_url"]]:
            image_path = os.path.join(
                settings.MEDIA_ROOT, image_url.removeprefix(settings.MEDIA_URL)
            )
            image = plt.imread(image_path)
            figure, axis = plt.subplots(figsize=(11, 6))
            axis.imshow(image)
            axis.axis("off")
            pdf.savefig(figure)
            plt.close(figure)

    with open(report_path, "rb") as report_file:
        response = HttpResponse(report_file.read(), content_type="application/pdf")
    response["Content-Disposition"] = f'attachment; filename="{REPORT_FILENAME}"'
    return response
