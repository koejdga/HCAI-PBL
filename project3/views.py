import csv
import hashlib
import os
import random
from functools import lru_cache
from urllib.error import URLError
from urllib.request import urlretrieve

import numpy as np
import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from django.conf import settings
from django.http import HttpResponse
from django.shortcuts import redirect, render
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


CLASS_NAMES = {
    1: "World",
    2: "Sports",
    3: "Business",
    4: "Sci/Tech",
}
CLASS_IDS = list(CLASS_NAMES.keys())
TRAIN_URL = (
    "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/"
    "data/ag_news_csv/train.csv"
)
TEST_URL = (
    "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/"
    "data/ag_news_csv/test.csv"
)
DEFAULT_TRAIN_SIZE = 8000
DEFAULT_TEST_SIZE = 2000
DEFAULT_DEFER_RATE = 0.30
DEFAULT_QUERY_BUDGET = 120
DEFAULT_EXPERT_ADVANTAGE = 0.08
REPORT_FILENAME = "project3_report.pdf"


FALLBACK_ROWS = [
    (1, "UN leaders discuss humanitarian aid", "Diplomats met to coordinate relief and peace talks."),
    (1, "Election observers monitor vote", "International observers watched polling stations after unrest."),
    (1, "Trade ministers reach border agreement", "Neighboring governments announced a new customs accord."),
    (2, "Local club wins final", "The team scored twice late to win the championship."),
    (2, "Tennis star reaches semifinal", "A strong serve helped the player advance in straight sets."),
    (2, "Coach praises defensive effort", "The squad held its rival scoreless during the match."),
    (3, "Stocks rise after earnings reports", "Technology shares lifted the market after quarterly results."),
    (3, "Oil prices affect airline profits", "Fuel costs pressured carriers despite strong travel demand."),
    (3, "Central bank keeps rates steady", "Investors watched the decision for signs of inflation risk."),
    (4, "New chip improves battery life", "Researchers introduced a processor for efficient mobile devices."),
    (4, "Space telescope sends images", "Scientists released detailed observations of distant galaxies."),
    (4, "Software update fixes security issue", "The patch closes a vulnerability in network services."),
]


def parse_sample_size(value, default):
    if value == "all":
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def parse_float(value, default, minimum=0.0, maximum=1.0):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return min(max(parsed, minimum), maximum)


def cache_dir():
    path = os.path.join(settings.MEDIA_ROOT, "project3", "ag_news")
    os.makedirs(path, exist_ok=True)
    return path


def artifact_dir():
    path = os.path.join(settings.MEDIA_ROOT, "project3")
    os.makedirs(path, exist_ok=True)
    return path


def cached_csv_path(split):
    return os.path.join(cache_dir(), f"{split}.csv")


def ensure_ag_news_file(split):
    path = cached_csv_path(split)
    if os.path.exists(path):
        return path

    url = TRAIN_URL if split == "train" else TEST_URL
    urlretrieve(url, path)
    return path


def read_ag_news_csv(path):
    examples = []
    with open(path, newline="", encoding="utf-8") as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            if len(row) < 3:
                continue
            label = int(row[0])
            text = f"{row[1]} {row[2]}".strip()
            examples.append({"label": label, "text": text})
    return examples


def balanced_sample(examples, max_rows):
    if max_rows is None or len(examples) <= max_rows:
        return list(examples)

    by_class = {class_id: [] for class_id in CLASS_IDS}
    for example in examples:
        by_class[example["label"]].append(example)

    per_class = max(1, max_rows // len(CLASS_IDS))
    sampled = []
    for class_id in CLASS_IDS:
        sampled.extend(by_class[class_id][:per_class])

    return sampled[:max_rows]


def build_fallback_dataset():
    examples = [
        {"label": label, "text": f"{title}. {description}"}
        for label, title, description in FALLBACK_ROWS
    ]
    train_examples = examples * 8
    test_examples = examples * 3
    return train_examples, test_examples


@lru_cache(maxsize=8)
def load_ag_news_dataset(train_size, test_size):
    try:
        train_examples = read_ag_news_csv(ensure_ag_news_file("train"))
        test_examples = read_ag_news_csv(ensure_ag_news_file("test"))
        source = "AG News CSV cache"
    except (OSError, URLError, ValueError):
        train_examples, test_examples = build_fallback_dataset()
        source = "built-in fallback sample"

    return {
        "train": balanced_sample(train_examples, train_size),
        "test": balanced_sample(test_examples, test_size),
        "full_train_rows": len(train_examples),
        "full_test_rows": len(test_examples),
        "source": source,
    }


def train_baseline_classifier(train_examples, test_examples):
    X_train = [example["text"] for example in train_examples]
    y_train = [example["label"] for example in train_examples]
    X_test = [example["text"] for example in test_examples]
    y_test = [example["label"] for example in test_examples]

    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    max_features=25000,
                    min_df=2,
                ),
            ),
            ("model", LinearSVC(random_state=42)),
        ]
    )
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    decision_scores = pipeline.decision_function(X_test)

    return {
        "pipeline": pipeline,
        "accuracy": accuracy_score(y_test, predictions),
        "predictions": predictions,
        "decision_scores": decision_scores,
        "per_class": per_class_accuracy(y_test, predictions),
        "confusion": build_confusion_rows(y_test, predictions),
    }


def deterministic_score(text, salt):
    digest = hashlib.sha256(f"{salt}:{text}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def simulated_expert_predict(example):
    label = example["label"]
    text = example["text"]
    lower_text = text.lower()

    strengths = {
        1: 0.70,
        2: 0.90,
        3: 0.58,
        4: 0.78,
    }

    sports_words = ["team", "game", "match", "coach", "season", "win", "player"]
    science_words = ["software", "technology", "space", "chip", "research", "security"]
    business_words = ["stocks", "market", "bank", "profit", "prices", "earnings"]

    if label == 2 and any(word in lower_text for word in sports_words):
        competence = 0.96
    elif label == 4 and any(word in lower_text for word in science_words):
        competence = 0.88
    elif label == 3 and any(word in lower_text for word in business_words):
        competence = 0.68
    else:
        competence = strengths[label]

    if deterministic_score(text, "expert-correctness") < competence:
        return label

    alternatives = [class_id for class_id in CLASS_IDS if class_id != label]
    wrong_index = int(deterministic_score(text, "expert-error") * len(alternatives))
    return alternatives[min(wrong_index, len(alternatives) - 1)]


def evaluate_simulated_expert(test_examples):
    y_true = [example["label"] for example in test_examples]
    predictions = [simulated_expert_predict(example) for example in test_examples]

    return {
        "accuracy": accuracy_score(y_true, predictions),
        "predictions": predictions,
        "per_class": per_class_accuracy(y_true, predictions),
        "confusion": build_confusion_rows(y_true, predictions),
    }


def per_class_accuracy(y_true, predictions):
    metrics = {}
    for class_id, class_name in CLASS_NAMES.items():
        total = sum(1 for label in y_true if label == class_id)
        correct = sum(
            1
            for label, prediction in zip(y_true, predictions)
            if label == class_id and prediction == class_id
        )
        metrics[class_name] = {
            "correct": correct,
            "total": total,
            "accuracy": correct / total if total else 0,
            "accuracy_percent": round((correct / total) * 100, 2) if total else 0,
        }
    return metrics


def build_confusion_rows(y_true, predictions):
    matrix = confusion_matrix(y_true, predictions, labels=CLASS_IDS)
    rows = []
    for index, class_id in enumerate(CLASS_IDS):
        rows.append(
            {
                "label": CLASS_NAMES[class_id],
                "values": [int(value) for value in matrix[index]],
            }
        )
    return rows


def class_metric_rows(per_class):
    return [
        {
            "class_name": class_name,
            "accuracy_percent": values["accuracy_percent"],
            "correct": values["correct"],
            "total": values["total"],
        }
        for class_name, values in per_class.items()
    ]


def prediction_margins(decision_scores):
    sorted_scores = np.sort(decision_scores, axis=1)
    return sorted_scores[:, -1] - sorted_scores[:, -2]


def evaluate_learning_to_defer(test_examples, baseline, expert, defer_rate):
    y_true = np.array([example["label"] for example in test_examples])
    baseline_predictions = np.array(baseline["predictions"])
    expert_predictions = np.array(expert["predictions"])
    margins = prediction_margins(baseline["decision_scores"])

    defer_count = int(round(len(test_examples) * defer_rate))
    defer_count = min(max(defer_count, 0), len(test_examples))
    deferred_mask = np.zeros(len(test_examples), dtype=bool)
    if defer_count:
        deferred_indices = np.argsort(margins)[:defer_count]
        deferred_mask[deferred_indices] = True

    team_predictions = np.where(deferred_mask, expert_predictions, baseline_predictions)
    deferred_total = int(deferred_mask.sum())
    non_deferred_total = len(test_examples) - deferred_total

    useful_defer = int(
        np.sum(
            deferred_mask
            & (baseline_predictions != y_true)
            & (expert_predictions == y_true)
        )
    )
    harmful_defer = int(
        np.sum(
            deferred_mask
            & (baseline_predictions == y_true)
            & (expert_predictions != y_true)
        )
    )
    both_correct_defer = int(
        np.sum(
            deferred_mask
            & (baseline_predictions == y_true)
            & (expert_predictions == y_true)
        )
    )

    return {
        "policy_name": "Confidence threshold",
        "accuracy": accuracy_score(y_true, team_predictions),
        "deferral_rate": deferred_total / len(test_examples) if test_examples else 0,
        "deferred_total": deferred_total,
        "non_deferred_total": non_deferred_total,
        "useful_defer": useful_defer,
        "harmful_defer": harmful_defer,
        "both_correct_defer": both_correct_defer,
        "confusion": build_confusion_rows(y_true, team_predictions),
    }


def selected_query_indices(strategy, train_examples, predicted_labels, margins, query_budget):
    if query_budget <= 0:
        return []

    if strategy == "random":
        rng = random.Random(42)
        indices = list(range(len(train_examples)))
        rng.shuffle(indices)
        return indices[:query_budget]

    if strategy == "balanced_uncertainty":
        selected_indices = []
        per_class_quota = max(1, query_budget // len(CLASS_IDS))
        for class_id in CLASS_IDS:
            candidates = [
                index
                for index, predicted_label in enumerate(predicted_labels)
                if predicted_label == class_id
            ]
            candidates = sorted(candidates, key=lambda index: margins[index])
            selected_indices.extend(candidates[:per_class_quota])

        if len(selected_indices) < query_budget:
            selected_set = set(selected_indices)
            remaining = [
                index for index in np.argsort(margins) if index not in selected_set
            ]
            selected_indices.extend(remaining[: query_budget - len(selected_indices)])
        return selected_indices[:query_budget]

    return [int(index) for index in np.argsort(margins)[:query_budget]]


def competence_from_queries(y_true, expert_predictions):
    values = {}
    for class_id in CLASS_IDS:
        class_indices = [
            index for index, label in enumerate(y_true) if label == class_id
        ]
        if not class_indices:
            values[class_id] = 0.5
            continue
        correct = sum(
            1
            for index in class_indices
            if expert_predictions[index] == y_true[index]
        )
        values[class_id] = correct / len(class_indices)
    return values


def active_learning_queries(train_examples, baseline, query_budget, strategy="balanced_uncertainty"):
    texts = [example["text"] for example in train_examples]
    decision_scores = baseline["pipeline"].decision_function(texts)
    margins = prediction_margins(decision_scores)
    predicted_labels = baseline["pipeline"].predict(texts)
    expert_predictions = [simulated_expert_predict(example) for example in train_examples]

    selected_indices = selected_query_indices(
        strategy,
        train_examples,
        predicted_labels,
        margins,
        query_budget,
    )
    selected_true = [train_examples[index]["label"] for index in selected_indices]
    selected_expert = [expert_predictions[index] for index in selected_indices]
    selected_predicted = [int(predicted_labels[index]) for index in selected_indices]

    estimated_competence = per_class_accuracy(selected_true, selected_expert)
    competence_by_class = competence_from_queries(selected_true, selected_expert)
    agreement = accuracy_score(selected_predicted, selected_expert) if selected_indices else 0

    rows = []
    for index in selected_indices[:8]:
        example = train_examples[index]
        rows.append(
            {
                "id": str(index),
                "model_class": CLASS_NAMES[int(predicted_labels[index])],
                "expert_class": CLASS_NAMES[expert_predictions[index]],
                "true_class": CLASS_NAMES[example["label"]],
                "margin": round(float(margins[index]), 3),
                "text": example["text"][:180]
                + ("..." if len(example["text"]) > 180 else ""),
            }
        )

    return {
        "strategy": strategy,
        "query_budget": len(selected_indices),
        "agreement_percent": round(agreement * 100, 2),
        "competence_by_class": competence_by_class,
        "estimated_competence_rows": class_metric_rows(estimated_competence),
        "selected_indices": selected_indices,
        "queried_rows": rows,
    }


def human_label_key(example):
    return hashlib.sha256(example["text"].encode("utf-8")).hexdigest()[:16]


def build_human_expert_context(request, train_examples, active_learning, limit=6):
    labels = request.session.get("project3_human_labels", {})
    selected_indices = active_learning["selected_indices"][:limit]
    rows = []
    correct = 0
    answered = 0

    for index in selected_indices:
        example = train_examples[index]
        key = human_label_key(example)
        selected_label = labels.get(key, "")
        is_answered = selected_label != ""
        is_correct = is_answered and int(selected_label) == example["label"]
        answered += 1 if is_answered else 0
        correct += 1 if is_correct else 0
        rows.append(
            {
                "key": key,
                "text": example["text"][:360]
                + ("..." if len(example["text"]) > 360 else ""),
                "true_class": CLASS_NAMES[example["label"]],
                "selected_label": selected_label,
                "answered": is_answered,
                "correct": is_correct,
            }
        )

    return {
        "rows": rows,
        "answered": answered,
        "total": len(rows),
        "correct": correct,
        "accuracy_percent": round((correct / answered) * 100, 2) if answered else 0,
        "class_options": [
            {"value": str(class_id), "label": class_name}
            for class_id, class_name in CLASS_NAMES.items()
        ],
    }


def compare_active_learning_strategies(train_examples, baseline, query_budget):
    strategies = [
        ("balanced_uncertainty", "Balanced uncertainty"),
        ("uncertainty", "Uncertainty only"),
        ("random", "Random sample"),
    ]
    rows = []
    for strategy, label in strategies:
        result = active_learning_queries(train_examples, baseline, query_budget, strategy)
        competence_values = list(result["competence_by_class"].values())
        rows.append(
            {
                "strategy": strategy,
                "label": label,
                "agreement_percent": result["agreement_percent"],
                "average_competence_percent": round(float(np.mean(competence_values)) * 100, 2),
                "min_competence_percent": round(float(np.min(competence_values)) * 100, 2),
            }
        )
    return rows


def evaluate_competence_aware_defer(test_examples, baseline, expert, competence_by_class):
    y_true = np.array([example["label"] for example in test_examples])
    baseline_predictions = np.array(baseline["predictions"])
    expert_predictions = np.array(expert["predictions"])
    margins = prediction_margins(baseline["decision_scores"])
    # Map the uncalibrated SVM margin to a conservative correctness estimate.
    # Without this guard the policy can defer nearly everything, which is both
    # poor collaboration and poor workload management.
    median_margin = max(float(np.median(margins)), 0.001)
    model_confidence = 1 / (1 + np.exp(-(margins / median_margin)))
    expert_competence = np.array(
        [competence_by_class.get(int(label), 0.5) for label in baseline_predictions]
    )
    expert_advantage = expert_competence - model_confidence
    deferred_mask = expert_advantage > DEFAULT_EXPERT_ADVANTAGE
    team_predictions = np.where(deferred_mask, expert_predictions, baseline_predictions)

    useful_defer = int(
        np.sum(
            deferred_mask
            & (baseline_predictions != y_true)
            & (expert_predictions == y_true)
        )
    )
    harmful_defer = int(
        np.sum(
            deferred_mask
            & (baseline_predictions == y_true)
            & (expert_predictions != y_true)
        )
    )

    return {
        "policy_name": "Competence-aware",
        "accuracy": accuracy_score(y_true, team_predictions),
        "deferred_total": int(deferred_mask.sum()),
        "non_deferred_total": int(len(test_examples) - deferred_mask.sum()),
        "useful_defer": useful_defer,
        "harmful_defer": harmful_defer,
        "average_expert_advantage": float(np.mean(expert_advantage)),
        "confusion": build_confusion_rows(y_true, team_predictions),
    }


def save_bar_plot(filename, title, labels, values, ylabel):
    path = os.path.join(artifact_dir(), filename)
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(labels, values, color="#00a6b2")
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.set_ylim(0, max(100, max(values) + 5 if values else 100))
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return settings.MEDIA_URL + f"project3/{filename}"


def build_project3_results(request):
    train_size = parse_sample_size(
        request.GET.get("train-size"),
        DEFAULT_TRAIN_SIZE,
    )
    test_size = parse_sample_size(
        request.GET.get("test-size"),
        DEFAULT_TEST_SIZE,
    )
    defer_rate = parse_float(request.GET.get("defer-rate"), DEFAULT_DEFER_RATE)
    query_budget = parse_sample_size(
        request.GET.get("query-budget"),
        DEFAULT_QUERY_BUDGET,
    )

    dataset = load_ag_news_dataset(train_size, test_size)
    train_examples = dataset["train"]
    test_examples = dataset["test"]
    if query_budget is None:
        query_budget = min(len(train_examples), DEFAULT_QUERY_BUDGET)
    query_budget = min(query_budget, len(train_examples))

    baseline = train_baseline_classifier(train_examples, test_examples)
    expert = evaluate_simulated_expert(test_examples)
    active_learning = active_learning_queries(train_examples, baseline, query_budget)
    human_expert = build_human_expert_context(
        request,
        train_examples,
        active_learning,
    )
    strategy_rows = compare_active_learning_strategies(
        train_examples,
        baseline,
        query_budget,
    )
    confidence_team = evaluate_learning_to_defer(test_examples, baseline, expert, defer_rate)
    competence_team = evaluate_competence_aware_defer(
        test_examples,
        baseline,
        expert,
        active_learning["competence_by_class"],
    )

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
            "deferral_rate_percent": round(confidence_team["deferred_total"] / len(test_examples) * 100, 2),
            "useful_defer": confidence_team["useful_defer"],
            "harmful_defer": confidence_team["harmful_defer"],
            "net_defer": confidence_team["useful_defer"] - confidence_team["harmful_defer"],
        },
        {
            "name": competence_team["policy_name"],
            "accuracy_percent": round(competence_team["accuracy"] * 100, 2),
            "deferred_total": competence_team["deferred_total"],
            "deferral_rate_percent": round(competence_team["deferred_total"] / len(test_examples) * 100, 2),
            "useful_defer": competence_team["useful_defer"],
            "harmful_defer": competence_team["harmful_defer"],
            "net_defer": competence_team["useful_defer"] - competence_team["harmful_defer"],
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
        "baseline_accuracy_percent": round(baseline["accuracy"] * 100, 2),
        "expert_accuracy_percent": round(expert["accuracy"] * 100, 2),
        "team_policy_name": best_team_row["name"],
        "team_accuracy_percent": round(selected_team["accuracy"] * 100, 2),
        "team_deferred_total": selected_team["deferred_total"],
        "team_non_deferred_total": selected_team["non_deferred_total"],
        "team_useful_defer": selected_team["useful_defer"],
        "team_harmful_defer": selected_team["harmful_defer"],
        "active_learning": active_learning,
        "human_expert": human_expert,
        "strategy_rows": strategy_rows,
        "policy_rows": policy_rows,
        "model_plot_url": model_plot_url,
        "strategy_plot_url": strategy_plot_url,
        "report_url": "report/",
        "baseline_class_rows": class_metric_rows(baseline["per_class"]),
        "expert_class_rows": class_metric_rows(expert["per_class"]),
        "baseline_confusion_rows": baseline["confusion"],
        "expert_confusion_rows": expert["confusion"],
        "sample_examples": sample_examples(test_examples),
    }


def sample_examples(examples, count=5):
    rng = random.Random(42)
    selected = rng.sample(examples, min(count, len(examples)))
    return [
        {
            "label": CLASS_NAMES[example["label"]],
            "text": example["text"][:260] + ("..." if len(example["text"]) > 260 else ""),
        }
        for example in selected
    ]


def index(request):
    if request.method == "POST":
        action = request.POST.get("action")
        if action == "clear-human-labels":
            request.session["project3_human_labels"] = {}
        else:
            labels = request.session.get("project3_human_labels", {}).copy()
            for key, value in request.POST.items():
                if key.startswith("human_label_") and value in {str(class_id) for class_id in CLASS_IDS}:
                    labels[key.removeprefix("human_label_")] = value
            request.session["project3_human_labels"] = labels
        request.session.modified = True
        query_string = request.META.get("QUERY_STRING", "")
        target = request.path + (f"?{query_string}" if query_string else "")
        return redirect(target)

    return render(request, "project3/index.html", build_project3_results(request))


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
                settings.MEDIA_ROOT,
                image_url.removeprefix(settings.MEDIA_URL),
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
