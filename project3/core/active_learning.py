import hashlib
import random
import numpy as np
from sklearn.metrics import accuracy_score
from .utils import CLASS_IDS, CLASS_NAMES
from .deferral import prediction_margins, simulated_expert_predict, per_class_accuracy, class_metric_rows

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
            candidates = [idx for idx, pred in enumerate(predicted_labels) if pred == class_id]
            candidates = sorted(candidates, key=lambda idx: margins[idx])
            selected_indices.extend(candidates[:per_class_quota])

        if len(selected_indices) < query_budget:
            selected_set = set(selected_indices)
            remaining = [idx for idx in np.argsort(margins) if idx not in selected_set]
            selected_indices.extend(remaining[: query_budget - len(selected_indices)])
        return selected_indices[:query_budget]

    return [int(index) for index in np.argsort(margins)[:query_budget]]

def competence_from_queries(y_true, expert_predictions):
    values = {}
    for class_id in CLASS_IDS:
        class_indices = [idx for idx, lbl in enumerate(y_true) if lbl == class_id]
        if not class_indices:
            values[class_id] = 0.5
            continue
        correct = sum(1 for idx in class_indices if expert_predictions[idx] == y_true[idx])
        values[class_id] = correct / len(class_indices)
    return values

def active_learning_queries(train_examples, baseline, query_budget, strategy="balanced_uncertainty"):
    texts = [example["text"] for example in train_examples]
    decision_scores = baseline["pipeline"].decision_function(texts)
    margins = prediction_margins(decision_scores)
    predicted_labels = baseline["pipeline"].predict(texts)
    expert_predictions = [simulated_expert_predict(example) for example in train_examples]

    selected_indices = selected_query_indices(strategy, train_examples, predicted_labels, margins, query_budget)
    selected_true = [train_examples[idx]["label"] for idx in selected_indices]
    selected_expert = [expert_predictions[idx] for idx in selected_indices]
    selected_predicted = [int(predicted_labels[idx]) for idx in selected_indices]

    estimated_competence = per_class_accuracy(selected_true, selected_expert)
    competence_by_class = competence_from_queries(selected_true, selected_expert)
    agreement = accuracy_score(selected_predicted, selected_expert) if selected_indices else 0

    rows = []
    for idx in selected_indices[:8]:
        example = train_examples[idx]
        rows.append({
            "id": str(idx),
            "model_class": CLASS_NAMES[int(predicted_labels[idx])],
            "expert_class": CLASS_NAMES[expert_predictions[idx]],
            "true_class": CLASS_NAMES[example["label"]],
            "margin": round(float(margins[idx]), 3),
            "text": example["text"][:180] + ("..." if len(example["text"]) > 180 else ""),
        })

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

    for idx in selected_indices:
        example = train_examples[idx]
        key = human_label_key(example)
        selected_label = labels.get(key, "")
        is_answered = selected_label != ""
        is_correct = is_answered and int(selected_label) == example["label"]
        answered += 1 if is_answered else 0
        correct += 1 if is_correct else 0
        rows.append({
            "key": key,
            "text": example["text"][:360] + ("..." if len(example["text"]) > 360 else ""),
            "true_class": CLASS_NAMES[example["label"]],
            "selected_label": selected_label,
            "answered": is_answered,
            "correct": is_correct,
        })

    return {
        "rows": rows,
        "answered": answered,
        "total": len(rows),
        "correct": correct,
        "accuracy_percent": round((correct / answered) * 100, 2) if answered else 0,
        "class_options": [{"value": str(cid), "label": cname} for cid, cname in CLASS_NAMES.items()],
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
        rows.append({
            "strategy": strategy,
            "label": label,
            "agreement_percent": result["agreement_percent"],
            "average_competence_percent": round(float(np.mean(competence_values)) * 100, 2),
            "min_competence_percent": round(float(np.min(competence_values)) * 100, 2),
        })
    return rows
