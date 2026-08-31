import hashlib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from .utils import CLASS_IDS, CLASS_NAMES

DEFAULT_DEFER_RATE = 0.30
DEFAULT_QUERY_BUDGET = 120
DEFAULT_EXPERT_ADVANTAGE = 0.08

# Realistic per-class difficulty tweaks (World/Business are often harder for lay readers).
CLASS_DIFFICULTY = {1: -0.04, 2: 0.05, 3: -0.03, 4: 0.02}

DOMAIN_KEYWORDS = {
    1: ["government", "minister", "election", "diplomat", "war", "peace", "un ", "nato"],
    2: ["team", "game", "match", "coach", "season", "win", "player", "league", "score"],
    3: ["stocks", "market", "bank", "profit", "prices", "earnings", "company", "trade"],
    4: ["software", "technology", "space", "chip", "research", "security", "internet", "computer"],
}

COMPETENCE_PROFILES = {
    "more-competent": {"in_field": 0.86, "out_field": 0.54, "keyword_boost": 0.08},
    "less-competent": {"in_field": 0.71, "out_field": 0.41, "keyword_boost": 0.06},
}


def deterministic_score(text, salt):
    digest = hashlib.sha256(f"{salt}:{text}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF

def train_baseline_classifier(train_examples, test_examples):
    X_train = [example["text"] for example in train_examples]
    y_train = [example["label"] for example in train_examples]
    X_test = [example["text"] for example in test_examples]
    y_test = [example["label"] for example in test_examples]

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2), max_features=25000, min_df=2)),
        ("model", LinearSVC(random_state=42)),
    ])
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

def _normalize_competence_level(competence_level):
    if competence_level in COMPETENCE_PROFILES:
        return competence_level
    return "more-competent"


def _normalize_expert_fields(expert_fields):
    if expert_fields is None:
        return list(CLASS_IDS)
    normalized = [field_id for field_id in expert_fields if field_id in CLASS_IDS]
    return normalized or list(CLASS_IDS)


def _has_domain_cues(text, label):
    lower_text = text.lower()
    return any(keyword in lower_text for keyword in DOMAIN_KEYWORDS.get(label, []))


def _correctness_probability(label, expert_fields, competence_level, text):
    fields = _normalize_expert_fields(expert_fields)
    profile = COMPETENCE_PROFILES[_normalize_competence_level(competence_level)]
    in_field = label in fields

    competence = profile["in_field"] if in_field else profile["out_field"]
    competence += CLASS_DIFFICULTY.get(label, 0)

    if in_field and _has_domain_cues(text, label):
        competence += profile["keyword_boost"]

    return min(0.97, max(0.22, competence))


def simulated_expert_predict(example, expert_fields=None, competence_level=None):
    label = example["label"]
    text = example["text"]
    competence = _correctness_probability(label, expert_fields, competence_level, text)

    if deterministic_score(text, "expert-correctness") < competence:
        return label

    return choose_wrong_label_deterministic(text, label, CLASS_IDS)


def choose_wrong_label_deterministic(text, correct_label, classes=CLASS_IDS):
    alternatives = [class_id for class_id in classes if class_id != correct_label]
    wrong_index = int(deterministic_score(text, "expert-error") * len(alternatives))
    return alternatives[min(wrong_index, len(alternatives) - 1)]


def trivial_expert_predict(example, expert_fields=None):
    label = example["label"]
    text = example["text"]
    
    if label in expert_fields:
        return label
    else:
        return choose_wrong_label_deterministic(text, label, CLASS_IDS)


def evaluate_trivial_expert(test_examples, expert_fields=None):
    y_true = [example["label"] for example in test_examples]
    predictions = [trivial_expert_predict(example, expert_fields) for example in test_examples]
    return {
        "accuracy": accuracy_score(y_true, predictions),
        "predictions": predictions,
        "per_class": per_class_accuracy(y_true, predictions),
        "confusion": build_confusion_rows(y_true, predictions),   
    }


def evaluate_simulated_expert(test_examples, expert_fields=None, competence_level=None):
    y_true = [example["label"] for example in test_examples]
    predictions = [
        simulated_expert_predict(
            example,
            expert_fields=expert_fields,
            competence_level=competence_level,
        )
        for example in test_examples
    ]
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
        correct = sum(1 for label, prediction in zip(y_true, predictions) if label == class_id and prediction == class_id)
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
        rows.append({
            "label": CLASS_NAMES[class_id],
            "values": [int(value) for value in matrix[index]],
        })
    return rows

def class_metric_rows(per_class):
    rows = [
        {"class_name": name, "accuracy_percent": val["accuracy_percent"], "correct": val["correct"], "total": val["total"]}
        for name, val in per_class.items()
    ]
    total_correct = sum(row["correct"] for row in rows)
    total_samples = sum(row["total"] for row in rows)
    total_accuracy = round((total_correct / total_samples) * 100, 2) if total_samples > 0 else 0.0
    rows.append({
        "class_name": "Total",
        "accuracy_percent": total_accuracy,
        "correct": total_correct,
        "total": total_samples,
        "is_total": True
    })
    return rows

def prediction_margins(decision_scores):
    sorted_scores = np.sort(decision_scores, axis=1)
    return sorted_scores[:, -1] - sorted_scores[:, -2]

def predict_expert(example, settings):
    if not settings or settings.get("type") == "TRIVIAL":
        fields = settings.get("fields", []) if settings else None
        return trivial_expert_predict(example, expert_fields=fields)
    return simulated_expert_predict(
        example,
        expert_fields=settings.get("fields"),
        competence_level=settings.get("competence_level"),
    )

def get_expert_cost(config):
    if not config:
        return 0.0
    if config.get("cost_presence") == "present" and config.get("cost") is not None:
        try:
            return float(config["cost"])
        except (ValueError, TypeError):
            return 0.0
    return 0.0

def evaluate_confidence_threshold_defer(test_examples, baseline, expert, defer_rate, expert_configs=None):
    y_true = np.array([example["label"] for example in test_examples])
    baseline_predictions = np.array(baseline["predictions"])
    margins = prediction_margins(baseline["decision_scores"])

    if isinstance(expert, list):
        expert_results_list = expert
    else:
        expert_results_list = [expert]

    if not expert_configs:
        expert_configs = [{"cost_presence": "absent", "cost": 0.0} for _ in expert_results_list]

    num_experts = len(expert_results_list)
    expert_preds = np.column_stack([np.array(e["predictions"]) for e in expert_results_list])
    expert_costs = np.array([get_expert_cost(cfg) for cfg in expert_configs])

    defer_count = int(round(len(test_examples) * defer_rate))
    defer_count = min(max(defer_count, 0), len(test_examples))
    deferred_mask = np.zeros(len(test_examples), dtype=bool)

    if defer_count:
        deferred_indices = np.argsort(margins)[:defer_count]
        deferred_mask[deferred_indices] = True

    team_predictions = baseline_predictions.copy()
    useful_defer = 0
    harmful_defer = 0
    both_correct_defer = 0
    total_cost = 0.0

    query_allocation = {
        cid: {e_idx: {"correct": 0, "queried": 0} for e_idx in range(num_experts)}
        for cid in CLASS_IDS
    }

    for i in range(len(test_examples)):
        if deferred_mask[i]:
            if num_experts == 1:
                chosen_expert_idx = 0
            else:
                pred_label = baseline_predictions[i]
                class_name = CLASS_NAMES.get(int(pred_label))
                scores = []
                for e_idx in range(num_experts):
                    acc = expert_results_list[e_idx]["per_class"].get(class_name, {}).get("accuracy", 0.5)
                    score = acc - expert_costs[e_idx]
                    scores.append(score)
                chosen_expert_idx = int(np.argmax(scores))

            chosen_pred = expert_preds[i, chosen_expert_idx]
            team_predictions[i] = chosen_pred
            total_cost += expert_costs[chosen_expert_idx]

            true_label = int(y_true[i])
            query_allocation[true_label][chosen_expert_idx]["queried"] += 1
            if chosen_pred == true_label:
                query_allocation[true_label][chosen_expert_idx]["correct"] += 1

            b_corr = (baseline_predictions[i] == y_true[i])
            e_corr = (chosen_pred == y_true[i])
            if not b_corr and e_corr:
                useful_defer += 1
            elif b_corr and not e_corr:
                harmful_defer += 1
            elif b_corr and e_corr:
                both_correct_defer += 1

    deferred_total = int(deferred_mask.sum())
    non_deferred_total = len(test_examples) - deferred_total

    return {
        "policy_name": "Confidence threshold",
        "accuracy": accuracy_score(y_true, team_predictions),
        "deferral_rate": deferred_total / len(test_examples) if test_examples else 0,
        "deferred_total": deferred_total,
        "non_deferred_total": non_deferred_total,
        "useful_defer": useful_defer,
        "harmful_defer": harmful_defer,
        "both_correct_defer": both_correct_defer,
        "total_cost": round(total_cost, 2),
        "query_allocation": query_allocation,
        "confusion": build_confusion_rows(y_true, team_predictions),
    }

def evaluate_competence_aware_defer(test_examples, baseline, expert, competence_by_class, expert_configs=None):
    y_true = np.array([example["label"] for example in test_examples])
    baseline_predictions = np.array(baseline["predictions"])
    margins = prediction_margins(baseline["decision_scores"])

    if isinstance(expert, list):
        expert_results_list = expert
    else:
        expert_results_list = [expert]

    num_experts = len(expert_results_list)
    competence_list = []
    for e_idx in range(num_experts):
        comp_dict = {}
        for cid in CLASS_IDS:
            class_name = CLASS_NAMES[cid]
            comp_dict[cid] = expert_results_list[e_idx]["per_class"].get(class_name, {}).get("accuracy", 0.5)
        competence_list.append(comp_dict)

    if not expert_configs:
        expert_configs = [{"cost_presence": "absent", "cost": 0.0} for _ in expert_results_list]

    num_experts = len(expert_results_list)
    expert_preds = np.column_stack([np.array(e["predictions"]) for e in expert_results_list])
    expert_costs = np.array([get_expert_cost(cfg) for cfg in expert_configs])

    median_margin = max(float(np.median(margins)), 0.001)
    model_confidence = 1 / (1 + np.exp(-(margins / median_margin)))

    deferred_mask = np.zeros(len(test_examples), dtype=bool)
    team_predictions = baseline_predictions.copy()
    useful_defer = 0
    harmful_defer = 0
    advantages_list = []
    total_cost = 0.0

    query_allocation = {
        cid: {e_idx: {"correct": 0, "queried": 0} for e_idx in range(num_experts)}
        for cid in CLASS_IDS
    }

    for i in range(len(test_examples)):
        label_pred = int(baseline_predictions[i])
        m_conf = model_confidence[i]

        advantages = []
        for e_idx in range(num_experts):
            comp_dict = competence_list[e_idx] if e_idx < len(competence_list) else competence_list[0]
            e_comp = comp_dict.get(label_pred, 0.5)
            c_cost = expert_costs[e_idx]
            adv = e_comp - m_conf - c_cost
            advantages.append(adv)

        best_e_idx = int(np.argmax(advantages))
        best_adv = advantages[best_e_idx]
        advantages_list.append(best_adv)

        if best_adv > DEFAULT_EXPERT_ADVANTAGE:
            deferred_mask[i] = True
            chosen_pred = expert_preds[i, best_e_idx]
            team_predictions[i] = chosen_pred
            total_cost += expert_costs[best_e_idx]

            true_label = int(y_true[i])
            query_allocation[true_label][best_e_idx]["queried"] += 1
            if chosen_pred == true_label:
                query_allocation[true_label][best_e_idx]["correct"] += 1

            b_corr = (baseline_predictions[i] == y_true[i])
            e_corr = (chosen_pred == y_true[i])
            if not b_corr and e_corr:
                useful_defer += 1
            elif b_corr and not e_corr:
                harmful_defer += 1

    return {
        "policy_name": "Competence-aware",
        "accuracy": accuracy_score(y_true, team_predictions),
        "deferred_total": int(deferred_mask.sum()),
        "non_deferred_total": int(len(test_examples) - deferred_mask.sum()),
        "useful_defer": useful_defer,
        "harmful_defer": harmful_defer,
        "average_expert_advantage": float(np.mean(advantages_list)) if advantages_list else 0.0,
        "total_cost": round(total_cost, 2),
        "query_allocation": query_allocation,
        "confusion": build_confusion_rows(y_true, team_predictions),
    }
