import hashlib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from .utils import CLASS_IDS, CLASS_NAMES
from enum import Enum

DEFAULT_DEFER_RATE = 0.30
DEFAULT_QUERY_BUDGET = 120
DEFAULT_EXPERT_ADVANTAGE = 0.08

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

def simulated_expert_predict(example):
    label = example["label"]
    text = example["text"]
    lower_text = text.lower()
    strengths = {1: 0.70, 2: 0.90, 3: 0.58, 4: 0.78}
    
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
    
    return choose_wrong_label_deterministic(text, label, CLASS_IDS)


def choose_wrong_label_deterministic(text, correct_label, classes=CLASS_IDS):
    alternatives = [class_id for class_id in classes if class_id != correct_label]
    wrong_index = int(deterministic_score(text, "expert-error") * len(alternatives))
    return alternatives[min(wrong_index, len(alternatives) - 1)]


class TrivialExpert(Enum):
    ALWAYS_CORRECT = 1
    ALWAYS_INCORRECT = 2
    ALWAYS_CORRECT_IN_ONE_FIELD = 3
    ALWAYS_CORRECT_IN_TWO_FIELDS = 4


def trivial_expert_predict(example, expert_type: TrivialExpert, expert_field=None, second_expert_field=None):
    label = example["label"]
    text = example["text"]

    match expert_type:
        case TrivialExpert.ALWAYS_CORRECT:
            return label
        case TrivialExpert.ALWAYS_INCORRECT:
            return choose_wrong_label_deterministic(text, label, CLASS_IDS)
        case TrivialExpert.ALWAYS_CORRECT_IN_ONE_FIELD:
            if expert_field is None:
                raise "Expert field must be specified for an expert with one expert field"
            if label == expert_field:
                return label
            else:
                return choose_wrong_label_deterministic(text, label, CLASS_IDS)
        case TrivialExpert.ALWAYS_CORRECT_IN_TWO_FIELDS:
            if expert_field is None or second_expert_field is None:
                raise "Two expert fields must be specified for an expert with two expert fields"
            if label == expert_field or label == second_expert_field:
                return label
            else:
                return choose_wrong_label_deterministic(text, label, CLASS_IDS)


def evaluate_trivial_expert(test_examples, expert_type: TrivialExpert, expert_field=None, second_expert_field=None):
    y_true = [example["label"] for example in test_examples]
    predictions = [trivial_expert_predict(example, expert_type, expert_field, second_expert_field) for example in test_examples]
    return {
        "accuracy": accuracy_score(y_true, predictions),
        "predictions": predictions,
        "per_class": per_class_accuracy(y_true, predictions),
        "confusion": build_confusion_rows(y_true, predictions),   
    }


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
    return [
        {"class_name": name, "accuracy_percent": val["accuracy_percent"], "correct": val["correct"], "total": val["total"]}
        for name, val in per_class.items()
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

    useful_defer = int(np.sum(deferred_mask & (baseline_predictions != y_true) & (expert_predictions == y_true)))
    harmful_defer = int(np.sum(deferred_mask & (baseline_predictions == y_true) & (expert_predictions != y_true)))
    both_correct_defer = int(np.sum(deferred_mask & (baseline_predictions == y_true) & (expert_predictions == y_true)))

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

def evaluate_competence_aware_defer(test_examples, baseline, expert, competence_by_class):
    y_true = np.array([example["label"] for example in test_examples])
    baseline_predictions = np.array(baseline["predictions"])
    expert_predictions = np.array(expert["predictions"])
    margins = prediction_margins(baseline["decision_scores"])

    median_margin = max(float(np.median(margins)), 0.001)
    model_confidence = 1 / (1 + np.exp(-(margins / median_margin)))
    expert_competence = np.array([competence_by_class.get(int(label), 0.5) for label in baseline_predictions])
    expert_advantage = expert_competence - model_confidence
    deferred_mask = expert_advantage > DEFAULT_EXPERT_ADVANTAGE
    team_predictions = np.where(deferred_mask, expert_predictions, baseline_predictions)

    useful_defer = int(np.sum(deferred_mask & (baseline_predictions != y_true) & (expert_predictions == y_true)))
    harmful_defer = int(np.sum(deferred_mask & (baseline_predictions == y_true) & (expert_predictions != y_true)))

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
