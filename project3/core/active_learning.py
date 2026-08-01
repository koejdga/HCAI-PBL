import hashlib
import random
import numpy as np
from sklearn.metrics import accuracy_score
from .utils import CLASS_IDS, CLASS_NAMES
from .deferral import prediction_margins, simulated_expert_predict, per_class_accuracy, class_metric_rows, predict_expert

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

    if strategy == "stream_selective":
        threshold = float(np.quantile(margins, 0.35)) if len(margins) else 0.0
        selected_indices = [
            idx for idx, margin in enumerate(margins)
            if margin <= threshold
        ]
        if len(selected_indices) < query_budget:
            selected_set = set(selected_indices)
            selected_indices.extend(
                idx for idx in np.argsort(margins)
                if idx not in selected_set
            )
        return [int(index) for index in selected_indices[:query_budget]]

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

def active_learning_queries(train_examples, baseline, query_budget, strategy="balanced_uncertainty", expert_configs=None):
    texts = [example["text"] for example in train_examples]
    decision_scores = baseline["pipeline"].decision_function(texts)
    margins = prediction_margins(decision_scores)
    predicted_labels = baseline["pipeline"].predict(texts)
    
    cfg1 = expert_configs[0] if expert_configs else None
    budget1 = cfg1.get("query_budget", query_budget) if cfg1 else query_budget
    cfg2 = expert_configs[1] if (expert_configs and len(expert_configs) > 1) else None
    budget2 = cfg2.get("query_budget", query_budget) if cfg2 else query_budget

    if cfg1 is None:
        expert1_predictions = [simulated_expert_predict(example) for example in train_examples]
    else:
        expert1_predictions = [predict_expert(example, cfg1) for example in train_examples]

    selected_indices1 = selected_query_indices(strategy, train_examples, predicted_labels, margins, budget1)
    selected_true1 = [train_examples[idx]["label"] for idx in selected_indices1]
    selected_expert1 = [expert1_predictions[idx] for idx in selected_indices1]
    selected_predicted1 = [int(predicted_labels[idx]) for idx in selected_indices1]

    estimated_competence1 = per_class_accuracy(selected_true1, selected_expert1)
    competence_by_class1 = competence_from_queries(selected_true1, selected_expert1)
    agreement1 = accuracy_score(selected_predicted1, selected_expert1) if selected_indices1 else 0

    estimated_competence_rows = []
    for cid in CLASS_IDS:
        class_name = CLASS_NAMES[cid]
        row = {
            "class_name": class_name,
            "accuracy_percent": round(estimated_competence1[class_name]["accuracy"] * 100, 2),
            "correct": estimated_competence1[class_name]["correct"],
            "total": estimated_competence1[class_name]["total"],
        }
        estimated_competence_rows.append(row)

    if cfg2 is not None:
        expert2_predictions = [predict_expert(example, cfg2) for example in train_examples]
        selected_indices2 = selected_query_indices(strategy, train_examples, predicted_labels, margins, budget2)
        selected_true2 = [train_examples[idx]["label"] for idx in selected_indices2]
        selected_expert2 = [expert2_predictions[idx] for idx in selected_indices2]
        estimated_competence2 = per_class_accuracy(selected_true2, selected_expert2)
        
        for row in estimated_competence_rows:
            class_name = row["class_name"]
            row["expert_2_accuracy_percent"] = round(estimated_competence2[class_name]["accuracy"] * 100, 2)
            row["expert_2_correct"] = estimated_competence2[class_name]["correct"]
            row["expert_2_total"] = estimated_competence2[class_name]["total"]

    # Calculate and append Total row
    total_correct1 = sum(row["correct"] for row in estimated_competence_rows)
    total_total1 = sum(row["total"] for row in estimated_competence_rows)
    total_accuracy1 = round(total_correct1 / total_total1 * 100, 2) if total_total1 > 0 else 0.0

    total_row = {
        "class_name": "Total",
        "accuracy_percent": total_accuracy1,
        "correct": total_correct1,
        "total": total_total1,
    }

    if cfg2 is not None:
        total_correct2 = sum(row["expert_2_correct"] for row in estimated_competence_rows)
        total_total2 = sum(row["expert_2_total"] for row in estimated_competence_rows)
        total_accuracy2 = round(total_correct2 / total_total2 * 100, 2) if total_total2 > 0 else 0.0
        total_row["expert_2_accuracy_percent"] = total_accuracy2
        total_row["expert_2_correct"] = total_correct2
        total_row["expert_2_total"] = total_total2

    estimated_competence_rows.append(total_row)

    rows = []
    for idx in selected_indices1[:8]:
        example = train_examples[idx]
        rows.append({
            "id": str(idx),
            "model_class": CLASS_NAMES[int(predicted_labels[idx])],
            "expert_class": CLASS_NAMES[expert1_predictions[idx]],
            "true_class": CLASS_NAMES[example["label"]],
            "margin": round(float(margins[idx]), 3),
            "text": example["text"],
        })

    return {
        "strategy": strategy,
        "query_budget": len(selected_indices1),
        "agreement_percent": round(agreement1 * 100, 2),
        "competence_by_class": competence_by_class1,
        "estimated_competence_rows": estimated_competence_rows,
        "selected_indices": selected_indices1,
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
            "text": example["text"],
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

def compare_active_learning_strategies(train_examples, test_examples, baseline, query_budget, expert_configs=None):
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC
    from sklearn.metrics import accuracy_score
    from sklearn.metrics.pairwise import cosine_similarity

    strategies = [
        ("balanced_uncertainty", "Balanced uncertainty"),
        ("uncertainty", "Uncertainty only"),
        ("random", "Random sample"),
        ("stream_selective", "Stream selective sampling"),
    ]
    rows = []
    for strategy, label in strategies:
        result = active_learning_queries(train_examples, baseline, query_budget, strategy, expert_configs=expert_configs)
        competence_values = list(result["competence_by_class"].values())
        
        # Calculate retraining accuracy
        selected_indices = result.get("selected_indices", [])
        X_queried = [train_examples[idx]["text"] for idx in selected_indices]
        y_queried = [train_examples[idx]["label"] for idx in selected_indices]
        
        if len(set(y_queried)) >= 2:
            pipeline = Pipeline([
                ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2), max_features=5000, min_df=1)),
                ("model", LinearSVC(random_state=42)),
            ])
            pipeline.fit(X_queried, y_queried)
            X_test = [ex["text"] for ex in test_examples]
            y_test = [ex["label"] for ex in test_examples]
            test_preds = pipeline.predict(X_test)
            retrained_acc = round(float(accuracy_score(y_test, test_preds)) * 100, 2)
        else:
            retrained_acc = 25.0
            
        # Calculate Query Redundancy / Overlap (avg pairwise cosine similarity of queries)
        if len(X_queried) > 1:
            vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
            X_queried_tfidf = vectorizer.fit_transform(X_queried)
            sim_matrix = cosine_similarity(X_queried_tfidf)
            n = sim_matrix.shape[0]
            avg_sim = (np.sum(sim_matrix) - n) / (n * (n - 1))
            redundancy = round(float(avg_sim) * 100, 2)
        else:
            redundancy = 0.0
            
        diversity = round(100.0 - redundancy, 2)
        
        rows.append({
            "strategy": strategy,
            "label": label,
            "agreement_percent": result["agreement_percent"],
            "average_competence_percent": round(float(np.mean(competence_values)) * 100, 2),
            "min_competence_percent": round(float(np.min(competence_values)) * 100, 2),
            "model_accuracy_percent": retrained_acc,
            "redundancy_percent": redundancy,
            "diversity_percent": diversity,
        })
    return rows
