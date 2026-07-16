from unittest.mock import patch

from django.test import RequestFactory, SimpleTestCase, TestCase
from django.urls import reverse

from . import views as project3_views
from .core.utils import build_fallback_dataset
from .views import (
    CLASS_NAMES,
    active_learning_queries,
    build_project3_results,
    evaluate_simulated_expert,
    evaluate_learning_to_defer,
    parse_sample_size,
    train_baseline_classifier,
)


class Project3ExperimentTests(SimpleTestCase):
    def test_sample_size_parser(self):
        self.assertEqual(parse_sample_size(None, 100), 100)
        self.assertEqual(parse_sample_size("all", 100), None)
        self.assertEqual(parse_sample_size("-4", 100), 100)
        self.assertEqual(parse_sample_size("25", 100), 25)

    def test_baseline_and_expert_return_metrics(self):
        train_examples, test_examples = build_fallback_dataset()
        baseline = train_baseline_classifier(train_examples, test_examples)
        expert = evaluate_simulated_expert(test_examples)
        team = evaluate_learning_to_defer(test_examples, baseline, expert, 0.25)
        active_learning = active_learning_queries(train_examples, baseline, 12)

        self.assertGreaterEqual(baseline["accuracy"], 0)
        self.assertLessEqual(baseline["accuracy"], 1)
        self.assertGreaterEqual(expert["accuracy"], 0)
        self.assertLessEqual(expert["accuracy"], 1)
        self.assertGreaterEqual(team["accuracy"], 0)
        self.assertLessEqual(team["accuracy"], 1)
        self.assertEqual(active_learning["query_budget"], 12)
        self.assertEqual(set(expert["per_class"].keys()), set(CLASS_NAMES.values()))


class Project3ViewTests(TestCase):
    def test_project3_page_loads(self):
        response = self.client.get(
            reverse("project3:index"),
            {"train-size": "40", "test-size": "20"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Active Learning for Learning-to-Defer")
        self.assertContains(response, "Baseline Classifier")
        self.assertContains(response, "Simulated Expert")
        self.assertContains(response, "Team Policy")
        self.assertContains(response, "Expert Queries")
        self.assertContains(response, "Human Expert")

    def test_build_project3_results_is_cached_for_same_parameters(self):
        project3_views.PROJECT3_RESULT_CACHE.clear()

        factory = RequestFactory()
        request = factory.get(
            "/project3/",
            {"train-size": "20", "test-size": "10", "defer-rate": "0.2", "query-budget": "8"},
        )

        dataset = {
            "train": [{"text": f"train {i}", "label": 1 + (i % 4)} for i in range(20)],
            "test": [{"text": f"test {i}", "label": 1 + (i % 4)} for i in range(10)],
            "full_train_rows": 20,
            "full_test_rows": 10,
            "source": "test",
        }
        baseline = {
            "accuracy": 0.6,
            "per_class": {name: {"correct": 1, "total": 1} for name in CLASS_NAMES.values()},
            "confusion": [],
            "pipeline": object(),
        }
        expert = {
            "accuracy": 0.7,
            "per_class": {name: {"correct": 1, "total": 1} for name in CLASS_NAMES.values()},
            "confusion": [],
        }
        active_learning = {
            "query_budget": 8,
            "estimated_competence_rows": [],
            "selected_indices": [0, 1, 2, 3],
            "queried_rows": [],
            "competence_by_class": {cid: 0.5 for cid in CLASS_NAMES},
            "strategy": "balanced_uncertainty",
        }

        class DummyClassifier:
            def __init__(self, *args, **kwargs):
                pass

            def fit(self, train_examples):
                return self

            def predict_and_evaluate(self, test_examples):
                return {
                    "policy_name": "Dummy policy",
                    "accuracy": 0.65,
                    "deferred_total": 1,
                    "useful_defer": 1,
                    "harmful_defer": 0,
                }

        with patch("project3.views.load_ag_news_dataset", return_value=dataset) as mocked_dataset, patch(
            "project3.views.train_baseline_classifier", return_value=baseline
        ) as mocked_baseline, patch(
            "project3.views.evaluate_simulated_expert", return_value=expert
        ) as mocked_expert, patch(
            "project3.views.active_learning_queries", return_value=active_learning
        ) as mocked_active_learning, patch(
            "project3.views.build_human_expert_context", return_value={"rows": [], "answered": 0, "correct": 0, "accuracy_percent": 0, "total": 0, "class_options": []}
        ), patch(
            "project3.views.compare_active_learning_strategies", return_value=[]
        ), patch(
            "project3.views.evaluate_learning_to_defer",
            return_value={"policy_name": "Confidence threshold", "accuracy": 0.65, "deferred_total": 2, "non_deferred_total": 8, "useful_defer": 1, "harmful_defer": 0},
        ), patch(
            "project3.views.evaluate_competence_aware_defer",
            return_value={"policy_name": "Competence-aware", "accuracy": 0.65, "deferred_total": 2, "non_deferred_total": 8, "useful_defer": 1, "harmful_defer": 0},
        ), patch("project3.views.class_metric_rows", return_value=[]), patch(
            "project3.views.save_bar_plot", return_value="/media/mock.png"
        ), patch("project3.views.TrueL2DClassifier", DummyClassifier), patch(
            "project3.views.TrueL2DClassifier_2", DummyClassifier
        ):
            build_project3_results(request)
            build_project3_results(request)

        self.assertEqual(mocked_dataset.call_count, 2)
        self.assertEqual(mocked_baseline.call_count, 1)
        self.assertEqual(mocked_expert.call_count, 1)
        self.assertEqual(mocked_active_learning.call_count, 1)

    def test_human_labels_can_be_submitted(self):
        response = self.client.get(
            reverse("project3:index"),
            {"train-size": "40", "test-size": "20"},
        )
        self.assertEqual(response.status_code, 200)

        label_inputs = [
            key
            for key in response.context["human_expert"]["rows"][0].keys()
            if key == "key"
        ]
        self.assertEqual(label_inputs, ["key"])
        first_key = response.context["human_expert"]["rows"][0]["key"]
        response = self.client.post(
            reverse("project3:index") + "?train-size=40&test-size=20",
            {f"human_label_{first_key}": "1"},
        )

        self.assertEqual(response.status_code, 302)
