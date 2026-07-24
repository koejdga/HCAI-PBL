from unittest.mock import patch

from django.test import RequestFactory, SimpleTestCase, TestCase
from django.urls import reverse

from . import views as project3_views
from .core.utils import build_fallback_dataset
from .core.deferral import _correctness_probability, evaluate_simulated_expert
from .views import (
    CLASS_NAMES,
    active_learning_queries,
    build_project3_results,
    evaluate_confidence_threshold_defer,
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
        team = evaluate_confidence_threshold_defer(test_examples, baseline, expert, 0.25)
        active_learning = active_learning_queries(train_examples, baseline, 12)

        self.assertGreaterEqual(baseline["accuracy"], 0)
        self.assertLessEqual(baseline["accuracy"], 1)
        self.assertGreaterEqual(expert["accuracy"], 0)
        self.assertLessEqual(expert["accuracy"], 1)
        self.assertGreaterEqual(team["accuracy"], 0)
        self.assertLessEqual(team["accuracy"], 1)
        self.assertEqual(active_learning["query_budget"], 12)
        self.assertEqual(set(expert["per_class"].keys()), set(CLASS_NAMES.values()))

    def test_simulated_expert_respects_fields_and_competence_level(self):
        _, test_examples = build_fallback_dataset()

        sports_specialist = evaluate_simulated_expert(
            test_examples, expert_fields=[2], competence_level="more-competent"
        )
        less_competent_specialist = evaluate_simulated_expert(
            test_examples, expert_fields=[2], competence_level="less-competent"
        )

        self.assertGreater(
            sports_specialist["per_class"]["Sports"]["accuracy"],
            sports_specialist["per_class"]["Business"]["accuracy"],
        )
        self.assertGreater(
            sports_specialist["accuracy"],
            less_competent_specialist["accuracy"],
        )
        self.assertGreater(
            _correctness_probability(2, [2], "more-competent", "The team won the game."),
            _correctness_probability(3, [2], "more-competent", "The bank reported earnings."),
        )

    def test_competence_aware_defer_trivial_expert(self):
        from .core.deferral import evaluate_competence_aware_defer, evaluate_trivial_expert
        from .core.utils import CLASS_IDS

        train_examples, test_examples = build_fallback_dataset()
        baseline = train_baseline_classifier(train_examples, test_examples)
        expert = evaluate_trivial_expert(test_examples, expert_fields=CLASS_IDS)
        expert_configs = [{
            "prefix": "expert-one",
            "type": "TRIVIAL",
            "fields": CLASS_IDS,
            "competence_level": "",
            "cost_presence": "absent",
            "cost": 0.0
        }]

        results = evaluate_competence_aware_defer(
            test_examples,
            baseline,
            expert,
            competence_by_class={},
            expert_configs=expert_configs
        )

        self.assertGreater(results["deferred_total"], 0)
        self.assertGreaterEqual(results["useful_defer"], 0)
        self.assertEqual(results["harmful_defer"], 0)


class Project3ViewTests(TestCase):
    def test_project3_page_loads(self):
        response = self.client.get(
            reverse("project3:index"),
            {"train-size": "40", "test-size": "20", "expert-one-type": "REALISTIC"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Active Learning for Learning-to-Defer")
        self.assertContains(response, "Baseline Classifier")
        self.assertContains(response, "Simulated Expert")
        self.assertContains(response, "Policy Comparison")
        self.assertContains(response, "Expert Queries")
        self.assertContains(response, "Human Expert")

    def test_project3_page_loads_no_experts(self):
        response = self.client.get(
            reverse("project3:index"),
            {"train-size": "40", "test-size": "20"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Active Learning for Learning-to-Defer")
        self.assertContains(response, "Baseline Classifier")
        self.assertContains(response, "No experts have been configured yet")
        self.assertIn("error_message", response.context)
        self.assertNotIn("policy_rows", response.context)

    def test_build_project3_results_is_cached_for_same_parameters(self):
        project3_views.PROJECT3_RESULT_CACHE.clear()

        factory = RequestFactory()
        request = factory.get(
            "/project3/",
            {"train-size": "20", "test-size": "10", "defer-rate": "0.2", "expert-one-type": "REALISTIC"},
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

            def fit(self, train_examples, *args, **kwargs):
                return self

            def predict_and_evaluate(self, test_examples, *args, **kwargs):
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
            "project3.views.evaluate_confidence_threshold_defer",
            return_value={"policy_name": "Confidence threshold", "accuracy": 0.65, "deferred_total": 2, "non_deferred_total": 8, "useful_defer": 1, "harmful_defer": 0},
        ), patch(
            "project3.views.evaluate_competence_aware_defer",
            return_value={"policy_name": "Competence-aware", "accuracy": 0.65, "deferred_total": 2, "non_deferred_total": 8, "useful_defer": 1, "harmful_defer": 0},
        ), patch("project3.views.class_metric_rows", return_value=[]), patch(
            "project3.views.save_bar_plot", return_value="/media/mock.png"
        ), patch("project3.views.L2DClassifier", DummyClassifier):
            build_project3_results(request)
            build_project3_results(request)

        self.assertEqual(mocked_dataset.call_count, 2)
        self.assertEqual(mocked_baseline.call_count, 1)
        self.assertEqual(mocked_expert.call_count, 1)
        self.assertEqual(mocked_active_learning.call_count, 3)

    def test_expert_accuracy_preview_returns_json(self):
        html_response = self.client.get(reverse("project3:index"), {"expert-one-type": "REALISTIC"})
        self.assertEqual(html_response.status_code, 200)
        self.assertContains(html_response, 'class="table-wrap accuracy-preview-table-wrap"')
        self.assertContains(html_response, 'id="expert-class-accuracy-body"')

        response = self.client.get(
            reverse("project3:index"),
            {
                "format": "json",
                "expert-one": "TRIVIAL",
                "expert-one-fields": "2",
                "expert-two": "REALISTIC",
                "expert-two-fields": "1,4",
                "expert-two-competence-level": "more-competent",
                "expert-two-cost-presence": "present",
                "expert-two-cost": "0.5",
                "test-size": "20",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["Content-Type"].split(";")[0], "application/json")
        data = response.json()
        self.assertIn("accuracy_percent", data)
        self.assertIn("experts", data)
        self.assertIn("expert_class_rows", data)
        self.assertIn("expert_settings", data)
        self.assertEqual(len(data["experts"]), 2)
        self.assertEqual(data["expert_count"], 2)
        self.assertEqual(len(data["expert_class_rows"]), len(CLASS_NAMES) + 1)
        self.assertEqual(data["expert_settings"][0]["fields"], [2])
        self.assertEqual(data["expert_settings"][1]["competence_level"], "more-competent")
        self.assertEqual(data["expert_settings"][1]["cost"], 0.5)
        first_row = data["expert_class_rows"][0]
        self.assertIn("accuracy_percent", first_row)
        self.assertIn("correct", first_row)
        self.assertIn("total", first_row)
        self.assertIn("expert_2_accuracy_percent", first_row)
        self.assertIn("expert_2_correct", first_row)
        self.assertIn("expert_2_total", first_row)

    def test_human_labels_can_be_submitted(self):
        response = self.client.get(
            reverse("project3:index"),
            {"train-size": "40", "test-size": "20", "expert-one-type": "REALISTIC"},
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
