from django.test import SimpleTestCase, TestCase
from django.urls import reverse

from .views import (
    CLASS_NAMES,
    active_learning_queries,
    build_fallback_dataset,
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
