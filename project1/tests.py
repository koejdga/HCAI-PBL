from django.test import SimpleTestCase

from .views import build_baseline_comparison, format_metric_value, train_model


class BaselineModelTests(SimpleTestCase):
    def test_classification_result_includes_majority_class_baseline(self):
        dataset = {
            "column_names": ["feature", "target"],
            "columns": {
                "feature": {
                    "values": [str(value) for value in range(30)],
                    "type": "numeric",
                },
                "target": {
                    "values": ["majority"] * 20 + ["minority"] * 10,
                    "type": "categorical",
                },
            },
            "features": ["feature"],
            "target": "target",
            "task_type": "classification",
            "row_count": 30,
        }

        result = train_model(dataset, "logistic_regression", 20)

        self.assertEqual(result["baseline"]["name"], "Most-frequent class")
        self.assertGreaterEqual(result["baseline"]["score"], 0)
        self.assertLessEqual(result["baseline"]["score"], 1)
        self.assertIsNone(result["baseline"]["rmse"])
        self.assertIn(result["comparison"]["status"], {"better", "similar", "worse"})
        self.assertIn("percentage points", result["comparison"]["difference_label"])
        self.assertTrue(result["best"]["display_score"].endswith("%"))
        self.assertTrue(result["baseline"]["display_score"].endswith("%"))

    def test_regression_result_includes_mean_baseline_metrics(self):
        dataset = {
            "column_names": ["feature", "target"],
            "columns": {
                "feature": {
                    "values": [str(value) for value in range(30)],
                    "type": "numeric",
                },
                "target": {
                    "values": [str(value * 2) for value in range(30)],
                    "type": "numeric",
                },
            },
            "features": ["feature"],
            "target": "target",
            "task_type": "regression",
            "row_count": 30,
        }

        result = train_model(dataset, "ridge_regression", 20)

        self.assertEqual(result["baseline"]["name"], "Training-target mean")
        self.assertIsInstance(result["baseline"]["score"], float)
        self.assertIsInstance(result["baseline"]["rmse"], float)
        self.assertGreaterEqual(result["baseline"]["rmse"], 0)
        self.assertIn("R2", result["comparison"]["difference_label"])

    def test_comparison_messages_cover_better_similar_and_worse_results(self):
        better = build_baseline_comparison("classification", 0.8, 0.5)
        similar = build_baseline_comparison("classification", 0.5, 0.5)
        worse = build_baseline_comparison("classification", 0.4, 0.5)

        self.assertEqual(better["status"], "better")
        self.assertEqual(similar["status"], "similar")
        self.assertEqual(worse["status"], "worse")

    def test_metric_values_are_formatted_for_non_expert_readers(self):
        self.assertEqual(format_metric_value("classification", 0.875), "87.50%")
        self.assertEqual(format_metric_value("regression", 0.875), "0.8750")
