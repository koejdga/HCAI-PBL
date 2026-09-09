from django.test import SimpleTestCase, TestCase
from django.core.files.uploadedfile import SimpleUploadedFile
from django.urls import reverse

from .views import (
    build_baseline_comparison,
    format_metric_value,
    train_model,
)
from .core.dataset import configure_dataset, parse_csv_dataset
from .core.training import train_all_and_compare


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


class Project1WorkflowTests(TestCase):
    def _csv_upload(self, name="mini.csv"):
        content = (
            "id,sepal_length,sepal_width,species\n"
            "1,5.1,3.5,setosa\n"
            "2,4.9,3.0,setosa\n"
            "3,6.2,3.4,versicolor\n"
            "4,5.9,3.0,versicolor\n"
            "5,6.9,3.1,virginica\n"
            "6,6.7,3.0,virginica\n"
        )
        return SimpleUploadedFile(name, content.encode("utf-8"), content_type="text/csv")

    def test_csv_upload_accepts_csv_and_rejects_other_files(self):
        response = self.client.post(
            reverse("project1:index"),
            {"action": "upload", "file": self._csv_upload()},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.client.session["project1_dataset"]["target"], "species")
        self.assertContains(response, "What should I check before trusting the data?")

        response = self.client.post(
            reverse("project1:index"),
            {
                "action": "upload",
                "file": SimpleUploadedFile(
                    "notes.txt",
                    b"not,a,project,csv",
                    content_type="text/plain",
                ),
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Please upload a CSV file.")

    def test_dataset_configuration_can_change_target_features_and_missing_policy(self):
        uploaded = SimpleUploadedFile(
            "missing.csv",
            (
                "id,feature_a,feature_b,target\n"
                "1,1.0,low,A\n"
                "2,,high,B\n"
                "3,3.0,low,A\n"
            ).encode("utf-8"),
            content_type="text/csv",
        )
        dataset = parse_csv_dataset(uploaded)

        configured = configure_dataset(
            dataset,
            target="target",
            features=["feature_a", "feature_b"],
            task_type="classification",
            drop_missing_rows=True,
        )

        self.assertEqual(configured["target"], "target")
        self.assertEqual(configured["features"], ["feature_a", "feature_b"])
        self.assertEqual(configured["task_type"], "classification")
        self.assertEqual(configured["dropped_row_count"], 1)
        self.assertEqual(configured["row_count"], 2)

    def test_ajax_visualization_creates_feature_plot(self):
        self.client.post(
            reverse("project1:index"),
            {"action": "upload", "file": self._csv_upload()},
        )

        response = self.client.post(
            reverse("project1:index"),
            {"action": "feature_target", "feature": "sepal_length"},
            HTTP_X_REQUESTED_WITH="XMLHttpRequest",
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "created")
        self.assertEqual(data["plot_type"], "feature_target")
        self.assertIn("url", data["plot"])

    def test_train_all_and_compare_returns_ranked_results(self):
        dataset = {
            "column_names": ["feature", "target"],
            "columns": {
                "feature": {
                    "values": [str(value) for value in range(40)],
                    "type": "numeric",
                },
                "target": {
                    "values": ["low"] * 20 + ["high"] * 20,
                    "type": "categorical",
                },
            },
            "features": ["feature"],
            "target": "target",
            "task_type": "classification",
            "row_count": 40,
        }

        results = train_all_and_compare(dataset, 25)

        self.assertGreaterEqual(len(results), 2)
        self.assertGreaterEqual(results[0]["score"], results[-1]["score"])
        self.assertIn("display_baseline", results[0])
        self.assertIn(results[0]["comparison"]["status"], {"better", "similar", "worse"})
