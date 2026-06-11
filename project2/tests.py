import os

from django.conf import settings
from django.test import SimpleTestCase
from django.urls import reverse

from .views import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    format_feature_name,
    load_clean_penguins,
    train_decision_tree,
)


class PenguinDatasetTests(SimpleTestCase):
    def test_tree_feature_names_are_human_readable(self):
        self.assertEqual(
            format_feature_name("categorical__island_Biscoe"),
            "Island = Biscoe",
        )
        self.assertEqual(
            format_feature_name("categorical__sex_female"),
            "Sex = female",
        )
        self.assertEqual(
            format_feature_name("numeric__bill_length_mm"),
            "Bill Length Mm",
        )

    def test_clean_dataset_has_expected_columns_and_no_missing_values(self):
        penguins = load_clean_penguins()

        self.assertEqual(
            list(penguins.columns),
            FEATURE_COLUMNS + [TARGET_COLUMN],
        )
        self.assertFalse(penguins.isna().any().any())

    def test_decision_tree_returns_valid_task_one_metrics(self):
        result = train_decision_tree(load_clean_penguins())
        tree = result["pipeline"].named_steps["model"]

        self.assertGreaterEqual(result["accuracy"], 0)
        self.assertLessEqual(result["accuracy"], 1)
        self.assertLessEqual(tree.get_n_leaves(), 5)
        self.assertEqual(result["train_rows"] + result["test_rows"], 333)


class Project2ViewTests(SimpleTestCase):
    def test_project2_page_loads_with_tree_results(self):
        response = self.client.get(reverse("project2:index"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Decision Tree")
        self.assertContains(response, "Test accuracy")
        self.assertContains(response, "leaf nodes")
        self.assertContains(response, "decision_tree.png")
        self.assertContains(response, "Data Transparency")
        self.assertContains(response, "11 of 344 rows")
        self.assertContains(response, "Adelie, Chinstrap, Gentoo")
        self.assertContains(response, "One-hot encoding")

    def test_tree_image_is_generated(self):
        self.client.get(reverse("project2:index"))
        image_path = os.path.join(
            settings.MEDIA_ROOT,
            "project2",
            "decision_tree.png",
        )

        self.assertTrue(os.path.exists(image_path))
