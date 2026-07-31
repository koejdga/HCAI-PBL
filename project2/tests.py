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
    DEFAULT_LAMBDA,
    MAX_LAMBDA,
    FEATURE_EFFECT_NUMERIC_FEATURES,
    CLASS_NAMES,
    compute_ale,
    compute_pdp,
    parse_lambda,
    split_penguin_data,
    train_logistic_regression,
    train_regression_candidates,
    train_tree_candidates,
    find_counterfactuals,
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

    def test_lambda_parser_handles_invalid_values(self):
        self.assertEqual(parse_lambda(None), DEFAULT_LAMBDA)
        self.assertEqual(parse_lambda("invalid"), DEFAULT_LAMBDA)
        self.assertEqual(parse_lambda("-2"), 0.0)
        self.assertEqual(parse_lambda("4"), MAX_LAMBDA)

    def test_candidate_with_highest_score_is_selected(self):
        selected, candidates = train_tree_candidates(
            load_clean_penguins(),
            lambda_value=0.5,
        )

        expected = max(
            candidates,
            key=lambda candidate: candidate["selection_score"],
        )

        self.assertIs(selected, expected)
        self.assertGreater(len(candidates), 1)

    def test_logistic_candidate_with_highest_score_is_selected(self):
        selected, candidates = train_regression_candidates(
            load_clean_penguins(),
            lambda_value=0.5,
        )

        expected = max(
            candidates,
            key=lambda candidate: candidate["selection_score"],
        )

        self.assertIs(selected, expected)
        self.assertGreater(len(candidates), 1)
        self.assertGreaterEqual(selected["non_zero_weights"], 0)

    def test_feature_effect_outputs_have_species_curves(self):
        penguins = load_clean_penguins()
        pipeline = train_logistic_regression(split_penguin_data(penguins))["pipeline"]
        feature_name = FEATURE_EFFECT_NUMERIC_FEATURES[0]

        pdp_data = compute_pdp(pipeline, penguins, feature_name, grid_size=6)
        ale_data = compute_ale(pipeline, penguins, feature_name, bins=5)

        self.assertEqual(set(pdp_data["curves"].keys()), set(CLASS_NAMES))
        self.assertEqual(set(ale_data["curves"].keys()), set(CLASS_NAMES))
        self.assertEqual(len(pdp_data["x_values"]), 6)
        self.assertGreaterEqual(len(ale_data["x_values"]), 1)
        for class_name in CLASS_NAMES:
            self.assertEqual(len(pdp_data["curves"][class_name]), 6)
            self.assertEqual(
                len(ale_data["curves"][class_name]),
                len(ale_data["x_values"]),
            )

    def test_counterfactual_search_retries_when_needed(self):
        class NeverMatchingPipeline:
            def predict(self, rows):
                return ["Not a penguin species"] * len(rows)

        penguins = load_clean_penguins()
        original_x = penguins.iloc[0][FEATURE_COLUMNS].to_dict()
        rows, attempted_rows = find_counterfactuals(
            NeverMatchingPipeline(),
            penguins,
            original_x,
            desired_class="Gentoo",
        )

        self.assertEqual(rows, [])
        self.assertGreater(attempted_rows, 2000)


class Project2ViewTests(SimpleTestCase):
    def test_project2_page_loads_with_tree_results(self):
        response = self.client.get(reverse("project2:index"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Decision Tree")
        self.assertContains(response, "Contents")
        self.assertContains(response, "#model-selection")
        self.assertContains(response, "Test accuracy")
        self.assertContains(response, "leaf nodes")
        self.assertContains(response, "decision_tree.png")
        self.assertContains(response, "Data Transparency")
        self.assertContains(response, "11 of 344 rows")
        self.assertContains(response, "Adelie, Chinstrap, Gentoo")
        self.assertContains(response, "One-hot encoding")
        self.assertContains(response, "Simplicity preference")
        self.assertContains(response, "Selection score")
        self.assertContains(response, "higher is better")
        self.assertContains(response, "test accuracy -")
        self.assertContains(response, "Apply preference")

    def test_tree_image_is_generated(self):
        self.client.get(reverse("project2:index"))
        image_path = os.path.join(
            settings.MEDIA_ROOT,
            "project2",
            "decision_tree.png",
        )

        self.assertTrue(os.path.exists(image_path))
