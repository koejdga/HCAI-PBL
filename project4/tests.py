from pathlib import Path
import json
import unittest

import numpy as np
import pandas as pd
from django.test import TestCase
from django.urls import reverse

from .feature_representation import build_movie_feature_matrix
from .preference_learning import (
    collect_comparisons,
    fit_preference_vector,
    learn_and_recommend,
    pairwise_choices_to_comparisons,
    ranking_to_comparisons,
    recommend_movies,
)


class Project4FeatureRepresentationTests(unittest.TestCase):
    def test_movie_feature_matrix_is_numeric_and_complete(self):
        dataset_path = Path(__file__).resolve().parent / "movie_metadata.csv"
        df = pd.read_csv(dataset_path).head(150)

        matrix = build_movie_feature_matrix(df, top_n_genres=8, top_n_directors=8)

        self.assertGreater(len(matrix.movies), 0)
        self.assertGreater(len(matrix.feature_columns), 0)
        self.assertEqual(len(matrix.movies), len(matrix.features))
        self.assertIn("movie_id", matrix.features.columns)
        self.assertFalse(matrix.features[matrix.feature_columns].isnull().any().any())
        self.assertEqual(
            len(matrix.feature_columns),
            len(matrix.features[matrix.feature_columns].select_dtypes(include=["number"]).columns),
        )
        self.assertIn("Feature vectors are content-based", " ".join(matrix.assumptions))


class Project4PreferenceLearningTests(unittest.TestCase):
    def setUp(self):
        self.features = pd.DataFrame(
            [
                {
                    "movie_id": "Action Favorite",
                    "movie_title": "Action Favorite",
                    "feature_genre_Action": 1.0,
                    "feature_genre_Drama": 0.0,
                    "feature_imdb_score_normalized": 1.0,
                },
                {
                    "movie_id": "Drama Low",
                    "movie_title": "Drama Low",
                    "feature_genre_Action": 0.0,
                    "feature_genre_Drama": 1.0,
                    "feature_imdb_score_normalized": 0.1,
                },
                {
                    "movie_id": "Balanced Good",
                    "movie_title": "Balanced Good",
                    "feature_genre_Action": 0.5,
                    "feature_genre_Drama": 0.5,
                    "feature_imdb_score_normalized": 0.8,
                },
                {
                    "movie_id": "Quiet Drama",
                    "movie_title": "Quiet Drama",
                    "feature_genre_Action": 0.0,
                    "feature_genre_Drama": 1.0,
                    "feature_imdb_score_normalized": 0.4,
                },
                {
                    "movie_id": "Action Candidate",
                    "movie_title": "Action Candidate",
                    "feature_genre_Action": 1.0,
                    "feature_genre_Drama": 0.0,
                    "feature_imdb_score_normalized": 0.9,
                },
                {
                    "movie_id": "Drama Candidate",
                    "movie_title": "Drama Candidate",
                    "feature_genre_Action": 0.0,
                    "feature_genre_Drama": 1.0,
                    "feature_imdb_score_normalized": 0.2,
                },
            ]
        )
        self.feature_columns = [
            "feature_genre_Action",
            "feature_genre_Drama",
            "feature_imdb_score_normalized",
        ]

    def test_pairwise_and_ranking_payloads_convert_to_comparisons(self):
        pairwise = pairwise_choices_to_comparisons(
            [
                {"left_id": "Action Favorite", "right_id": "Drama Low", "choice": "left"},
                {"winner_id": "Balanced Good", "loser_id": "Quiet Drama"},
            ]
        )
        ranking = ranking_to_comparisons(["Action Favorite", "Balanced Good", "Drama Low"])

        self.assertEqual(pairwise[0].preferred_id, "Action Favorite")
        self.assertEqual(pairwise[0].other_id, "Drama Low")
        self.assertEqual(len(ranking), 3)
        self.assertIn(("Action Favorite", "Drama Low"), [(c.preferred_id, c.other_id) for c in ranking])

    def test_fit_scores_and_recommendations_exclude_elicitation_movies(self):
        comparisons = collect_comparisons(
            pairwise_choices=[
                {"preferred_id": "Action Favorite", "other_id": "Drama Low"},
                {"preferred_id": "Action Favorite", "other_id": "Quiet Drama"},
            ],
            rankings=["Action Favorite", "Balanced Good", "Quiet Drama", "Drama Low"],
        )

        fit = fit_preference_vector(self.features, comparisons, feature_columns=self.feature_columns)
        result = recommend_movies(
            self.features,
            fit,
            shown_movie_ids=["Action Favorite", "Drama Low", "Balanced Good", "Quiet Drama"],
            top_n=2,
            feature_labels={"feature_genre_Action": "genre: Action"},
        )

        recommended_ids = {movie["movie_id"] for movie in result.recommendations}
        self.assertTrue(recommended_ids)
        self.assertFalse(recommended_ids.intersection({"Action Favorite", "Drama Low", "Balanced Good", "Quiet Drama"}))
        self.assertIn("Action Candidate", recommended_ids)
        self.assertTrue(any(movie["explanations"] for movie in result.recommendations))
        self.assertEqual(fit.comparisons_used, len(comparisons))
        self.assertEqual(fit.source_counts["ranking"], 6)

    def test_neutral_vector_is_returned_without_preferences(self):
        fit = fit_preference_vector(self.features, [], feature_columns=self.feature_columns)
        result = recommend_movies(self.features, fit, shown_movie_ids=[], top_n=1)

        self.assertTrue(np.allclose(fit.weights, np.zeros(len(self.feature_columns))))
        self.assertIn("No valid preferences were submitted", " ".join(fit.warnings))
        self.assertEqual(len(result.recommendations), 1)

    def test_end_to_end_helper_returns_transparent_recommendations(self):
        result = learn_and_recommend(
            self.features,
            pairwise_choices=[{"preferred_id": "Action Favorite", "other_id": "Drama Low"}],
            rankings=["Action Favorite", "Balanced Good", "Quiet Drama"],
            shown_movie_ids=["Action Favorite", "Balanced Good", "Quiet Drama"],
            top_n=2,
            feature_columns=self.feature_columns,
        )

        self.assertLessEqual(len(result.recommendations), 2)
        self.assertIn("Movies already shown", " ".join(result.assumptions))
        self.assertIn("ranking", result.fit.source_counts)


class Project4ViewTests(TestCase):
    def post_json(self, url_name, payload):
        return self.client.post(
            reverse(url_name),
            data=json.dumps(payload),
            content_type="application/json",
        )

    def test_landing_page_exposes_report_and_study_actions(self):
        response = self.client.get(reverse("project4:index"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Download PDF Report")
        self.assertContains(response, "Start Study")
        self.assertContains(response, "Feature Representation")

    def test_study_page_creates_session_stable_sample(self):
        first_response = self.client.get(reverse("project4:study"))
        first_sample = self.client.session["project4_sample_movie_ids"]
        second_response = self.client.get(reverse("project4:study"))
        second_sample = self.client.session["project4_sample_movie_ids"]

        self.assertEqual(first_response.status_code, 200)
        self.assertEqual(second_response.status_code, 200)
        self.assertEqual(len(first_sample), 20)
        self.assertEqual(first_sample, second_sample)
        self.assertContains(first_response, "Pairwise Selection")
        self.assertContains(first_response, "Ranking Interface")

    def test_report_endpoint_returns_pdf(self):
        response = self.client.get(reverse("project4:report"))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["Content-Type"], "application/pdf")
        self.assertIn("movie_recommender_feature_report.pdf", response["Content-Disposition"])

    def test_pairwise_endpoint_requires_all_five_choices(self):
        self.client.get(reverse("project4:study"))
        sample = self.client.session["project4_sample_movie_ids"]

        response = self.post_json(
            "project4:submit_pairwise",
            {"choices": [{"winner_id": sample[0], "loser_id": sample[1]}]},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("five pairwise", response.json()["error"])

    def test_pairwise_and_ranking_generate_recommendations(self):
        self.client.get(reverse("project4:study"))
        sample = self.client.session["project4_sample_movie_ids"]
        pairwise_choices = [
            {"pair_index": index // 2 + 1, "winner_id": sample[index], "loser_id": sample[index + 1]}
            for index in range(0, 10, 2)
        ]

        pairwise_response = self.post_json("project4:submit_pairwise", {"choices": pairwise_choices})
        ranking_response = self.post_json(
            "project4:submit_ranking",
            {
                "ranking": [{"rank": rank + 1, "movie_id": movie_id} for rank, movie_id in enumerate(sample[10:20])],
                "feedback": {"relevance_rating": "4", "effort_rating": "3"},
            },
        )
        payload = ranking_response.json()

        self.assertEqual(pairwise_response.status_code, 200)
        self.assertEqual(ranking_response.status_code, 200)
        self.assertGreater(len(payload["recommendations"]), 0)
        self.assertTrue({movie["movie_id"] for movie in payload["recommendations"]}.isdisjoint(sample))
        self.assertIn("comparisons_used", payload["summary"])
        self.assertIn("session-local", " ".join(payload["assumptions"]))

    def test_reset_study_clears_project4_session(self):
        self.client.get(reverse("project4:study"))
        self.assertIn("project4_sample_movie_ids", self.client.session)

        response = self.post_json("project4:reset_study", {"reset": True})

        self.assertEqual(response.status_code, 200)
        self.assertNotIn("project4_sample_movie_ids", self.client.session)
