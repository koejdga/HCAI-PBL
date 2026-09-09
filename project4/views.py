import json
import os
import random
from functools import lru_cache

from django.conf import settings
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import ensure_csrf_cookie
from django.views.decorators.http import require_POST
import pandas as pd

from project4.feature_representation import build_movie_feature_matrix
from project4.preference_learning import learn_and_recommend
from pbl.pdf_generator import render_to_pdf


SAMPLE_SESSION_KEY = "project4_sample_movie_ids"
PAIRWISE_SESSION_KEY = "project4_pairwise_choices"
RANKING_SESSION_KEY = "project4_ranking"
FEEDBACK_SESSION_KEY = "project4_feedback"
TIMING_SESSION_KEY = "project4_timing"
PROJECT4_SESSION_KEYS = [
    SAMPLE_SESSION_KEY,
    PAIRWISE_SESSION_KEY,
    RANKING_SESSION_KEY,
    FEEDBACK_SESSION_KEY,
    TIMING_SESSION_KEY,
]


def landing(request):
    content_menu_items = [
        {"label": "Overview", "href": "#project4-overview"},
        {"label": "Study Protocol", "href": "#study-protocol"},
        {"label": "Transparency Notes", "href": "#hcai-notes"},
    ]
    context = {
        "content_menu_items": content_menu_items,
        "is_study": False,
        "feature_summary": get_feature_summary(),
    }
    return render(request, "project4/index.html", context)


@ensure_csrf_cookie
def study(request):
    matrix = get_movie_matrix()
    sampled_ids = get_or_create_sample_ids(request, matrix)
    movies_by_id = {movie["movie_id"]: movie for movie in matrix.movies}
    sampled_movies = [movies_by_id[movie_id] for movie_id in sampled_ids if movie_id in movies_by_id]

    design1_movies = sampled_movies[:10]
    design2_movies = sampled_movies[10:20]
    design1_pairs = [
        {
            "pair_index": index // 2 + 1,
            "movie1": design1_movies[index],
            "movie2": design1_movies[index + 1],
        }
        for index in range(0, len(design1_movies), 2)
    ]

    content_menu_items = [
        {"label": "Consent", "href": "#participant-consent"},
        {"label": "Pairwise Selection", "href": "#design-1-section"},
        {"label": "Ranking Interface", "href": "#design-2-section"},
        {"label": "Recommendations", "href": "#recommendation-results"},
        {"label": "Report", "href": "#report-download"},
    ]

    context = {
        "content_menu_items": content_menu_items,
        "design1_pairs": design1_pairs,
        "design2_movies": design2_movies,
        "feature_summary": get_feature_summary(matrix),
        "is_study": True,
    }
    return render(request, "project4/index.html", context)


@require_POST
def submit_pairwise(request):
    payload = read_json_payload(request)
    matrix = get_movie_matrix()
    sampled_ids = get_or_create_sample_ids(request, matrix)
    valid_ids = set(sampled_ids[:10])

    choices = []
    for choice in payload.get("choices", []):
        winner_id = clean_submitted_id(choice.get("winner_id") or choice.get("winner_title"))
        loser_id = clean_submitted_id(choice.get("loser_id") or choice.get("loser_title"))
        if winner_id not in valid_ids or loser_id not in valid_ids or winner_id == loser_id:
            return JsonResponse({"error": "Pairwise choices must use movies from the current pairwise task."}, status=400)
        choices.append(
            {
                "pair_index": choice.get("pair_index"),
                "winner_id": winner_id,
                "loser_id": loser_id,
            }
        )

    if len(choices) != 5:
        return JsonResponse({"error": "Please complete all five pairwise comparisons before submitting."}, status=400)

    request.session[PAIRWISE_SESSION_KEY] = choices
    save_design_metadata(request, "pairwise", payload)
    request.session.modified = True

    response = {"message": "Pairwise choices saved for this browser session."}
    response.update(build_recommendation_payload(request, matrix))
    return JsonResponse(response)


@require_POST
def submit_ranking(request):
    payload = read_json_payload(request)
    matrix = get_movie_matrix()
    sampled_ids = get_or_create_sample_ids(request, matrix)
    expected_ids = set(sampled_ids[10:20])

    ranking_payload = payload.get("ranking", [])
    ranking = []
    for item in ranking_payload:
        if isinstance(item, dict):
            movie_id = clean_submitted_id(item.get("movie_id") or item.get("title"))
        else:
            movie_id = clean_submitted_id(item)
        if movie_id:
            ranking.append(movie_id)

    if len(ranking) != 10 or set(ranking) != expected_ids:
        return JsonResponse({"error": "Ranking must contain the ten movies from the current ranking task."}, status=400)

    request.session[RANKING_SESSION_KEY] = ranking
    save_design_metadata(request, "ranking", payload)
    request.session.modified = True

    response = {"message": "Ranking saved for this browser session."}
    response.update(build_recommendation_payload(request, matrix))
    return JsonResponse(response)


@require_POST
def recommendations(request):
    matrix = get_movie_matrix()
    return JsonResponse(build_recommendation_payload(request, matrix))


@require_POST
def reset_study(request):
    for key in PROJECT4_SESSION_KEYS:
        request.session.pop(key, None)
    request.session.modified = True
    return JsonResponse({"message": "Project 4 study session reset."})


@lru_cache(maxsize=1)
def get_movie_matrix():
    csv_path = os.path.join(settings.BASE_DIR, "project4", "movie_metadata.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")
    return build_movie_feature_matrix(pd.read_csv(csv_path))


def get_feature_summary(matrix=None):
    if matrix is None:
        matrix = get_movie_matrix()
    return {
        "movie_count": len(matrix.movies),
        "feature_count": len(matrix.feature_columns),
        "assumptions": matrix.assumptions,
        "warnings": matrix.warnings,
    }


def get_or_create_sample_ids(request, matrix):
    available_ids = [movie["movie_id"] for movie in matrix.movies]
    available_set = set(available_ids)
    sampled_ids = request.session.get(SAMPLE_SESSION_KEY)
    if (
        isinstance(sampled_ids, list)
        and len(sampled_ids) == 20
        and set(sampled_ids).issubset(available_set)
    ):
        return sampled_ids

    sampled_ids = random.sample(available_ids, 20)
    request.session[SAMPLE_SESSION_KEY] = sampled_ids
    request.session.pop(PAIRWISE_SESSION_KEY, None)
    request.session.pop(RANKING_SESSION_KEY, None)
    request.session.pop(FEEDBACK_SESSION_KEY, None)
    request.session.pop(TIMING_SESSION_KEY, None)
    request.session.modified = True
    return sampled_ids


def read_json_payload(request):
    try:
        return json.loads(request.body.decode("utf-8") or "{}")
    except json.JSONDecodeError:
        return {}


def clean_submitted_id(value):
    return str(value or "").strip()


def save_design_metadata(request, design, payload):
    timing = sanitize_timing(payload)
    feedback_by_design = sanitize_feedback(payload, design)

    if timing:
        timings = dict(request.session.get(TIMING_SESSION_KEY, {}))
        timings[design] = timing
        request.session[TIMING_SESSION_KEY] = timings

    if feedback_by_design:
        existing_feedback = dict(request.session.get(FEEDBACK_SESSION_KEY, {}))
        for key, value in feedback_by_design.items():
            existing_feedback[key] = value
        request.session[FEEDBACK_SESSION_KEY] = existing_feedback


def sanitize_timing(payload):
    raw_timing = payload.get("timing", {})
    if not isinstance(raw_timing, dict):
        raw_timing = {}

    duration = (
        raw_timing.get("duration_seconds")
        or payload.get("duration_seconds")
        or raw_timing.get("task_duration_seconds")
    )
    if duration is None:
        duration_ms = raw_timing.get("duration_ms") or payload.get("duration_ms") or raw_timing.get("task_duration_ms")
        if duration_ms is not None:
            try:
                duration = float(duration_ms) / 1000
            except (TypeError, ValueError):
                duration = None

    timing = {}
    if duration is not None:
        try:
            timing["duration_seconds"] = round(max(float(duration), 0.0), 2)
        except (TypeError, ValueError):
            pass

    for key in ["started_at", "submitted_at"]:
        value = raw_timing.get(key) or payload.get(key)
        if isinstance(value, str) and value.strip():
            timing[key] = value.strip()[:64]

    return timing


def sanitize_feedback(payload, design):
    raw_feedback = payload.get("feedback", {})
    if not isinstance(raw_feedback, dict):
        return {}

    if any(isinstance(raw_feedback.get(key), dict) for key in ["pairwise", "ranking"]):
        feedback_by_design = {}
        for feedback_design in ["pairwise", "ranking"]:
            if isinstance(raw_feedback.get(feedback_design), dict):
                feedback_by_design[feedback_design] = sanitize_feedback_values(raw_feedback[feedback_design])
        overall_feedback = {
            key: value
            for key, value in raw_feedback.items()
            if key not in ["pairwise", "ranking"]
        }
        if overall_feedback:
            feedback_by_design["overall"] = sanitize_feedback_values(overall_feedback)
        return {key: value for key, value in feedback_by_design.items() if value}

    feedback = sanitize_feedback_values(raw_feedback)
    return {design: feedback} if feedback else {}


def sanitize_feedback_values(raw_feedback):
    feedback = {}
    for key, value in raw_feedback.items():
        if isinstance(value, (str, int, float, bool)):
            feedback[str(key)[:64]] = str(value).strip()[:500]
    return {key: value for key, value in feedback.items() if value != ""}


def build_recommendation_payload(request, matrix):
    pairwise_choices = request.session.get(PAIRWISE_SESSION_KEY, [])
    ranking = request.session.get(RANKING_SESSION_KEY, [])
    sampled_ids = request.session.get(SAMPLE_SESSION_KEY, [])
    if not pairwise_choices and not ranking:
        return {
            "recommendations": [],
            "recommendation_groups": {},
            "summary": {"model_note": "No preferences submitted yet."},
            "assumptions": [],
            "telemetry": build_telemetry_payload(request),
        }

    groups = {}
    if pairwise_choices:
        groups["pairwise"] = build_recommendation_group(
            matrix,
            pairwise_choices=pairwise_choices,
            rankings=None,
            shown_movie_ids=sampled_ids,
        )
    if ranking:
        groups["ranking"] = build_recommendation_group(
            matrix,
            pairwise_choices=None,
            rankings=ranking,
            shown_movie_ids=sampled_ids,
        )

    groups["combined"] = build_recommendation_group(
        matrix,
        pairwise_choices=pairwise_choices,
        rankings=ranking,
        shown_movie_ids=sampled_ids,
    )
    combined = groups["combined"]

    return {
        "recommendations": combined["recommendations"],
        "recommendation_groups": groups,
        "summary": combined["summary"],
        "assumptions": combined["assumptions"],
        "warnings": combined["warnings"],
        "telemetry": build_telemetry_payload(request),
    }


def build_recommendation_group(matrix, pairwise_choices=None, rankings=None, shown_movie_ids=None):
    result = learn_and_recommend(
        feature_table=matrix.features,
        pairwise_choices=pairwise_choices,
        rankings=rankings,
        shown_movie_ids=shown_movie_ids,
        top_n=8,
        feature_columns=matrix.feature_columns,
        feature_labels=matrix.feature_labels,
    )
    movie_lookup = {movie["movie_id"]: movie for movie in matrix.movies}
    recommendations = []
    for item in result.recommendations:
        movie = dict(movie_lookup.get(item["movie_id"], {"title": item["title"]}))
        movie["score"] = item["utility_score"]
        movie["explanations"] = [
            explanation["label"] if isinstance(explanation, dict) else str(explanation)
            for explanation in item.get("explanations", [])
        ]
        recommendations.append(movie)

    return {
        "recommendations": recommendations,
        "summary": {
            "comparisons_used": result.fit.comparisons_used,
            "source_counts": result.fit.source_counts,
            "model_type": result.fit.model_type,
        },
        "assumptions": result.fit.assumptions + result.assumptions,
        "warnings": result.fit.warnings + result.warnings,
    }


def build_telemetry_payload(request):
    return {
        "timing": request.session.get(TIMING_SESSION_KEY, {}),
        "feedback": request.session.get(FEEDBACK_SESSION_KEY, {}),
        "storage": "session-local",
    }


def report(request):
    return render_to_pdf(
        template_src="project4/report_pdf.html",
        context_dict={},
        filename="movie_recommender_feature_report.pdf",
    )
