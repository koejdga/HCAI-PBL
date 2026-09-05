from django.urls import path

from . import views

app_name = "project4"

urlpatterns = [
    path("", views.landing, name="index"),
    path("study/", views.study, name="study"),
    path("api/pairwise/", views.submit_pairwise, name="submit_pairwise"),
    path("api/ranking/", views.submit_ranking, name="submit_ranking"),
    path("api/recommendations/", views.recommendations, name="recommendations"),
    path("api/reset/", views.reset_study, name="reset_study"),
    path("report/", views.report, name="report"),
]
