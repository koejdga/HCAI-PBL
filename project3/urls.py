from django.urls import path

from . import views

app_name = "project3"

urlpatterns = [
    path("", views.index, name="index"),
    path("study/", views.study, name="study"),
    path("report/", views.report, name="report"),
]
