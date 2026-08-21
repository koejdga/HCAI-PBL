from django.urls import path

from . import views

app_name = "project4"

urlpatterns = [
    path("", views.index, name="index"),
    path("report/", views.report, name="report"),
]
