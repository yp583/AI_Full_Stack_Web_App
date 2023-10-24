from django.urls import path
from . import views

urlpatterns = [
    path("nothome/", views.home, name="nothome"),
    path("", views.websocket, name="websocket")

]