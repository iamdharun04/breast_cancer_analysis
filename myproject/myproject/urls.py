from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('', include('myapp.urls')),  # Include your app's URLs
]
