from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),  # Example: Home page
    path('predict/', views.predict, name='predict'),  # Example: Prediction page
    path('clustering/', views.clustering_view, name='clustering'),
    path('pca/', views.pca_view, name='pca'),
    path('chi-square', views.chi_square_test_view, name='chi-square'),
    path('factor analysis', views.factor_analysis_view, name='fa')# Another prediction-related page
    # Add other URL patterns as needed
]
