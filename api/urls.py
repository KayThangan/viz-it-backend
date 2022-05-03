from django.urls import path
from . import views

urlpatterns = [
    path('', views.getRoutes),
    path('image-search/upload/', views.uploadImageSearch),
    path('image-search/result/', views.getImageResult),
    path('image-search/datas/', views.getImageDatas),
]