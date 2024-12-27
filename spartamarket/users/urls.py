from django.urls import include, path
from .views import UserSignupView, UserLoginView, UserProfileView
from . import views
from rest_framework_simplejwt.views import (
    TokenObtainPairView,
    TokenRefreshView,
)

app_name = "users"
urlpatterns = [

    path("", views.users, name="users"),
    path("signup/", UserSignupView.as_view(), name="signup"),
    path("login/", UserLoginView.as_view(), name="login"),
    # path('', views.getRoutes),
    path('profile/<int:user_id>/', UserProfileView.as_view(), name='profile'),
    path("<int:user_id>/update", views.update, name="update"),
    # path('password/', views.change_password, name="change_password"),
    # path("<int:user_id>/delete/", views.delete, name="delete"),
    # path("<int:user_id>/follow/", views.follow, name="follow"),
    # path("<int:user_id>/followings/", views.followings, name="followings"),
    # path("<int:user_id>/followers/", views.followers, name="followers"),
]