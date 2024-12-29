from django.urls import include, path
from .views import UserSignupView, UserLoginView, UserProfileView, FollowListView, FollowCreateView, OtherProfileView, UserUpdateView, UserDeleteView
from . import views
from rest_framework_simplejwt.views import (
    TokenObtainPairView,
    TokenRefreshView,
)

app_name = "users"
urlpatterns = [

   
    path("signup/", UserSignupView.as_view(), name="signup"),
    path("login/", UserLoginView.as_view(), name="login"),
    path('profile/', UserProfileView.as_view(), name='profile'),
    path('update/profile/', UserUpdateView.as_view(), name='update'),
    path('<int:user_id>/profile/', OtherProfileView.as_view(), name='other_profile'),
    # path('password/', views.change_password, name="change_password"),
    # path("<int:user_id>/delete/", views.delete, name="delete"),
    path("", FollowListView.as_view(), name="users"),
    path("<int:user_id>/follow/",FollowCreateView.as_view(), name="follow"),
     path("<int:user_id>/delete/",UserDeleteView.as_view(), name="delete"),
]