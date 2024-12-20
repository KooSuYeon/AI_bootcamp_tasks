from django.urls import include, path
from . import views



app_name = "users"
urlpatterns = [
    path("", views.users, name="users"),
    path("signup/", views.signup, name="signup"),
    path("login/", views.login, name="login"),
    path("logout/", views.logout, name="logout"),
    path("<int:user_id>/profile", views.profile, name="profile"),
    path("<int:user_id>/update", views.update, name="update"),
    path('password/', views.change_password, name="change_password"),
    path("<int:user_id>/delete/", views.delete, name="delete"),
    path("<int:user_id>/follow/", views.follow, name="follow"),
    path("<int:user_id>/followings/", views.followings, name="followings"),
    path("<int:user_id>/followers/", views.followers, name="followers"),
]