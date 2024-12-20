from django.urls import include, path
from . import views


app_name = "products"
urlpatterns = [
    path("", views.products, name="products"),
    path("create/", views.create, name="create"),
    path("<int:pk>/delete/", views.delete, name="delete"),
    path("<int:pk>/detail/", views.detail, name="detail"),
]