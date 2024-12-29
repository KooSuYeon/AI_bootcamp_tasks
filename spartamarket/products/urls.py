from django.urls import include, path
from .views import ProductCreateView, ProductDetailView, ProductListView, ProductUpdateView, ProductDeleteView, MyCartListView, ProductSaveView, ProductLikeView, ProductFilterListView


app_name = "products"
urlpatterns = [
    path("", ProductListView.as_view(), name="products"),
    path("filter/", ProductFilterListView.as_view(), name="filter"),
    path("create/", ProductCreateView.as_view(), name="create"),

    path("<int:product_id>/detail/", ProductDetailView.as_view(), name="detail"),
    path("<int:product_id>/update/", ProductUpdateView.as_view(), name="update"),
    path("<int:product_id>/delete/", ProductDeleteView.as_view(), name="delete"),

    path("<int:product_id>/save/", ProductSaveView.as_view(), name="save"),
    path("<int:product_id>/like/", ProductLikeView.as_view(), name="like"),
    path("mycart/", MyCartListView.as_view(), name="mycart"),
]