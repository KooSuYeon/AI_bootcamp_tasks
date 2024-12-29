from rest_framework import serializers
from .models import Product
import django_filters
from django_filters.views import FilterView
from django.db.models import Q

class ProductListSerializer(serializers.ModelSerializer):

    class Meta:
        model = Product
        fields = "__all__"

    def get_image(self, obj):
        request = self.context.get('request')
        if obj.image:
            return request.build_absolute_uri(obj.image.url)
        return None


class ProductDetailSerializer(serializers.ModelSerializer):

    author = serializers.ReadOnlyField(source="author.id")

    class Meta:
        model = Product
        exclude = ['like_users', 'saved_users',]


class ProductFilter(django_filters.FilterSet):
    name = django_filters.CharFilter(field_name='name', lookup_expr='icontains', label="Name")
    type = django_filters.ChoiceFilter(choices=Product.CATEGORY_CHOICES, label="Category")
    min_price = django_filters.NumberFilter(field_name='price', lookup_expr='gte', label="Min Price")
    max_price = django_filters.NumberFilter(field_name='price', lookup_expr='lte', label="Max Price")

    class Meta:
        model = Product
        fields = ['name', 'type', 'min_price', 'max_price']