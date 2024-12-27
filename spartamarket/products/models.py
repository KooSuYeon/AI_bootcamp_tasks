from django.db import models
from django.conf import settings

# Create your models here.
class Product(models.Model):

    CATEGORY_CHOICES = [
        ('idea', 'Idea'),
        ('template', 'Template'),
        ('maintenance', 'Maintenance'),
        ('develop', 'Develop'),
        ('other', 'Other'),
    ]

    name = models.CharField(max_length=130)
    description = models.TextField()
    type = models.CharField(max_length=30, choices=CATEGORY_CHOICES, default="idea", help_text="Category of the product")
    price = models.CharField(max_length=10,
    help_text="Price of the product in thousand units (no decimals)",
    default=0)  # 기본값을 0으로 설정
    image = models.ImageField(upload_to="products/", blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    modified_at = models.DateTimeField(auto_now=True)

    author = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="products"
    )

    like_users = models.ManyToManyField(
        settings.AUTH_USER_MODEL, related_name="like_products"
    )
