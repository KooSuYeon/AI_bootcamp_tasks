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

    name = models.CharField(max_length=130, blank=True, null=True)
    description = models.TextField()
    type = models.CharField(max_length=30, choices=CATEGORY_CHOICES, blank=True, null=True, help_text="Category of the product")
    price = models.PositiveIntegerField(blank=True, null=True, help_text="Price of the product in thousand units (no decimals)")
    image = models.ImageField(upload_to="products/", blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    modified_at = models.DateTimeField(auto_now=True)

    author = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="products"
    )

    like_users = models.ManyToManyField(
        settings.AUTH_USER_MODEL, related_name="like_products"
    )
