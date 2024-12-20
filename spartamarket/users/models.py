from django.db import models
from django.conf import settings

# Create your models here.
from django.contrib.auth.models import AbstractUser


# Create your models here.
class User(AbstractUser):

    first_name = None
    last_name = None
    email = models.EmailField(blank=True, null=True)
    nickname = models.CharField(max_length=130, blank=True, null=True)
    location = models.CharField(max_length=255, blank=True, null=True)
    create_at = models.DateTimeField(auto_now_add=True)
    modified_at = models.DateTimeField(auto_now=True)

class Follow(models.Model):

    from_user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE,
        related_name="followings"
    )

    to_user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE,
        related_name="followers"
    )

    created_at = models.DateTimeField(auto_now_add=True)
    modified_at = models.DateTimeField(auto_now=True)
