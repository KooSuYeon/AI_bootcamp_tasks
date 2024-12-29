from django.contrib.auth.models import User
from django.contrib.auth.password_validation import validate_password
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer
from rest_framework import serializers
from rest_framework.serializers import Serializer, CharField
from .models import User, Follow
from django.contrib.auth import authenticate


class SignupSerializer(serializers.ModelSerializer):
     
    class Meta:
        model = User
        fields = "__all__"

    def create(self, validated_data):
        user = User.objects.create_user(
            username = validated_data['username'],
            password = validated_data['password']
        )
        return user

class LoginSerializer(Serializer):
    username = CharField()
    password = CharField()

    def validate(self, data):
        user = authenticate(username=data["username"], password=data["password"])

        if not user:
            raise serializers.ValidationError("아이디 또는 비밀번호가 잘못되었습니다.")
        return {'user': user}

class FollowingSerializer(serializers.ModelSerializer):

    class Meta:
        model = Follow
        fields = ("from_user", "to_user", "created_at", "modified_at")

class UserProfileSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = "__all__"