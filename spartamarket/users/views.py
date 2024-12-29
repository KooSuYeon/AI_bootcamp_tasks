from django.http import HttpResponseRedirect, HttpResponse, HttpResponseNotAllowed
from django.shortcuts import render, redirect, get_object_or_404
from django.views.decorators.http import require_POST, require_http_methods
from django.urls import reverse
from django.contrib.auth.forms import AuthenticationForm, PasswordChangeForm

from .forms import CustomUserCreateForm, CustomUserUpdateForm
from django.contrib.auth import login as auth_login
from django.contrib.auth import logout as auth_logout
from django.contrib.auth import update_session_auth_hash
from django.contrib.auth import get_user_model
from rest_framework.views import APIView
from .models import User, Follow
from .forms import CustomUserCreateForm

from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.renderers import TemplateHTMLRenderer
from rest_framework.response import Response
from rest_framework import status, generics
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer
from django.contrib.auth.hashers import check_password
from .models import User
from .serializers import LoginSerializer, UserProfileSerializer
import jwt
from django.contrib.auth import authenticate
from django.shortcuts import render, get_object_or_404
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.decorators import api_view
import boto3
from botocore.exceptions import BotoCoreError, NoCredentialsError
from django.conf import settings
from django.http import JsonResponse

import hashlib
from rest_framework_simplejwt.views import (
    TokenObtainPairView,
    TokenRefreshView,
)
from rest_framework.response import Response
from dotenv import load_dotenv
import os

# .env 파일 로드
load_dotenv()

# SECRET_KEY 환경변수에서 가져오기
SECRET_KEY = os.getenv('SECRET_KEY')

def get_user_id(request):
    access_token = request.COOKIES.get('access', None)
    user_id = None
    if access_token:
        try:
            # access token을 decode하여 user_id 추출
            payload = jwt.decode(access_token, SECRET_KEY, algorithms=['HS256'])
            user_id = payload.get('user_id')
            return user_id
        except jwt.ExpiredSignatureError:
            print("Token has expired.")
            return redirect('users:login')
        except jwt.InvalidTokenError:
            print("Invalid token.")
            return redirect('users:login')


class UserSignupView(APIView):
    renderer_classes = [TemplateHTMLRenderer]
    template_name = "users/signup.html"

    def get(self, request):
        form = CustomUserCreateForm
        json_response = { "form": form }
        return Response(json_response)
    
    def post(sel, request):
        form = CustomUserCreateForm(request.POST)
        if form.is_valid():
            form.save()
            return HttpResponseRedirect(reverse('users:login')) 
        else:
            json_response = {
                "form": form,
                "errors": form.errors  
            }
            return Response(json_response)
        
class UserLoginView(APIView):
    renderer_classes = [TemplateHTMLRenderer]
    template_name = "users/login.html"
    profile_template_name = "users/profile.html"

    def get(self, request):
        user_id = get_user_id(request)
        if user_id:
            user = get_user_model().objects.get(pk=user_id)
            return HttpResponseRedirect(reverse('users:profile'))
        else:
            # access_token이 없거나 만료된 경우, 로그인 화면을 띄움
            form = LoginSerializer()
            return Response({"form": form})


    def post(self, request):
        user_id = get_user_id(request)
        if user_id:
            user = get_user_model().objects.get(pk=user_id)
            return HttpResponseRedirect(reverse('users:profile'))
        
        form = LoginSerializer(data=request.POST)

        if form.is_valid():
            user = form.validated_data['user']

            if user is not None:
                token = TokenObtainPairSerializer.get_token(user)
                refresh_token = str(token)
                access_token = str(token.access_token)
                response = HttpResponseRedirect(reverse('users:profile'))
                # jwt 토큰 => 쿠키에 저장
                response.set_cookie("access", access_token, httponly=True)
                response.set_cookie("refresh", refresh_token, httponly=True)
                
                return response
            else:
                form = LoginSerializer()  # 빈 폼을 다시 반환
                return Response({"form": form, "error": "아이디 또는 비밀번호가 잘못되었습니다."}, status=401)
        else:
            print("폼 유효하지 않음:", form.errors)
            return Response({"form": form, "error": "이디 또는 비밀번호가 잘못되었습니다."}, status=401)
        

    def delete(self, request):
        response = Response({
            "message": "Logout success"
            }, status=status.HTTP_202_ACCEPTED)
        response.delete_cookie("access")
        response.delete_cookie("refresh")
        return response
        
    	
class UserProfileView(APIView):
    """
    로그인된 사용자 정보를 반환하는 API
    유효한 JWT를 통해 인증된 사용자 정보를 반환.
    """
    renderer_classes = [TemplateHTMLRenderer]
    template_name = "users/profile.html"
    

    def get(self, request):

        try:
            user_id = get_user_id(request)
            user = get_user_model().objects.get(pk=user_id)
            return render(request, self.template_name, {"user": user})
        
        except get_user_model().DoesNotExist:
            # 예외 발생 시 users:login으로 리다이렉트
            return redirect('users:login')




class UserUpdateView(APIView):
    template_name = "users/put_profile.html"

    def get(self, request):
        user_id = get_user_id(request)
        user = get_user_model().objects.get(pk=user_id)
        return render(request, self.template_name, {"user": user})

    def post(self, request):
        if request.POST.get('_method') == 'PUT':
            return self.put(request)

        return HttpResponseNotAllowed(['PUT'])
    
    def put(self, request):
        user_id = get_user_id(request)
        user = get_user_model().objects.get(pk=user_id)

        data = request.data
        profile_image = request.FILES.get('profile_image')  # Get the uploaded image

        if profile_image:
            try:
                # S3 버킷에 이미지 업로드
                s3 = boto3.client(
                    's3',
                    aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                    aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                )
                bucket_name = settings.AWS_STORAGE_BUCKET_NAME
                s3_file_name = f'users/{profile_image.name}'
                s3.upload_fileobj(profile_image, bucket_name, s3_file_name)

                # S3에 업로드된 파일의 URL 가져오기
                profile_image_url = f"https://{bucket_name}.s3.{settings.AWS_DEFAULT_REGION}.amazonaws.com/{s3_file_name}"

                # 사용자 모델의 프로필 이미지 URL 업데이트
                user.image = profile_image_url

            except (BotoCoreError, NoCredentialsError) as e:
                return JsonResponse({"error": f"Error uploading to S3: {str(e)}"}, status=500)

        serializer = UserProfileSerializer(user, data=data, partial=True)
        if serializer.is_valid(raise_exception=True):
            serializer.save()
            return redirect("users:profile")
        return render(request, self.template_name, {"user": user, "form": serializer.errors})


class UserDeleteView(APIView):
    def post(self, request, *args, **kwargs):
        if request.POST.get('_method') == 'DELETE':
            return self.delete(request, *args, **kwargs)
        return HttpResponseNotAllowed(['POST'])

    def delete(self, request, *args, **kwargs):
        user_id = self.kwargs.get("user_id") 
        user = get_object_or_404(User, pk=user_id)
        user.delete()

        response = HttpResponseRedirect(reverse("users:login"))  # 인덱스로 리다이렉트 (여기서 '/'는 'index' 페이지)
        response.delete_cookie("access")
        response.delete_cookie("refresh")

        return response
    


class OtherProfileView(APIView):
    renderer_classes = [TemplateHTMLRenderer]
    template_name = "users/other_profile.html"

    def get(self, request, *args, **kwargs):  # *args와 **kwargs를 추가
        user_id = self.kwargs.get("user_id")  # kwargs에서 user_id 가져오기
        user = get_object_or_404(get_user_model(), pk=user_id)  # 안전한 조회를 위해 get_object_or_404 사용
        return render(request, self.template_name, {"user": user})



class FollowListView(APIView):

    def get(self, request):

        try:
            users = User.objects.all()

            user_id = get_user_id(request)
            user = get_user_model().objects.get(pk=user_id)
            followings = Follow.objects.filter(from_user=user).select_related('to_user')
            real_followings = [follow.to_user for follow in followings]  

            followers = Follow.objects.filter(to_user=user).select_related('from_user')
            real_followers = [follow.from_user for follow in followers] 

            not_following = users.exclude(id__in=[user.id for user in real_followings]).exclude(id=user_id)

            me = user

            context = {
                'users': users,
                'me': me,
                'not_followings': not_following,
                'followings': real_followings,
                'followers': real_followers,

            }
            
            return render(request, "users/users.html", context)
        
        except get_user_model().DoesNotExist:
            # 예외 발생 시 users:login으로 리다이렉트
            return redirect('users:login')
        
        



class FollowCreateView(APIView):
    
    def post(self, request, *args, **kwargs):
        user_id = self.kwargs.get("user_id") 
        to_user = get_object_or_404(get_user_model(), pk=user_id)
        from_user = get_object_or_404(get_user_model(), pk=get_user_id(request))
        following = Follow.objects.filter(from_user=from_user, to_user=to_user)
        if following.exists():
            following.delete()
        else:
            Follow.objects.create(from_user=from_user, to_user=to_user)
        return redirect("users:users")


     

@require_http_methods(["POST", "GET"])
def change_password(request):
    if request.method == "POST":
        form = PasswordChangeForm(request.user, request.POST)
        if form.is_valid():
            form.save()
            update_session_auth_hash(request, form.user)
            return redirect("index")
    else:
        form = PasswordChangeForm(request.user)
    context = {"form": form}

    return render(request, "users/change_password.html", context)

                