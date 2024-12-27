from django.http import HttpResponseRedirect
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
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer, TokenRefreshSerializer
from django.contrib.auth.hashers import check_password
from .models import User
from .serializers import LoginSerializer, SignupSerializer, UserProfileSerializer
import jwt
from django.contrib.auth import authenticate
from django.shortcuts import render, get_object_or_404
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.decorators import api_view

import hashlib
from rest_framework_simplejwt.views import (
    TokenObtainPairView,
    TokenRefreshView,
)
from rest_framework.response import Response

SECRET_KEY = "823e399822c5170927c9802b3feb60b1fe54debefb406ca5f4eaf05e0014ea63"
# Create your views here.
def users(request):

    me = request.user
    if request.user.is_authenticated:
        users = User.objects.all()
        followings = Follow.objects.filter(from_user=request.user).values_list('to_user_id', flat=True)
        followers = Follow.objects.filter(to_user=request.user).values_list('from_user_id', flat=True)
        not_following = users.exclude(id__in=followings).exclude(id=request.user.id)
        context = {
            'users': users,
            'not_followings': not_following,
            'followings': followings,
            'followers': followers,
        }
        return render(request, 'users/users.html', context)
    else:
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
            return HttpResponseRedirect(reverse('users:users')) 
        else:
            # form이 유효하지 않으면, 오류 메시지를 포함한 form을 반환
            json_response = {
                "form": form,
                "errors": form.errors  # form의 오류 메시지를 함께 반환
            }
            return Response(json_response)
        


class UserLoginView(APIView):
    renderer_classes = [TemplateHTMLRenderer]
    template_name = "users/login.html"

    def get(self, request):
        # 쿠키에서 access token을 가져오기
        access_token = request.COOKIES.get('access', None)

        print("ACCESS_TOKEN", access_token)
        if access_token:
            try:
                # access token을 decode하여 user_id 추출
                payload = jwt.decode(access_token, SECRET_KEY, algorithms=['HS256'])
                user_id = payload.get('user_id')

                if user_id is None:
                    raise jwt.exceptions.InvalidTokenError("user_id가 없음")

                # 유효한 사용자라면 profile로 리디렉션
                user = get_user_model().objects.get(pk=user_id)
                return HttpResponseRedirect(reverse('users:profile')) 
                context = {
                    "user": user,
                    "user_id": user_id,  # user_id를 추가로 전달
                }

                # profile.html 템플릿에 사용자 정보 전달
                return render(request, "users/profile.html", context)

            except jwt.ExpiredSignatureError:
                return Response({"error": "Access token이 만료되었습니다."}, status=status.HTTP_401_UNAUTHORIZED)
            except jwt.exceptions.InvalidTokenError as e:
                return Response({"error": f"Invalid token: {str(e)}"}, status=status.HTTP_401_UNAUTHORIZED)

        # access_token이 없거나 만료된 경우, 로그인 화면을 띄움
        form = LoginSerializer()
        return Response({"form": form})


    def post(self, request):
        # Serializer에 요청 데이터를 전달하여 유효성 검사를 수행
        form = LoginSerializer(data=request.POST)

        if form.is_valid():
            user = form.validated_data['user']
            if user is not None:
                print("인증된 사용자:", user)
                token = TokenObtainPairSerializer.get_token(user)
                refresh_token = str(token)
                access_token = str(token.access_token)
                res = Response(
                    {
                        "user": user,
                        "message": "login success",
                        "token": {
                            "access": access_token,
                            "refresh": refresh_token,
                        },
                    },
                    status=status.HTTP_200_OK,
                )
                # jwt 토큰 => 쿠키에 저장
                res.set_cookie("access", access_token, httponly=True)
                res.set_cookie("refresh", refresh_token, httponly=True)
                
                return res

            else:
                form = LoginSerializer()  # 빈 폼을 다시 반환
                return Response({"form": form, "error": "아이디 또는 비밀번호가 잘못되었습니다."}, status=400)
        else:
            print("폼 유효하지 않음:", form.errors)
            return Response({"form": form, "error": "입력된 데이터가 유효하지 않습니다."}, status=400)
        
    def delete(self, request):
        # 쿠키에 저장된 토큰 삭제 => 로그아웃 처리
        response = Response({
            "message": "Logout success"
            }, status=status.HTTP_202_ACCEPTED)
        response.delete_cookie("access")
        response.delete_cookie("refresh")
        return response
        
    	
@require_http_methods(["GET", "POST"])
def signup(request):
    
    if request.method == "POST":
        form = CustomUserCreateForm(request.POST)
        if form.is_valid():
            user = form.save()
            auth_login(request, user)
            return redirect("users:users")
        
    else:
        form = CustomUserCreateForm()

    context = {"form": form}
    return render(request, "users/signup.html", context)

    
@require_http_methods(["GET", "POST"])
def login(request):
    if request.method == "POST":
        form = AuthenticationForm(data = request.POST)
        if form.is_valid():
            auth_login(request, form.get_user())
            next_path = request.GET.get("next") or "users:users"
            return redirect(next_path)
    else:
        form = AuthenticationForm()

    context = {"form": form}
    return render(request, "users/login.html", context)

@require_POST
def logout(request):
    auth_logout(request)
    return redirect("users:users")


class UserProfileView(APIView):
    """
    로그인된 사용자 정보를 반환하는 API
    유효한 JWT를 통해 인증된 사용자 정보를 반환.
    """
    renderer_classes = [TemplateHTMLRenderer]
    template_name = "users/profile.html"

    def get(self, request, user_id):
        access_token = request.COOKIES.get('access', None)

        print(">>>>>>>>", access_token)
        if not access_token:
            return redirect('users:login')  # 로그인 페이지로 리디렉션

        try:
            # access_token을 디코드하여 사용자 정보 추출
            payload = jwt.decode(access_token, SECRET_KEY, algorithms=['HS256'])
            token_user_id = payload.get('user_id')

            if token_user_id != user_id:
                raise jwt.exceptions.InvalidTokenError("user_id가 일치하지 않음")

            # user 객체 가져오기
            user = get_user_model().objects.get(pk=user_id)
            serializer = UserProfileSerializer(user, context={'request': request})
            return Response(serializer.data, status=200)

        except jwt.ExpiredSignatureError:
            return redirect('users:login')  # 토큰이 만료되었을 때 로그인 페이지로 리디렉션
        except jwt.exceptions.InvalidTokenError:
            return redirect('users:login')  # 유효하지 않은 토큰일 때 로그인 페이지로 리디렉션



@require_http_methods(["POST", "GET"])
def update(request, user_id):

    if request.method == "POST":
        form = CustomUserUpdateForm(request.POST, instance=request.user)
        if form.is_valid():
            form.save()
            return redirect("index")
    else:
        form = CustomUserUpdateForm(instance=request.user)
    context = {"form": form}
    return render(request, "users/update.html", context)

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

@require_POST
def delete(request, user_id):
    if request.user.is_authenticated:
        request.user.delete()
        auth_logout(request)
    return redirect("index")

@require_POST
def follow(request, user_id):
    if request.user.is_authenticated:

        to_user = get_object_or_404(get_user_model(), pk=user_id)
        from_user = get_object_or_404(get_user_model(), pk=request.user.id)

        following = Follow.objects.filter(from_user=from_user, to_user=to_user)
        
        if following:
            following[0].delete()
        else:
            Follow.objects.create(
                from_user = from_user,
                to_user   = to_user, 
            )
        
        return redirect("users:users")
    

def followers(request, pk):

    user = get_object_or_404(get_user_model(), pk=pk)
    followers = Follow.objects.filter(to_user=user)

    context = {
        "followers": followers
    }

    return render(request, "users:followers", context)


def followings(request, pk):

    user = get_object_or_404(get_user_model(), pk=pk)
    followings = Follow.objects.filter(from_user=user)

    context = {
        "followings": followings
    }

    return render(request, "users:followings", context)
                