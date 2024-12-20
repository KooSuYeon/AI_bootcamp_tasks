from django.shortcuts import render, redirect, get_object_or_404
from django.views.decorators.http import require_POST, require_http_methods
from django.contrib.auth.forms import AuthenticationForm, PasswordChangeForm
from .forms import CustomUserCreateForm, CustomUserUpdateForm
from django.contrib.auth import login as auth_login
from django.contrib.auth import logout as auth_logout
from django.contrib.auth import update_session_auth_hash
from django.contrib.auth import get_user_model
from .models import User, Follow

# Create your views here.
def users(request):

    if request.user.is_authenticated:
        users = User.objects.all()
        followings = Follow.objects.filter(from_user=request.user).values_list('to_user_id', flat=True)
        followers = Follow.objects.filter(to_user=request.user).values_list('from_user_id', flat=True)
        context = {
            'users': users,
            'followings': followings,
            'followers': followers,
        }
        return render(request, 'users/users.html', context)
    else:
        return redirect('users:login')


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


def profile(request, user_id):
    user = get_object_or_404(User, pk=user_id)

    context = {
        "user" : user
    }

    return render(request, "users/profile.html", context)

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
                