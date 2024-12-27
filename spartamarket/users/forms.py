from django.contrib.auth.forms import UserChangeForm, UserCreationForm
from django.contrib.auth import get_user_model
from django.urls import reverse

class CustomUserCreateForm(UserCreationForm):
    class Meta:
        model = get_user_model()
        fields = UserCreationForm.Meta.fields + ()

    def save(self, commit=True):
        user = super().save(commit=False)
        # 비밀번호 해싱
        user.set_password(self.cleaned_data["password1"])
        if commit:
            user.save()
        return user

class CustomUserUpdateForm(UserChangeForm):
    class Meta:
        model = get_user_model() 
        fields = [
            "email",
            "nickname",
            "location",
        ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        password = self.fields.get("password")
        if password:
            password_help_text = (
                "You can change the password"
                '<a href="{}">this form</a>.'
            ).format(f"{reverse("users:change_password")}")
            self.fields["password"].help_text = password_help_text