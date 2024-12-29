from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from .forms import ProductForm
from .models import Product
from .serializers import ProductListSerializer, ProductDetailSerializer, ProductFilter
from django.views.decorators.http import require_POST
from rest_framework.views import APIView
from django.contrib.auth import get_user_model
from rest_framework.renderers import TemplateHTMLRenderer
import jwt
from rest_framework.response import Response
import boto3
from botocore.exceptions import BotoCoreError, NoCredentialsError
from django.conf import settings
from django.http import HttpResponseBadRequest, HttpResponseNotAllowed, JsonResponse
from django_filters.views import FilterView
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

class ProductCreateView(APIView):
    renderer_classes = [TemplateHTMLRenderer]
    template_name = "products/create.html"
    
    def get(self, request):
        
        serializer = ProductListSerializer()

        context = {
            "form": serializer
        }
        return render(request, self.template_name, context)
    
    def post(self, request):
        user_id = get_user_id(request)
        user = get_user_model().objects.get(pk=user_id)

        profile_image = request.FILES.get('image')  # Get the uploaded image

        serializer = ProductDetailSerializer(data=request.data)
        if serializer.is_valid():
            # 예비 저장 (commit=False)으로 Product 객체 생성
            product = Product(**serializer.validated_data)
            product.author = user  # author 필드 추가

            if profile_image:
                try:
                    # S3 버킷에 이미지 업로드
                    s3 = boto3.client(
                        's3',
                        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                    )
                    bucket_name = settings.AWS_STORAGE_BUCKET_NAME
                    s3_file_name = f'products/{profile_image.name}'
                    s3.upload_fileobj(profile_image, bucket_name, s3_file_name)

                    # S3에 업로드된 파일의 URL 가져오기
                    profile_image_url = f"https://{bucket_name}.s3.{settings.AWS_DEFAULT_REGION}.amazonaws.com/{s3_file_name}"

                    # Product 객체의 이미지 필드 업데이트
                    product.image = profile_image_url
                    print(product.image)

                except (BotoCoreError, NoCredentialsError) as e:
                    return JsonResponse({"error": f"Error uploading to S3: {str(e)}"}, status=500)

            # 최종 저장
            product.save()
            return redirect("products:products")
        else:
            print(serializer.errors)  # 에러 출력
            return Response(serializer.errors, status=400)



class ProductListView(APIView):
    renderer_classes = [TemplateHTMLRenderer]
    template_name = "products/shop.html"

    def get(self, request):
        products = Product.objects.all()
        context = {"products": products}  # 직렬화된 데이터 대신 모델 객체 전달
        return render(request, self.template_name, context)

class ProductFilterListView(APIView):
    template_name = "products/shop.html"

    def get(self, request):
        # 필터링을 위한 쿼리셋을 불러옵니다.
        queryset = Product.objects.all()

        # 필터셋 적용
        filterset = ProductFilter(request.GET, queryset=queryset)

        # 필터링된 결과
        products = filterset.qs

        context = {
            'products': products,
            'filter': filterset,
        }

        # 필터링된 데이터를 렌더링하여 반환
        return render(request, self.template_name, context)

    

class ProductDetailView(APIView):

    renderer_classes = [TemplateHTMLRenderer]
    template_name = "products/detail.html"

    def get(self, request, *args, **kwargs):  # *args와 **kwargs를 추가
        product_id = self.kwargs.get("product_id")  # kwargs에서 user_id 가져오기
        product = get_object_or_404(Product, pk=product_id)  # 안전한 조회를 위해 get_object_or_404 사용
        user_id = get_user_id(request)
        user = get_user_model().objects.get(pk=user_id)

        is_liked = product.like_users.filter(id=user.id).exists()

        context = {
            "product": product,
            "user": user,
            "is_liked": is_liked,
        }
        return render(request, self.template_name, context)



class ProductUpdateView(APIView):
    renderer_classes = [TemplateHTMLRenderer]
    template_name = "products/update.html"
    
    def get(self, request, *args, **kwargs):  # *args와 **kwargs를 추가
        product_id = self.kwargs.get("product_id")  # kwargs에서 user_id 가져오기
        product = get_object_or_404(Product, pk=product_id)  # 안전한 조회를 위해 get_object_or_404 사용
        serializer = ProductDetailSerializer(instance=product)


        context = {
            "form": serializer,
            "product": product,
        }
        return render(request, self.template_name, context)
    
    def post(self, request, *args, **kwargs):  # *args와 **kwargs를 추가
        product_id = self.kwargs.get("product_id")  # kwargs에서 product_id 가져오기
        product = get_object_or_404(Product, pk=product_id)  # 안전한 조회를 위해 get_object_or_404 사용

        if request.POST.get('_method') == 'PUT':
            return self.put(request, *args, **kwargs)

        return HttpResponseNotAllowed(['PUT'])
    
    def put(self, request, *args, **kwargs):  # *args와 **kwargs를 추가
        product_id = self.kwargs.get("product_id")  # kwargs에서 product_id 가져오기
        product = get_object_or_404(Product, pk=product_id)  # 안전한 조회를 위해 get_object_or_404 사용

        data = request.data
        profile_image = request.FILES.get('image')  # Get the uploaded image

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
                product.image = profile_image_url

            except (BotoCoreError, NoCredentialsError) as e:
                return JsonResponse({"error": f"Error uploading to S3: {str(e)}"}, status=500)

        serializer = ProductDetailSerializer(product, data=data, partial=True)
        if serializer.is_valid(raise_exception=True):
            serializer.save()
            return redirect("products:detail", product_id=product.id)

        return render(request, self.template_name, {"product": product, "form": serializer.errors})


class ProductDeleteView(APIView):
    def post(self, request, *args, **kwargs):
        if request.POST.get('_method') == 'DELETE':
            return self.delete(request, *args, **kwargs)
        return HttpResponseNotAllowed(['POST'])

    def delete(self, request, *args, **kwargs):
        product_id = self.kwargs.get('product_id')
        product = get_object_or_404(Product, pk=product_id)
        product.delete()
        return redirect('products:products')  # 삭제 후 목록 페이지로 리다이렉트


class ProductSaveView(APIView):

    def post(self, request, *args, **kwargs):  # *args와 **kwargs를 추가
        product_id = self.kwargs.get("product_id")  # kwargs에서 product_id 가져오기
        product = get_object_or_404(Product, pk=product_id)  # 안전한 조회를 위해 get_object_or_404 사용

        user_id = get_user_id(request)
        user = get_user_model().objects.get(pk=user_id)

        product.saved_users.add(user)

        return redirect("products:detail", product_id=product.id)
    

class ProductLikeView(APIView):
    def post(self, request, *args, **kwargs):
        product_id = self.kwargs.get("product_id")
        product = get_object_or_404(Product, pk=product_id)

        user_id = get_user_id(request)
        user = get_user_model().objects.get(pk=user_id)

        # Toggle like status
        if product.like_users.filter(id=user.id).exists():
            product.like_users.remove(user)
            is_liked = False
        else:
            product.like_users.add(user)
            is_liked = True

        # 리다이렉트와 함께 상태를 전달
        context = {
            "product": product,
            "is_liked": is_liked,
        }
        return redirect("products:detail", product_id=product.id)

    


class MyCartListView(APIView):
    renderer_classes = [TemplateHTMLRenderer]
    template_name = "products/mycart.html"

    def get(self, request):

        try:

            user_id = get_user_id(request)
            user = get_user_model().objects.get(pk=user_id)

            products = Product.objects.filter(saved_users=user)

            context = {"products": products}
            return render(request, self.template_name, context)

        except:
            return redirect('users:login')
        
    def post(self, request, *args, **kwargs):
        # Check if DELETE is requested via POST
        if request.POST.get('_method') == 'DELETE':
            return self.delete(request, *args, **kwargs)
        return HttpResponseNotAllowed(['POST'])

    def delete(self, request, *args, **kwargs):

        user_id = get_user_id(request)
        user = get_user_model().objects.get(pk=user_id)
    # Get product_id from POST data or query string
        product_id = request.POST.get("product_id") or request.GET.get("product_id")
        if not product_id:
            return HttpResponseBadRequest("Product ID not provided.")

        product = get_object_or_404(Product, pk=product_id)
        user_id = get_user_id(request)
        user = get_user_model().objects.get(pk=user_id)

        # Remove the user from saved_users
        product.saved_users.remove(user)
        return redirect('products:mycart')

