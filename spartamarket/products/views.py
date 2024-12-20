from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from .forms import ProductForm
from .models import Product
from django.views.decorators.http import require_POST

# Create your views here.
def products(request):

    products = Product.objects.all()
    context = {
        "products": products
    }
    return render(request, "products/shop.html", context)


@login_required
def create(request):
    if request.method == "POST":
        form = ProductForm(request.POST, request.FILES)
        if form.is_valid():
            article = form.save(commit=False)
            article.author = request.user
            article.save()
            return redirect("products:products")
    
    else:
        form = ProductForm()
    
    context = {"form": form}
    return render(request, "products/create.html", context)

@require_POST
def delete(request, pk):

    if request.user.is_authenticated:
        product = get_object_or_404(Product, pk=pk)
        product.delete()
    return redirect("products:products")


def detail(request, pk):
    product = get_object_or_404(Product, pk=pk)
    context = {
        "product": product,
    }

    return render(request, "products/detail.html", context)
