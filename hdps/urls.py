from django.conf.urls import url
from .views import *
from django.views.generic.base import TemplateView

app_name='hdps'
urlpatterns = [
url(r'^$',  TemplateView.as_view(template_name="index.html"), name="home"),
url(r'^123/$',prediction,name='prediction'),
url(r'^result/(?P<prediction>[0-9]+)/$',result,name='result'),
]