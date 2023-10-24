
from django.urls import re_path
  
from . import consumers
  
websocket_urlpatterns = [
    re_path(r'ws/livec/1/$', consumers.StreamMLInfo.as_asgi())
]