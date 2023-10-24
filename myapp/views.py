from rest_framework.response import Response
from rest_framework.decorators import api_view
from django.shortcuts import render

import plotly.express as px
import numpy as np
from . import dataset
from sklearn.datasets import make_moons
# Create your views here.

from django.template.defaulttags import register

@register.filter
def get_range(value):
    return range(value)

def websocket(request):

    dataset.make_data(20)
    """
    y = []
    for i in range(len(Y)):
        if Y[i] == 0:
            y.append(np.array([1, 0]))
        if Y[i] == 1:
            y.append(np.array([0, 1]))
        y[-1] = y[-1].reshape(2, 1)
    """

    range_feat_1 = max(dataset.x_train.T[0]) - min(dataset.x_train.T[0])
    range_feat_2 = max(dataset.x_train.T[1]) - min(dataset.x_train.T[1])
    perc_feat_1 = (((dataset.x_train.T[0] - min(dataset.x_train.T[0]))/range_feat_1)) * 98 
    perc_feat_2 = (((dataset.x_train.T[1] - min(dataset.x_train.T[1]))/range_feat_2)) * 98

    data_info = []
    for i in range(len(perc_feat_1)):
        data_info.append([perc_feat_1[i], perc_feat_2[i], "#FF0000" if dataset.y_train[i][0] == 1 else "#0000FF"])

    return render(request, "websocket.html", {"data_info": data_info})

def home(request):
    return render(request, "test.html")