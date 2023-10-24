
import json
from channels.generic.websocket import WebsocketConsumer
from . import machinelearning as ml
import numpy as np
from . import dataset

nn = ml.NeuralNet(4, [2, 4, 4, 2]) #Network with 3 layers, input of 2 features, and output of 2 buckets.

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

def get_percentages(data):
    range_feat_1 = max(data.T[0]) - min(data.T[0])
    range_feat_2 = max(data.T[1]) - min(data.T[1])
    perc_feat_1 = (((data.T[0] - min(data.T[0]))/range_feat_1)) * 98 
    perc_feat_2 = (((data.T[1] - min(data.T[1]))/range_feat_2)) * 98

    data_info = []
    for i in range(len(perc_feat_1)):
        data_info.append([perc_feat_1[i], perc_feat_2[i], "#FF0000" if dataset.y_train[i][0] == 1 else "#0000FF"])
    return data_info

class StreamMLInfo(WebsocketConsumer):
    def connect(self):
        self.accept()
  
    def disconnect(self, close_code):
        self.close()   
  
    def receive(self, text_data):
        global nn
        text_data_json = json.loads(text_data)
        if ('random_seed' in text_data_json):
            dataset.make_data(int(text_data_json['random_seed']))
            self.send(text_data=json.dumps({
                        'newdata': get_percentages(dataset.x_train)
                    }))
        if ('noise' in text_data_json):
            dataset.make_data(None, float(text_data_json["noise"]))
            self.send(text_data=json.dumps({
                        'newdata': get_percentages(dataset.x_train)
                    }))
        if ('reset' in text_data_json and text_data_json['reset'] == "true"):
            nn.reset()
            self.send(text_data=json.dumps({
                        'cost': "Not Training",
                        'colors': dataset.y_train_list
                    }))
        elif ('train' in text_data_json and text_data_json['train'] == "true"): 
            for i in range(text_data_json['epochs']):
                cost = nn.train(dataset.x_train, dataset.y_train, text_data_json['batch_size'], 1, 0.005)
                if (i%100 == 0):
                    y_pred = nn.predict(dataset.x_train)
                    pred_labels = ml.prediction_to_label(y_pred, .7)
                    self.send(text_data=json.dumps({
                        'cost': "Cost: " + str(cost),
                        'colors': pred_labels
                    }))
                else:
                    self.send(text_data=json.dumps({
                        'cost': cost,
                    }))

