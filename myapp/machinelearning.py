import numpy as np

def sigmoid(x):
    return 1/(1 + np.exp(-x))
def deriv_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))
def labels_equal(y_pred, y, threshold):
    if (len(y_pred) != len(y)):
        raise Exception("Prediction dimension does not match label dimension")
    else:
        error = 0
        for i in range(len(y)):
            error += abs(y_pred[i] - y[i])
        if (error < threshold):
            return 1
        else:
            return 0 
def prediction_to_label(predictions, threshold):
    labels = []
    for i in range(len(predictions)):
        prediction = predictions[i]
        label = [0 for x in range(len(prediction))]
        for j in range(len(prediction)):
            if (prediction[j] > threshold):
                label[j] = 1
        labels.append(label)
    return labels

def accuracy(y_pred, y):
    if (len(y_pred) != len(y)):
        raise Exception("Number of predictions does not match number of correct labels")
    else:
        num_correct = 0
        for i in range(len(y)):
            num_correct += labels_equal(y_pred[i], y[i], 0.001)
        return num_correct, len(y)

class Layer():
    def __init__(self, in_node_num, out_node_num, outLayer = False):
        self.in_dim = in_node_num
        self.out_dim = out_node_num
        self.nodes = np.random.rand(in_node_num, 1) - 0.5
        if not (outLayer):
            self.weights = np.random.rand(out_node_num, in_node_num) - 0.5
            self.biases = np.random.rand(out_node_num, 1) - 0.5
    def set_nodes(self, X):
        if (len(self.nodes) != len(X)):
            raise Exception("Nodes do not match dimensions with inputted data")
        else:
            self.nodes = X.reshape(len(X), 1)

class NeuralNet():
    def __init__(self, num_layers, layer_dims):
        if (num_layers != len(layer_dims)):
            raise Exception("Number of layers do not match dimensions with layers passed into model")
        else:
            self.layers = []
            self.num_layers = num_layers
            self.layer_dims = layer_dims
            for i in range(num_layers-1):
                layer = Layer(layer_dims[i], layer_dims[i+1])
                self.layers.append(layer)
            
            layer = Layer(layer_dims[i], None, True)
            self.layers.append(layer)
    def feedforward(self, X):
        self.layers[0].set_nodes(X)
        for i in range(1, self.num_layers):
            prev_layer = self.layers[i-1]
            #print(prev_layer.weights, "\n", prev_layer.nodes)
            z = prev_layer.weights @ prev_layer.nodes + prev_layer.biases
            self.layers[i].nodes = sigmoid(z)
        return self.layers[-1].nodes
    def cost_function(self, y_pred, y):
        return np.sum(np.power(y_pred-y, 2))
    def backpropagation(self, y):

        weight_grads = [0 for x in range(self.num_layers-1)]
        nodes_grads = [0 for x in range(self.num_layers)]
        biases_grads = [0 for x in range(self.num_layers-1)]

        nodes_grads[-1] = 2 * (self.layers[-1].nodes - y)

        for i in range(self.num_layers-2, -1, -1):
            z = self.layers[i].weights @ self.layers[i].nodes + self.layers[i].biases
            da_dz = deriv_sigmoid(z)

            nodes_grads[i] = self.layers[i].weights.T @ (nodes_grads[i+1] * da_dz) 

            dz_dw = self.layers[i].nodes

            weight_grads[i] = (nodes_grads[i+1] * da_dz) @ (dz_dw.T)
            biases_grads[i] = nodes_grads[i+1] * da_dz
        return weight_grads, biases_grads, nodes_grads
    def update_weights_and_biases(self, weight_grads, biases_grads, learning_rate):
        for i in range(self.num_layers-1):
            self.layers[i].weights = np.add(self.layers[i].weights, -(weight_grads[i] * learning_rate))
            self.layers[i].biases = np.add(self.layers[i].biases ,-(biases_grads[i] * learning_rate))
    def train(self, X, y, batch_size, epochs, learning_rate):
        batch_num = len(y)//batch_size
        cost = []
        for e in range(epochs):
            for i in range(batch_num):
                weight_grad_avg = [0 for i in range (self.num_layers-1)]
                biases_grad_avg = [0 for i in range (self.num_layers-1)]

                batch_cost = 0
                for j in range(batch_size):

                    self.feedforward(X[i * batch_size + j])
                    #print(self.layers[-1].nodes)
                    batch_cost += self.cost_function(self.layers[-1].nodes, y[i * batch_size + j])

                    weight_grad, biases_grad, _ = self.backpropagation(y[i * batch_size + j])

                    for l in range(self.num_layers-1):
                        if (j != 0):
                            weight_grad_avg[l] = np.add(weight_grad[l], weight_grad_avg[l])
                            biases_grad_avg[l] = np.add(biases_grad[l], biases_grad_avg[l])
                        else:
                            weight_grad_avg[l] = weight_grad[l]
                            biases_grad_avg[l] = biases_grad[l]
                for l in range(self.num_layers-1):
                    weight_grad_avg[l] = weight_grad_avg[l]/batch_size
                    biases_grad_avg[l] = biases_grad_avg[l]/batch_size

                cost.append(batch_cost/batch_size)

                self.update_weights_and_biases(weight_grad_avg, biases_grad_avg, learning_rate)
        return (cost[-1])
    def predict(self, X):
        predictions = []
        for i in range(len(X)):
            predictions.append(self.feedforward(X[i]))
        return predictions
    def reset(self):
        self.__init__(self.num_layers, self.layer_dims)

#how to use the class
def createNetwork():
    nn = NeuralNet(3, [2, 5, 2]) #Network with 3 layers, input of 4 features, and output of 3 buckets.

    from sklearn.datasets import make_moons

    X, Y = make_moons(random_state=42, n_samples=(1000, 1000), noise=0.25)
    y = []

    for i in range(len(Y)):
        if Y[i] == 0:
            y.append(np.array([1, 0]))
        if Y[i] == 1:
            y.append(np.array([0, 1]))
        y[-1] = y[-1].reshape(2, 1)

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)

    nn.train(x_train, y_train, 1, 100, 0.001)



    y_pred = nn.predict(x_test)
    pred_labels = prediction_to_label(y_pred, .7)
    correct, total = accuracy(pred_labels, y_test)
    print(f"The model had an accuracy of {correct/total} with {correct} cases correct out of {total} total cases")