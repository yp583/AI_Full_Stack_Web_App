import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
X = []
Y = []
x_train = []
x_test = [] 
y_train_list = [] 
y_train = []
y_test = []
global_seed = 20
global_noise = 0.25
def make_data(seed = None, noise = None):
    global x_train, x_test, y_train, y_test, y_train_list, X, Y, global_seed, global_noise
    if (seed is not None and seed != global_seed):
        global_seed = seed
    if (noise is not None and noise != global_noise):
        global_noise = noise
    X, Y = make_moons(random_state=global_seed, n_samples=(500, 500), noise=global_noise)
    y = []
    for i in range(len(Y)):
        if Y[i] == 0:
            y.append(np.array([1, 0]))
        if Y[i] == 1:
            y.append(np.array([0, 1]))
        y[-1] = y[-1].reshape(2, 1)

    #split data
    
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)
    y_train_list = []
    for i in range(len(y_train)):
        element = y_train[i].tolist()
        y_train_list.append([element[0][0], element[1][0]])