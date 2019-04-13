import numpy as np


def generate_linear(n=100):
    pts = np.random.uniform(0, 1, (n, 2))
    inputs, labels = [], []
    for x, y in pts:
        inputs.append((x, y))
        labels.append(0 if x > y else 1)
    return np.array(inputs), np.array(labels).reshape(-1, 1)

def generate_XOR_easy():
    inputs, labels = [], []
    for i in range(11):
        x = 0.1 * i
        inputs.append((x, x))
        labels.append(0)
        if x == 1 - x:
            continue
        inputs.append((x, 1 - x))
        labels.append(1)
    return np.array(inputs), np.array(labels).reshape(-1, 1)

def show_result(x, y, y_pred):
    import matplotlib.pyplot as plt
    plot = None

    def plot(s, title, y):
        plt.subplot(1, 2, s)
        plt.title(title, fontsize=18)
        for i in range(x.shape[0]):
            if y[i] == 0:
                plt.plot(*x[i], 'ro')
            else:
                plt.plot(*x[i], 'bo')

    plot(1, 'Ground truth', y)
    plot(2, 'Predict result', y_pred)
    plt.show()

def derivative(f, *args):
    return f(*args, derivative=True)

def sigmoid(x, derivative=False):
    if derivative:
        return x * (1. - x)
    return 1. / (1 + np.exp(-x))

def mse(y_pred, y_data, derivative=False):
    if derivative:
        return y_pred - y_data
    return (y_pred - y_data) ** 2 / 2

class Dense:
    """Dense Layer"""

    def __init__(self, dim, activation=sigmoid, loss=mse):
        def init_weights(d):
            #return np.ones(d)
            #return np.random.randn(*d) / 100
            r = 4 * np.sqrt(6. / sum(d))
            return np.random.uniform(-r, r, d)

        self.layers = [None] * len(dim)
        self.weights = [init_weights(d) for d in zip(dim[:-1], dim[1:])]
        self.act = activation
        self.loss = loss

    def predict(self, X):
        assert(X.shape[0] == 1), 'X should be a row vector'
        Z = X
        self.layers[0] = np.array(Z)
        for i, W in enumerate(self.weights):
            Z = self.act(Z @ W)
            self.layers[i + 1] = np.array(Z)
        return Z

    def backprop(self, y_data):
        assert(y_data.shape[0] == 1), 'y should be a row vector'
        Y = self.layers[-1]
        dL = derivative(self.loss, Y, y_data)
        dz = np.multiply(dL, derivative(self.act, Y))
        dW, dZ = [], [dz]
        for W, Z in zip(self.weights[::-1], self.layers[:-1][::-1]):
            # dw = dL / dW
            dw = np.kron(Z.T, dz)
            dW.append(np.array(dw))
            # dz = dL / dZ * s(Z)
            dz = np.multiply(dz @ W.T, derivative(self.act, Z))
            dZ.append(np.array(dz))
        self.dW = dW[::-1]

    def update(self, lr):
        #print([w.shape for w in self.weights])
        #print([w.shape for w in self.dW])
        for i, dw in enumerate(self.dW):
            self.weights[i] -= lr * dw

nn = Dense(dim=[2, 3, 3, 1])
lr, epoch, done = 0.8, 0, False

X, Y = generate_XOR_easy()
#X, Y = generate_linear()
while not done:
    loss = []
    for x, y in zip(X, Y):
        x, y = x.reshape(1, -1), y.reshape(1, -1)
        y_pred = nn.predict(x)
        nn.backprop(y)
        nn.update(lr)
        loss += [mse(y_pred, y)]
        epoch += 1
        if epoch % 5000 == 0:
            print('epoch', epoch, 'loss:', loss[-1])
    done = all(np.array(loss) < 0.04)

Y_pred = [int(nn.predict(x.reshape(1, -1)) >= 0.5) for x in X]
show_result(X, Y, Y_pred)
