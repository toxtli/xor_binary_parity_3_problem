import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

class PerceptronRule(object):
    def __init__(self, eta=0.01, epochs=50):
        self.eta = eta
        self.epochs = epochs

    def train(self, X, y):

        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.epochs):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] +=  update * xi
                self.w_[0] +=  update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return self.net_input(X)
        return np.where(self.net_input(X) >= 0.0, 1, -1)

class GradientDescentRule(object):

    def __init__(self, eta=0.01, epochs=50):
        self.eta = eta
        self.epochs = epochs

    def train(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []
        iterations = range(self.epochs)
        err = []
        for i in iterations:
            output = self.net_input(X)
            errors = (y - output)
            err.append(np.mean(np.abs(errors)))
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        plt.plot(iterations, err)
        plt.xlabel('Iterations')
        plt.ylabel('Error')
        plt.grid(True)
        plt.show()
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        return self.net_input(X)

    def predict(self, X):
        return self.activation(X)
        return np.where(self.activation(X) >= 0.0, 1, -1)

class StochasticGradientDescent(object):

    def __init__(self, eta=0.01, epochs=50):
        self.eta = eta
        self.epochs = epochs

    def train(self, X, y, reinitialize_weights=True):

        if reinitialize_weights:
            self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for i in range(self.epochs):
            for xi, target in zip(X, y):
                output = self.net_input(xi)
                error = (target - output)
                self.w_[1:] += self.eta * xi.dot(error)
                self.w_[0] += self.eta * error

            cost = ((y - self.activation(X))**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        return self.net_input(X)

    def predict(self, X):
        return self.activation(X)
        return np.where(self.activation(X) >= 0.0, 1, -1)

def process_input(X):
    x = X[:,[0,1,2]]
    x = np.insert(x, 3, X[:,0] & X[:,1], axis=1)
    x = np.insert(x, 4, X[:,0] & X[:,2], axis=1)
    x = np.insert(x, 5, X[:,1] & X[:,2], axis=1)
    x = np.insert(x, 6, X[:,0] & X[:,1] & X[:,2], axis=1)
    return x

epochs = 10000
eta = 0.1
X = np.array([[1, 1, 1],
              [1, 0, 0],
              [0, 1, 0],
              [0, 0, 1],
              [0, 0, 0],
              [1, 0, 1],
              [0, 1, 1],
              [1, 1, 0]])

y = np.array([1, 1, 1, 1, 0, 0, 0, 0])

ppn = PerceptronRule(epochs=epochs, eta=eta)
ppn.train(X, y)
print('Weights: %s' % ppn.w_)
for x in X:
    print(x, ppn.predict(x))

ppn = GradientDescentRule(epochs=epochs, eta=eta)
ppn.train(X, y)
print('Weights: %s' % ppn.w_)
for x in X:
    print(x, ppn.predict(x))

ppn = StochasticGradientDescent(epochs=epochs, eta=eta)
ppn.train(X, y)
print('Weights: %s' % ppn.w_)
for x in X:
    print(x, ppn.predict(x))


X = process_input(X)

ppn = PerceptronRule(epochs=epochs, eta=eta)
ppn.train(X, y)
print('Weights: %s' % ppn.w_)
for x in X:
    print(x, ppn.predict(x))

ppn = GradientDescentRule(epochs=epochs, eta=eta)
ppn.train(X, y)
print('Weights: %s' % ppn.w_)
for x in X:
    print(x, ppn.predict(x))

ppn = StochasticGradientDescent(epochs=epochs, eta=eta)
ppn.train(X, y)
print('Weights: %s' % ppn.w_)
for x in X:
    print(x, ppn.predict(x))

# plot_decision_regions(X, y, clf=ppn)
# plt.title('Perceptron')
# plt.xlabel('sepal length [cm]')
# plt.ylabel('petal length [cm]')
# plt.show()

# plt.plot(range(1, len(ppn.errors_)+1), ppn.errors_, marker='o')
# plt.xlabel('Iterations')
# plt.ylabel('Misclassifications')
# plt.show()