import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x, deriv=False):
	if deriv == True:
		return x * (1 - x)
	return 1 / (1 + np.exp(-x))

def backprop_2_layers(X, y):
	np.random.seed(1)

	#synapses
	syn0 = 2*np.random.random((X.shape[1], X.shape[0])) - 1
	syn1 = 2*np.random.random((y.shape[0], y.shape[1])) - 1

	steps = 10000
	iterations = range(steps)
	errors = []
	#training step
	for j in iterations:
		l0 = X
		l1 = sigmoid(np.dot(l0, syn0))
		l2 = sigmoid(np.dot(l1, syn1))
		l2_error = y - l2

		errors.append(np.mean(np.abs(l2_error)))

		l2_delta = l2_error * sigmoid(l2, deriv=True)
		l1_error = l2_delta.dot(syn1.T)
		l1_delta = l1_error * sigmoid(l1, deriv=True)

		#update weights
		syn1 += l1.T.dot(l2_delta)
		syn0 += l0.T.dot(l1_delta)

	print('Output after training')
	print('layer 0')
	print(l0)
	print('layer 1')
	print(l1)
	print('layer 2')
	print(l2)
	print('weights 0')
	print(syn0)
	print('weights 1')
	print(syn1)
	print('layer 1 delta')
	print(l1_delta)
	print('layer 2 delta')
	print(l2_delta)
	print('layer 1 error')
	print(l1_error)
	print('layer 2 error')
	print(l2_error)
	plt.plot(iterations, errors)
	plt.xlabel('Iterations')
	plt.ylabel('Error')
	plt.grid(True)
	plt.show()

def process_input(X):
	x = X[:,[0,1,2]]
	x = np.insert(x, 3, X[:,0] & X[:,1], axis=1)
	x = np.insert(x, 4, X[:,0] & X[:,2], axis=1)
	x = np.insert(x, 5, X[:,1] & X[:,2], axis=1)
	x = np.insert(x, 6, X[:,0] & X[:,1] & X[:,2], axis=1)
	return x

def backprop_1_layer(X, y):
	print(y.shape)
	X = process_input(X)
	np.random.seed(1)

	#synapses
	syn0 = 2*np.random.random((X.shape[1], X.shape[0])) - 1
	syn1 = 2*np.random.random((y.shape[0], y.shape[1])) - 1

	#training step
	steps = 10000
	iterations = range(steps)
	errors = []
	for j in iterations:
		l0 = X
		l1 = sigmoid(np.dot(l0, syn0))
		l1_error = y - l1

		errors.append(np.mean(np.abs(l1_error)))

		l1_delta = l1_error * sigmoid(l1, deriv=True)

		#update weights
		syn0 += l0.T.dot(l1_delta)

	print('Output after training')
	print(l1)
	print(syn0)
	plt.plot(iterations, errors)
	plt.xlabel('Iterations')
	plt.ylabel('Error')
	plt.grid(True)
	plt.show()

X = np.array([[1, 1, 1],
			  [1, 0, 0],
			  [0, 1, 0],
			  [0, 0, 1],
			  [0, 0, 0],
			  [1, 0, 1],
			  [0, 1, 1],
			  [1, 1, 0]])

y = np.array([[1],
			  [1],
			  [1],
			  [1],
			  [0],
			  [0],
			  [0],
			  [0]])

#backprop_2_layers(X, y)
backprop_1_layer(X, y)