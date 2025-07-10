import numpy as np
import random

# Calculate the minimal cost function using the normal equation
def minimal_cost_function(X, y):
	product = X.T @ X # or np.dot
	print("Product: ", product, product.shape)
	invTerm = np.linalg.inv(product)
	print("Inverse Term: ", invTerm, invTerm.shape)
	secondTerm = X.T @ y # or np.dot
	print("Second Term: ", secondTerm, secondTerm.shape)
	theta = np.dot(invTerm, secondTerm)
	print("Inverse Term X Transpose: ", secondTerm, secondTerm.shape)
	# theta = inv(X^T * X) * (X^T * y)
	# where X is the design matrix and y is the target variable
	return theta

# Calculate the cost function J for linear regression
def costFunctionJ(X,y,theta):
	m 			= X.shape[0]
	prediction 	= X @ theta # or np.matmul(X, theta)
	# print("Prediction: ", prediction, prediction.shape)
	# print("Y: ", y, y.shape)
	Error  		= np.subtract(prediction, y)
	sqrError	= np.power(Error, 2)
	J   		= (1.0/(2*m)) * np.sum(sqrError) 
	return J

# Perform gradient descent to minimize the cost function
def gradientDescent(X,y,theta,alpha,numIter):
	m = X.shape[0]
	Xt = X.T
	for _ in range(numIter):
		prediction 	= X @ theta # or np.matmul(X, theta)
		Error  		= np.subtract(prediction, y)
		print("Error: ", Error, Error.shape)
		gradient = Xt @ Error
		gradient = gradient * (1.0/m)
		print("Gradient: " , gradient, gradient.shape)
		theta = theta - (alpha * gradient)
		print("Theta: ", theta, theta.shape)
	return theta

def genData(numPoints, bias, variance):
	x = np.ones(shape=(numPoints, 2))
	y = np.ones(shape=(numPoints, 1))
	# basically a straight line
	for i in range(0, numPoints):
		x[i][1] = i
		# our target variable
		y[i] = (i + bias) + random.uniform(0, 1) * variance
	return x, y

if __name__ == '__main__':
	X , y = genData(100,2.5,0.1)
	theta = np.ones((2, 1)) # Shape (2, 1) for two features (bias and x)
	numIter = 10
	alpha = 0.05


	initialcost = costFunctionJ(X,y,theta)

	# Using gradient descent
	theta = gradientDescent(X,y,theta,alpha,numIter)
	print("Theta after Gradient Descent: ", theta, theta.shape)

	finalcost = costFunctionJ(X,y,theta)
	print("Initial Cost: ", initialcost)
	print("Final Cost: ", finalcost)

	# Using the normal equation
	theta = minimal_cost_function(X, y)
	print("Theta after Normal Equation: ", theta, theta.shape)

	finalcost = costFunctionJ(X,y,theta)
	print("Final Cost Using Normal Equation: ", finalcost)



