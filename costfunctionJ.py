import numpy as np
import random

# Calculate the cost function J for linear regression
def costFunctionJ(X,y,theta):
	m 			= X.shape[0]
	prediction 	= X * theta
	print("Prediction: ", prediction, prediction.shape)
	print("Y: ", y, y.shape)
	Error  		= np.subtract(prediction, y)
	sqrError	= np.power(Error, 2)
	J   		= (1.0/(2*m)) * np.sum(sqrError) 
	return J

# Perform gradient descent to minimize the cost function
def gradientDescent(X,y,theta,alpha,m,numIter):
	xTrans      = np.transpose(X)
	for x in range(numIter):
		prediction 	= np.dot(xTrans, theta)
		Error  		= np.subtract(prediction, y)
		print("Error: ", Error, Error.shape)
		print("Theta: ", theta, theta.shape)
		print("X Transpose: ", xTrans, xTrans.shape)
		gradient = np.dot(X, Error)
		gradient = gradient * (1.0/m)
		print("Gradient: " , gradient, gradient.shape)
		theta = theta - (alpha * gradient)
		print("Theta: ", theta, theta.shape)
	return theta

# Calculate the minimal cost function using the normal equation
def minimal_cost_function():
	xTrans = np.transpose(X)
	invTerm = np.dot(xTrans, X)
	print("Inverse Term: ", invTerm, invTerm.shape)
	invTerm = np.linalg.inv(invTerm)
	return np.dot(np.dot(invTerm, xTrans), y)

def genData(numPoints, bias, variance):
	x = np.zeros(shape=(2, numPoints))
	y = np.zeros(shape=numPoints)
	# basically a straight line
	for i in range(0, numPoints):
		# bias feature 
		x[0][i] = 1
		x[1][i] = i
		# our target variable
		y[i] = (i + bias) + random.uniform(0, 1) * variance
	return x, y

if __name__ == '__main__':
	X , y = genData(100,2.5,0.1)
	theta = np.ones((2, 100))
	numIter = 10
	alpha = 0.005

	initialcost = costFunctionJ(X,y,theta)

	# Using gradient descent
	theta = gradientDescent(X,y,theta,alpha,100,numIter)
	print(theta)

	finalcost = costFunctionJ(X,y,theta)
	print("Initial Cost: ", initialcost)
	print("Final Cost: ", finalcost)

	# # Using the normal equation
	# theta = minimal_cost_function()
	# print(theta)