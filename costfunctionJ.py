import numpy as np
import os
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
		try:
			cost = costFunctionJ(X,y,theta)
			print("Iteration %d | Cost: %f" % (x, cost))
		except Exception as e:
			print("Error calculating cost function: ", e)
			break
		prediction 	= X * theta
		Error  		= np.subtract(prediction, y)
		gradient = (xTrans * Error) * (1.0/m)
		print("Gradient: " , gradient)
		theta = theta - alpha * gradient
	return theta

# Calculate the minimal cost function using the normal equation
def minimal_cost_function():
	xTrans = np.transpose(X)
	invTerm = np.dot(xTrans, X)
	print("Inverse Term: ", invTerm, invTerm.shape)
	invTerm = np.linalg.inv(invTerm)
	return np.dot(np.dot(invTerm, xTrans), y)

def genData(numPoints, bias, variance):
    # x = np.zeros(shape=(2,numPoints))
	x = np.zeros(shape=numPoints)
	y = np.zeros(shape=numPoints)
    # basically a straight line
	for i in range(0, numPoints):
        # bias feature 
		# x[0][i] = 1
		x[i] = i
        # our target variable
		y[i] = (i + bias) + random.uniform(0, 1) * variance
	return x, y

if __name__ == '__main__':
	X , y = genData(100,2.5,0.1)
	theta = np.ones(100)
	numIter = 10
	alpha = 0.005

	# Using gradient descent
	theta = gradientDescent(X,y,theta,alpha,100,numIter)
	print(theta)

	# # Using the normal equation
	# theta = minimal_cost_function()
	# print(theta)