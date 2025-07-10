import numpy as np
import os
import random


def costFunctionJ(X,y,theta):
	m 			= X.shape[0]
	prediction 	= X*theta
	Error  		= np.subtract(prediction,y)
	sqrError	= np.power(Error,2)
	J   		= (1.0/(2*m)) * np.sum(sqrError) 
	return J

def gradientDescent(X,y,theta,alpha,m,numIter):
	xTrans = np.transpose(X)
	for x in xrange(numIter):
		cost = costFunctionJ(X,y,theta)
		print("Iteration %d | Cost: %f" % (x, cost))
		prediction 	= X*theta
		Error  		= np.subtract(prediction,y)
        gradient = np.dot(xTrans, Error) * (1.0/m)
        print "Gradient: " , gradient
        theta = theta - alpha * gradient
	

	return theta

def genData(numPoints, bias, variance):
    x = np.zeros(shape=(2,numPoints))
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
	theta = np.ones(100)
	numIter = 100000
	alpha = 0.005
	theta = gradientDescent(X,y,theta,alpha,100,numIter)
	print(theta)