import numpy as np
import os


def costFunctionJ(X,y,theta):
	m 			= X.shape[0]
	prediction 	= np.dot(X,theta)
	Error  		= np.subtract(prediction,y)
	sqrError	= np.power(Error,2)
	J   		= (1.0/(2*m)) * np.sum(sqrError) 
	return J

def gradientDescent(X,y,theta,alpha,m,numIter):
	for i in xrange(numIter):
		gradient = theta - alpha * (1.0/m) * np.sum(np.subtract(np.dot(X,theta),y))
		print "Gradient: ", gradient, "Iteration: ", i 
		theta = gradient
	return theta

def generateData():
	X = np.zeros(shape=(rows,cols))
	y = np.zeros(shape=(rows,1))
	theta = np.ones(shape=(cols,1))
	for i in xrange(0,rows):
		for j in xrange(0,cols):
			if(j == 0):
				X[i][j] = 1
			else:
				X[i][j] = (i*j/17) *100
	for i in xrange(rows):
		y[i][0] = 15
	return X,y,theta


if __name__ == '__main__':
#	X = np.matrix([[1,1],[1,2],[1,3]]) #Shape: m x (featureSize)
#	y = np.matrix([[1],[5],[7]])       #Shape: m x 1
#	theta = np.matrix([[1],[1]])       #Shape: (featureSize) x 1
	rows = 100
	cols = 2
	X,y,theta = generateData()
	m = X.shape[0]
	alpha = 0.0001
	numIter = 10000
	beforeLearn = costFunctionJ(X,y,theta)
	theta = gradientDescent(X,y,theta,alpha,m,numIter)
	print "Before Learning: "
	print beforeLearn
	print "After learning: "
	print costFunctionJ(X,y,theta)
