import numpy as np

def costFunctionJ(X, y, theta):
	m 			= X.shape[0]
	prediction 	= np.dot(X, theta)
	Error  		= np.subtract(prediction, y)
	sqrError	= np.power(Error, 2)
	J   		= ( 1.0 / (2 * m) ) * np.sum(sqrError) 
	return J

def gradientDescent(X, y, theta, alpha, numIter):
	m = X.shape[0]
	for iter in range(numIter):
		gradient = theta - alpha * (1.0/m) * np.sum(np.subtract(np.dot(X, theta), y))
		print("Gradient: ", gradient, "Iteration: ", iter)
		theta = gradient
	return theta

def generateData(rows=100, cols=2):
    """
    Generates synthetic data for linear regression.
    X: Feature matrix with an added bias (intercept) term.
    y: Target variable.
    theta: Initial parameters (not directly used in scikit-learn fitting,
           but useful for understanding the underlying linear model concept).
    """
    X = np.ones(shape=(rows, cols))
    y = np.ones(shape=(rows, 1))
    theta = np.ones(shape=(cols, 1))

    # Populate X with some varying values
    for i in range(0, rows):
        for j in range(1, cols): # Start from 1 because X[i][0] is for bias and remains 1
            X[i][j] = (i * j / 17) * 100 + np.random.randn() * 5 # Add some noise

    # y = intercept + slope * feature + noise
    intercept_val = 15
    slope_val = 0.5 # Example slope for the single feature X[:, 1]
    for i in range(rows):
        # y[i][0] = intercept_val + slope_val * X[i][1] + np.random.randn() * 2
        # For simplicity, let's make y a function of the second column of X (index 1)
        y[i][0] = intercept_val + slope_val * X[i][1] + np.random.randn() * 10 # Adding more noise for regularization effect

    return X, y, theta

# def generateData(rows=100, cols=2):
# 	X = np.ones(shape=(rows, cols))
# 	y = np.ones(shape=(rows, 1))
# 	theta = np.ones(shape=(cols, 1))
# 	for i in range(0, rows):
# 		for j in range(1, cols):
# 			X[i][j] = (i * j / 17) * 100
# 	for i in range(rows):
# 		y[i][0] = 15
# 	return X, y, theta


if __name__ == '__main__':
#	X = np.matrix([[1,1],[1,2],[1,3]]) #Shape: m x (featureSize)
#	y = np.matrix([[1],[5],[7]])       #Shape: m x 1
#	theta = np.matrix([[1],[1]])       #Shape: (featureSize) x 1
	X,y,theta = generateData()
	alpha = 0.005
	numIter = 1000
	beforeLearn = costFunctionJ(X, y, theta)
	theta = gradientDescent(X, y, theta, alpha, numIter)
	print("Before Learning: ")
	print(beforeLearn)
	print("After learning: ")
	print(costFunctionJ(X, y, theta))

