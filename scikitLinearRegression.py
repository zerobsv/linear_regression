import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# X = np.matrix([[1,1],[1,2],[1,3]]) #Shape: m x (featureSize)
# y = np.matrix([[1],[5],[7]])      #Shape: m x 1
# theta = np.matrix([[1],[1]])      #Shape: (featureSize) x 1

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
        # A more direct way using matrix multiplication for clarity:
        # For simplicity, let's make y a function of the second column of X (index 1)
        y[i][0] = intercept_val + slope_val * X[i][1] + np.random.randn() * 10 # Adding more noise for regularization effect

    return X, y, theta

X, y, theta = generateData()

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

X, y, theta = generateData()

print("Generated Data:")
print("X (first 5 rows):\n", X[:5], "\nShape:", X.shape)
print("y (first 5 rows):\n", y[:5], "\nShape:", y.shape)
print("-" * 30)

# Split data into training and testing sets (good practice for evaluation)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Data Shapes:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("Testing Data Shapes:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)
print("-" * 30)

# Create linear regression object (Ordinary Least Squares - no regularization)
print("1. Ordinary Linear Regression (OLS):")
lr = LinearRegression()
lr.fit(X_train, y_train)

print(f"  Coefficients (excluding intercept): {lr.coef_[0][1:]}") # Exclude the coefficient for the bias term if X contains 1s
print(f"  Intercept: {lr.intercept_[0]}")

# Make predictions and evaluate
y_pred_lr = lr.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred_lr)
print(f"  Mean Squared Error (OLS): {mse_lr:.4f}")
print("-" * 30)


# 2. Ridge Regression (L2 Regularization)
# Alpha is the regularization strength. Higher alpha means stronger regularization.
print("2. Ridge Regression (L2 Regularization):")
ridge_model = Ridge(alpha=1.0) # You can experiment with different alpha values
ridge_model.fit(X_train, y_train)

print(f"  Coefficients (excluding intercept): {ridge_model.coef_}")
print(f"  Intercept: {ridge_model.intercept_}")

# Make predictions and evaluate
y_pred_ridge = ridge_model.predict(X_test)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
print(f"  Mean Squared Error (Ridge): {mse_ridge:.4f}")
print("-" * 30)

# 3. Lasso Regression (L1 Regularization)
# Lasso can lead to sparse coefficients (some coefficients become exactly zero),
# which is useful for feature selection.
print("3. Lasso Regression (L1 Regularization):")
lasso_model = Lasso(alpha=0.1) # You can experiment with different alpha values
lasso_model.fit(X_train, y_train)

print(f"  Coefficients (excluding intercept): {lasso_model.coef_[1:]}") # Lasso coef_ is 1D if y is 1D
print(f"  Intercept: {lasso_model.intercept_[0]}")

# Make predictions and evaluate
y_pred_lasso = lasso_model.predict(X_test)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
print(f"  Mean Squared Error (Lasso): {mse_lasso:.4f}")
print("-" * 30)

# Comparing the models
print("\nSummary of Mean Squared Errors:")
print(f"OLS MSE:   {mse_lr:.4f}")
print(f"Ridge MSE: {mse_ridge:.4f}")
print(f"Lasso MSE: {mse_lasso:.4f}")


# Split the data into training and testing sets
# This is crucial for evaluating how well the model generalizes to unseen data.
# test_size=0.2 means 20% of the data will be used for testing.
# random_state ensures reproducibility of the split.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Data Split Shapes:")
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)
print("-" * 50)

# Initialize the Linear Regression model (Ordinary Least Squares)
# This model aims to minimize the sum of squared residuals, which directly
# translates to minimizing the Mean Squared Error.
model = LinearRegression()

# Train the model using the training data
print("Fitting the Linear Regression model...")
model.fit(X_train, y_train)

# Make predictions on the test data
print("Making predictions on the test set...")
y_pred = model.predict(X_test)

# Evaluate the model using Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)

print("\nModel Evaluation:")
print(f"Mean Squared Error (MSE): {mse:.4f}")

# Display the learned coefficients and intercept
# model.coef_ will contain the coefficients for each feature in X.
# Since X has a column of ones at index 0, model.coef_[0][0] will be the coefficient
# for that bias term, and model.coef_[0][1] for the actual feature.
print(f"Coefficients (excluding intercept term's coefficient if X has bias col): {model.coef_[0][1:]}")
print(f"Intercept: {model.intercept_[0]}")
