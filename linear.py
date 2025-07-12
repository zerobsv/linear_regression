# import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Test the GPU
print(tf.config.list_physical_devices('GPU'))
print(tf.add(tf.constant([1.0, 2.0]), tf.constant([3.0, 4.0])).device)

# Import Keras
from tensorflow import keras

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

print("X.shape:", X,  X.shape)
print("y.shape:", y,  y.shape)
print("theta.shape:", theta, theta.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# --- Feature Scaling ---
scaler = StandardScaler()

# This calculates the mean and standard deviation from the training set.
scaler.fit(X_train)

# 3. Transform both the training and testing data using the fitted scaler
# This prevents "data leakage" from the test set.
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nScaled X_train shape:", X_train_scaled.shape)
print("Example of scaled data (first 5 rows):\n", X_train_scaled[:5])


model = keras.Sequential([
    # keras.layers.Input(shape=(2,)),
    keras.layers.Input(shape=(2,)),
    keras.layers.Dense(1, activation='linear')
])

model.compile(optimizer='sgd', loss='mse', metrics=['mae'])

model.fit(X_train_scaled, y_train, epochs=120, validation_data=(X_test_scaled, y_test))

model.evaluate(X_test_scaled, y_test)

predictions = model.predict(X_test_scaled)

print("Sample predictions vs. actual labels:")
for i in range(10):
    print(f"X_Values: {X_test_scaled[i]}, Predicted: {predictions[i]}, Actual: {y_test[i]}")

model.summary()
