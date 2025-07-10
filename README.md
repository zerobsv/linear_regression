# Multivariate Linear Regression from scratch

- Custom Regularized Linear Regression: Using the lambda parameter, and also appropriately writing
equations for the bias term vs the other terms

- Normal Equation: The cost function optimization without the need for squaring the terms,
using inv(XtX) * XtY was tried, 

- Will be trying it on my GPU using CUDA

Results for the custom regularized Linear Regression: (best runs), This is what I'm observing with the following params

## Custom Regularized Linear Regression: (best runs)

#### Lambda     = 0.05
#### Alpha      = 0.0005
#### Iterations = 180000

------------------------------------------------------------

Gradient:  [[15.00530592]
 [ 0.50177813]] Iteration:  179998
Gradient:  [[15.00535389]
 [ 0.50177796]] Iteration:  179999

------------------------------------------------------------

Before Learning:
12368.136049380459
After learning:
47.34815634266483

------------------------------------------------------------

Product:  [[1.00000000e+02 2.91274940e+04]
 [2.91274940e+04 1.13647479e+07]] (2, 2)
Inverse Term:  [[ 3.94521785e-02 -1.01114701e-04]
 [-1.01114701e-04  3.47145215e-07]] (2, 2)
Second Term:  [[  16116.06214659]
 [6107282.94720604]] (2, 1)
Inverse Term X Transpose:  [[  16116.06214659]
 [6107282.94720604]] (2, 1)
Theta after Normal Equation:  [[18.27766961]
 [ 0.49054324]] (2, 1)

------------------------------------------------------------

Final Cost Using Normal Equation:  50.97526904695224

------------------------------------------------------------

Gradient:  [[15.85971359]
 [ 0.50116183]] Iteration:  179998
Gradient:  [[15.85976449]
 [ 0.50116166]] Iteration:  179999

Before Learning:
12326.426859801128
After learning:
60.477858924924206

------------------------------------------------------------
Product:  [[1.00000000e+02 2.92039694e+04]
 [2.92039694e+04 1.14063729e+07]] (2, 2)
Inverse Term:  [[ 3.96377415e-02 -1.01485319e-04]
 [-1.01485319e-04  3.47505223e-07]] (2, 2)
Second Term:  [[  16221.87631402]
 [6151334.91106871]] (2, 1)
Inverse Term X Transpose:  [[  16221.87631402]
 [6151334.91106871]] (2, 1)
Theta after Normal Equation:  [[18.7283544 ]
 [ 0.49133872]] (2, 1)

------------------------------------------------------------

Final Cost Using Normal Equation:  64.05000170940527

------------------------------------------------------------

## Scikit Learn Linear Regression Models: (best runs)

Generated Data:

------------------------------
X (first 5 rows):
 [[ 1.         -2.49653081]
 [ 1.         -1.9800878 ]
 [ 1.         14.58034267]
 [ 1.         14.12071876]
 [ 1.         20.04096805]]
Shape: (100, 2)

y (first 5 rows):
 [[11.26550098]
 [ 8.25986341]
 [18.20742223]
 [11.87128823]
 [13.60138997]]
Shape: (100, 1)

------------------------------
Training Data Shapes:
X_train: (80, 2)
y_train: (80, 1)
Testing Data Shapes:
X_test: (20, 2)
y_test: (20, 1)

------------------------------------------------------------
#### Ordinary Linear Regression (OLS):

Coefficients (excluding intercept): [0.49568963]

Intercept: 15.516799375033514

Mean Squared Error (OLS): 82.8131

------------------------------------------------------------

#### Ridge Regression (L2 Regularization):

Coefficients (excluding intercept): [0.  0.49568942

Intercept: [15.51686294]

Mean Squared Error (Ridge): 82.8133

------------------------------------------------------------

#### Lasso Regression (L1 Regularization):

Coefficients (excluding intercept): [0.49568619]

Intercept: 15.517825218570579

Mean Squared Error (Lasso): 82.8160

------------------------------------------------------------

## Summary of Mean Squared Errors:

OLS MSE:   82.8131

Ridge MSE: 82.8133

Lasso MSE: 82.8160

------------------------------------------------------------

### Data Split Shapes:

X_train shape: (80, 2)

y_train shape: (80, 1)

X_test shape: (20, 2)

y_test shape: (20, 1)

------------------------------------------------------------