import pandas as pd
from sklearn import metrics

# Loading the CSV File
data = pd.read_csv('assignment1dataset.csv')
print(data.describe())

Y = data['Performance Index']
# -------------------------------------------------------------------------------------------------
# Column 1 = Hours Studied
X1 = data['Hours Studied']
L1 = 0.01  # learning Rate
epochs1 = 1000
m1 = 0
c1 = 0
n1 = float(len(X1))
for i in range(epochs1):
    Y_prediction1 = m1 * X1 + c1  # The current predicted value of Y
    Dm1 = (-2/n1) * sum((Y - Y_prediction1) * X1)
    Dc1 = (-2/n1) * sum(Y - Y_prediction1)
    m1 = m1 - L1 * Dm1  # Update m
    c1 = c1 - L1 * Dc1  # Update c
prediction1 = m1 * X1 + c1
print('Mean Square Error of (Hours Studied)', metrics.mean_squared_error(Y, prediction1))

# -------------------------------------------------------------------------------------------------

# Column 2 = Hours Studied
X2 = data['Previous Scores']
L2 = 0.0001  # learning Rate
epochs2 = 100
m2 = 0
c2 = 0
n2 = float(len(X2))
for i in range(epochs2):
    Y_prediction2 = m2 * X2 + c2  # The current predicted value of Y
    Dm2 = (-2/n2) * sum((Y - Y_prediction2) * X2)
    Dc2 = (-2/n2) * sum(Y - Y_prediction2)
    m2 = m2 - L2 * Dm2  # Update m
    c2 = c2 - L2 * Dc2  # Update c
prediction2 = m2 * X2 + c2
print('Mean Square Error of (Previous Scores)', metrics.mean_squared_error(Y, prediction2))

# -------------------------------------------------------------------------------------------------

# Column 3 = Hours Studied
X3 = data['Sleep Hours']
L3 = 0.01  # learning Rate
epochs3 = 10000
m3 = 0
c3 = 0
n3 = float(len(X3))
for i in range(epochs3):
    Y_prediction3 = m3 * X3 + c3  # The current predicted value of Y
    Dm3 = (-2/n3) * sum((Y - Y_prediction3) * X3)
    Dc3 = (-2/n3) * sum(Y - Y_prediction3)
    m3 = m3 - L3 * Dm3  # Update m
    c3 = c3 - L3 * Dc3  # Update c
prediction3 = m3 * X3 + c3
print('Mean Square Error of (Sleep Hours)', metrics.mean_squared_error(Y, prediction3))

# -------------------------------------------------------------------------------------------------

# Column 4 = Hours Studied
X4 = data['Sample Question Papers Practiced']
L4 = 0.01  # learning Rate
epochs4 = 1000
m4 = 0
c4 = 0
n4 = float(len(X4))
for i in range(epochs4):
    Y_prediction4 = m4 * X4 + c4  # The current predicted value of Y
    Dm4 = (-2/n4) * sum((Y - Y_prediction4) * X4)
    Dc4 = (-2/n4) * sum(Y - Y_prediction4)
    m4 = m4 - L4 * Dm4  # Update m
    c4 = c4 - L4 * Dc4  # Update c
prediction4 = m4 * X4 + c4
print('Mean Square Error of (Sample Question Papers Practiced)', metrics.mean_squared_error(Y, prediction4))

# -------------------------------------------------------------------------------------------------

