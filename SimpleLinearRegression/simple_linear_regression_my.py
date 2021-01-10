# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Fitting Simple Linear Regression To Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test Results
y_pred = regressor.predict(X_test)

# Visualising the Training Set Results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train,regressor.predict(X_train), color = 'blue')
plt.title("Salary vs Experience (Train Set)")
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the Test Set Results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train,regressor.predict(X_train), color = 'blue')
plt.title("Salary vs Experience (Test Set)")
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()



















