# MLR Using Backward Elimination: Automatic Elimination

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('kc_house_data.csv')
dataset = dataset.drop(['id', 'date'], axis = 1)
X = dataset.iloc[:, 1:].values
y = dataset.iloc[:, 0].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# fitting Multiple Linear Regression to Test Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting The Test set Results
y_pred = regressor.predict(X_test)



import statsmodels.formula.api as sm
def backwardElimination(X, SL):
    numVars = len(X[0])

    
    temp = np.zeros((21613,19)).astype(int)
    
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, X).fit()
        
        maxVar = max(regressor_OLS.pvalues).astype(float)
        
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        
        if maxVar > SL:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    temp[:,j] = X[:, j]

                    X = np.delete(X, j, 1)
                    
                    tmp_regressor = sm.OLS(y, X).fit()
                    
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    
                    if (adjR_before >= adjR_after):
                        x_rollback = np.hstack((X, temp[:,[0,j]]))
                        
                        x_rollback = np.delete(x_rollback, j, 1)
                        
                        print (regressor_OLS.summary())
                        
                        return x_rollback
                    else:
                        continue
    regressor_OLS.summary()
    return X
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5,6,7,8,9,10,11,12,13,14,15,16,17]]
X_Modeled = backwardElimination(X_opt, SL)
