import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

#-----------------------------------------------------------------------------------------------------------------------------
#Reading the dataset
dataset = pd.read_csv('./dataset/datafile.csv')
#----------------------------------------------------------------------------------------------------------------------------
#Extracting X and Y values
#Preparing the Data
#divide the data into "attributes" and "labels". Attributes are the independent variables while labels are dependent variables whose values are to be predicted

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

#------------------------------------------------------------------------------------------------------------------------------
#converting to arrays of data
X = np.array(X).reshape(-1, 1) 
y = np.array(X).reshape(-1, 1)

#-----------------------------------------------------------------------------------------------------------------------------
#we have our attributes and labels, the next step is to split this data into training and test sets. We'll do this by using Scikit-Learn's built-in train_test_split()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2) 

#The above script splits 80% of the data to training set while 20% of the data to test set. The test_size variable is where we actually specify the proportion of test set.
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Training the Algorithm

regr = LinearRegression() 
regr.fit(X_train, y_train) 
print(regr.score(X_test, y_test)) 

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Making Predictions

y_pred = regr.predict(X_test) 

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Evaluating the Algorithm
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Plotting the scatter
plt.scatter(X_test, y_test, color ='b') 
plt.plot(X_test, y_pred, color ='k') 
plt.show()