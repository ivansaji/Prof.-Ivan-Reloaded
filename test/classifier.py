#Module Imports
import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn import preprocessing, svm 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 

#Reading the dataset
#--------------------------------------------------------------------------------------------
# Taking only the selected two attributes from the dataset 
df = pd.read_csv('./dataset/datafile.csv') 
df_binary = df[['x1', 'x2']] 
  
# Renaming the columns for easier writing of the code 
df_binary.columns = ['Sal', 'Temp'] 
  
# Displaying only the 1st  rows along with the column names 
df_binary.head() 

#--------------------------------------------------------------------------------------------------
# Plotting the data scatter 

sns.lmplot(x ="Sal", y ="Temp", data = df_binary, order = 2, ci = None)