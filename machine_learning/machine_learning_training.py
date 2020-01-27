# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 13:15:19 2020

@author: matcc
introduction to machine learning, from Python training course
"""
import os 
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn import linear_model
from sklearn.model_selection import train_test_split 
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

os.chdir("M:\LIDA_internship\python_training\machine_learning")

#import data
countries_df = pd.read_csv("..\data\countries of the world.csv")

#%%linear regression
#drop nans

new_countries_df = countries_df.dropna().copy()

#independent variables
ind_vars = new_countries_df.loc[:,['Population', 'Net migration', 'GDP ($ per capita)', 'Literacy (%)', 'Birthrate', 'Climate']]

#dependent variable
dep_var = new_countries_df.loc[:, ['Infant mortality (per 1000 births)']]

#fit the ordinary least squares (OLS) model
model = sm.OLS(dep_var, ind_vars)
results = model.fit()

#%% linear regression - machine learning approach

#fit the model with the dep. var. and ind. vars. 
lm = linear_model.LinearRegression()
model = lm.fit(ind_vars,dep_var)

#estimate the dependent variable
predictions = lm.predict(ind_vars)

#%% testing the model (testing and training set)

#create training and testing sets:
ind_vars_train, ind_vars_test, dep_var_train, dep_var_test = train_test_split(ind_vars, dep_var, test_size = 0.2)

#fit the model on the training data
lm = linear_model.LinearRegression()
model = lm.fit(ind_vars_train,dep_var_train)
predictions = lm.predict(ind_vars_test)

#plot predictions
plt.scatter(dep_var_test,predictions)
plt.xlabel('True values')
plt.ylabel('Prediction')

#accuracy score
print("Score:", model.score(ind_vars_test,dep_var_test))

#%% artificial neural networks (ANN)
#multi-layer perceptron (MLP) supervised learning

#rescale data (normalised data makes it easier for the ANN to converge)
scaler = StandardScaler()
#fit to the training data
scaler.fit(ind_vars_train)
#apply transformations to data
X_train = scaler.transform(ind_vars_train)
X_test = scaler.transform(ind_vars_test)

#define the solver and the hidden layer sizes
#lbfgs = optimizer in quasi-Newton methods family (best suited for small data sets)
#hidden_layer_sizes - how many layers to include, and the number of neurons at each layer 
#no need for > 1 in this case, as data isn't inherently hierarchical
#we have chosen the same number of neurons as the number of ind vars
mlp = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(5))

#two layer example
#mlp = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(5,5))

#fit the MLP model
#mlp.fit(ind_vars_train,dep_var_train.values.ravel())
#with scaled inputs:
Y_train = dep_var_train.values.ravel().astype('int')
mlp.fit(ind_vars_train,Y_train)
#.values will give the values in an array. (shape: (n,1)
#.ravel will convert that array shape to (n,) 

predictions = mlp.predict(ind_vars_test)

#how good is the model?
print(confusion_matrix(dep_var_test,predictions))
print(classification_report(dep_var_test,predictions))
mlp.coefs_
mlp.intercepts_







