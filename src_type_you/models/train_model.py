#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 22:11:46 2020

@author: sarakohnke
"""
#Set working directory
import os
path="/Users/sarakohnke/Desktop/data_type_you/interim-tocsv/"
os.chdir(path)
os.getcwd()

#Import required packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Import final cleaned dataframe
dataframe=pd.read_csv('dataframe210620.csv',index_col=0)

from sklearn.tree import DecisionTreeRegressor
from pprint import pprint
from sklearn.model_selection import train_test_split

X_rf = dataframe.drop(['Patient ID','A1C (%)','Year'],1)
y_rf = dataframe['A1C (%)']
X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X_rf, y_rf, random_state = 0)

clf_rf = DecisionTreeRegressor(random_state = 0).fit(X_train_rf, y_train_rf)
print('Parameters currently in use:\n')
pprint(clf_rf.get_params())

X_train_rf.to_csv('X_train.csv')
y_train_rf.to_csv('y_train.csv')

from sklearn.model_selection import RandomizedSearchCV


# Number of features to consider at every split - for regressor, none is good
max_features = None

# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)

# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree
bootstrap = True

# Create the random grid
random_grid = {
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf
               
               }
pprint(random_grid)

# Use the random grid to search for best hyperparameters
# First create the base model to tune
clf_rf = DecisionTreeRegressor(random_state=0)

# Random search of parameters, using 3 fold cross validation, 
# search across 10 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = clf_rf, param_distributions = random_grid, n_iter=50,cv = 3, verbose=2, random_state=0, n_jobs = -1)

# Fit the random search model
rf_random.fit(X_train_rf, y_train_rf)
rf_random.best_params_

# Make features and target objects, split train and test data
X_rf2 = dataframe.drop(['Patient ID','A1C (%)','Year'],1)
y_rf2 = dataframe['A1C (%)']
X_train_rf2, X_test_rf2, y_train_rf2, y_test_rf2 = train_test_split(X_rf2, y_rf2, random_state = 0)

#Train model
clf_rf2 = DecisionTreeRegressor(max_depth=10,min_samples_leaf=4,
                min_samples_split=10,random_state = 0).fit(X_train_rf2, y_train_rf2)

# Print r2 score
print('R-squared score (training): {:.3f}'
     .format(clf_rf2.score(X_train_rf2, y_train_rf2)))
print('R-squared score (test): {:.3f}'
     .format(clf_rf2.score(X_test_rf2, y_test_rf2)))

#pickle the model so can upload trained model to app
import pickle
#import bz2
#import _pickle as cPickle
with open('model_pkl.pickle','wb') as output_file:
    pickle.dump(clf_rf2,output_file)
    
# Pickle a file and then compress it into a file with extension 
#def compressed_pickle(model, clf_rf2):
#    with bz2.BZ2File(model + '.bz2', 'w') as f: 
#        cPickle.dump(clf_rf2, f)
 
##file=bz2.BZ2File('c.pkl','w')
#pickle.dump(clf_rf2,sfile)
#compressed_pickle('model',clf_rf2)
    
# to open compressed pickle
#def decompress_pickle(file):
#    model=bz2.BZ2File(file,'rb')
#    model=cPickle.load(model)
#    return model

#model=decompress_pickle('model.pbz2')

#to open normal pickle
with open('model_pkl.pickle','rb') as input_file:
    model=pickle.load(input_file)

    

