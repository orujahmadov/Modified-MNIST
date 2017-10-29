#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 16:52:38 2017

@author: orujahmadov
"""

# Importing the libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

def preprocess_data(file_x, file_y):
    # Importing the dataset
    dataset_x = pd.read_csv("train_x.csv", header=None)
    X = dataset_x.iloc[:,1]
    dataset_y = pd.read_csv('train_y.csv', header=None)
    Y = dataset_y.iloc[:,0]
    
    
classifier = LogisticeRegression()
classifier.fit(X_train, y_train)