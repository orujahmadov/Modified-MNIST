# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 20:46:30 2017

@author: orujahmadov
"""

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

import csv
import sys
import pandas as pd
import numpy as np

def export_kaggle_results(file_name, header1_name, header2_name, results):
    with open(file_name, 'wb') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        filewriter.writerow([header1_name, header2_name])
        index = 0
        for result in results:
            filewriter.writerow([index,result])
            index+=1

def build_cnn():
    # Part 1 - Building the cnn

    # Initializing CNN
    classifier = Sequential()

    # Adding layers

    # Convolution layer
    classifier.add(Convolution2D(32, 3, 3, input_shape=(64, 64, 1), activation = 'relu'))

    # Max Pooling layer
    classifier.add(MaxPooling2D(pool_size = (2, 2)))

    # Flattening
    classifier.add(Flatten())

    # Full connection
    classifier.add(Dense(units = 128, activation = 'relu'))
    classifier.add(Dense(units = 40, activation = 'sigmoid'))

    # Compiling the CNN
    classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

    return classifier

if __name__=='__main__':

    url_x = 'https://www.googleapis.com/download/storage/v1/b/modified-mnist-bucket/o/train_x.csv?generation=1509260014086323&alt=media'
    url_y = 'https://www.googleapis.com/download/storage/v1/b/modified-mnist-bucket/o/train_y.csv?generation=1509256324554912&alt=media'

    train_x_file = pd.read_csv(url_x, header=None)
    train_y_file = pd.read_csv(url_y, header=None)

    # Importing Data
    X = np.array(train_x_file.iloc[:])
    X = X.reshape(10000, 64, 64, 1)

    Y = np.array(train_y_file.iloc[:,0])
    Y = Y.reshape(-1, 1)

    X_train = X[:40000]
    X_test = X[40000:]
    y_train = Y[:40000]
    y_test = Y[40000:]

    classifier = build_cnn()
    classifier.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32)
    score = classifier.evaluate(X_test, y_test, verbose=0)

    print(score[1]*100)

    # Save model
    classifier.save('cnn.h5')
