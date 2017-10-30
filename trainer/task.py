# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 20:46:30 2017

@author: orujahmadov
"""

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.utils.np_utils import to_categorical

import csv
import sys
import pandas as pd
import numpy as np

def squueze_labels(labels):
    squeezed_labels =

def export_kaggle_results(file_name, results):
    with open(file_name, 'wb') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        filewriter.writerow(['Id', 'Label'])
        index = 1
        for result in results:
            filewriter.writerow([index,result])
            index+=1

def build_cnn():
    # Part 1 - Building the cnn

    # Initializing CNN
    classifier = Sequential()

    # Adding layers

    # Convolution layer
    classifier.add(Conv2D(32, 3, 3, input_shape=(64, 64, 1), activation = 'relu'))

    # Max Pooling layer
    classifier.add(MaxPooling2D(pool_size = (2, 2)))

    # Flattening
    classifier.add(Flatten())

    # Full connection
    classifier.add(Dense(units = 128, activation = 'relu'))
    classifier.add(Dense(units = 82, activation = 'sigmoid'))

    # Compiling the CNN
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    return classifier

if __name__=='__main__':

    url_x = 'https://www.googleapis.com/download/storage/v1/b/modified-mnist-bucket/o/train_x.csv?generation=1509260014086323&alt=media'
    url_y = 'https://www.googleapis.com/download/storage/v1/b/modified-mnist-bucket/o/train_y.csv?generation=1509256324554912&alt=media'
    #url_kaggle = ''
    train_x_file = pd.read_csv(url_x, header=None)
    train_y_file = pd.read_csv(url_y, header=None)

    # Importing Data
    X = np.array(train_x_file.iloc[:])
    X = X.reshape(50000, 64, 64, 1)

    Y = np.array(train_y_file.iloc[:,0])
    Y = to_categorical(Y, 82)

    X_train = X[:40000]
    X_test = X[40000:]
    y_train = Y[:40000]
    y_test = Y[40000:]

    classifier = build_cnn()
    classifier.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=15, batch_size=32)
    score = classifier.evaluate(X_test, y_test, verbose=0)

    print(score[1]*100)

    # Export kaggle results
    # test_kaggle_file = pd.read_csv(url_kaggle, header=None)
    # X_kaggle = np.array(test_kaggle_file.iloc[:])
    # X_kaggle = X_kaggle.reshape(50000, 64, 64, 1)
    # export_kaggle_results("kaggle_results.csv", np.argmax(classifier.predict(X_kaggle)))

    # Save model
    classifier.save('cnn.h5')
