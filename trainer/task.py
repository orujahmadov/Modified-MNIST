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
import os

from sklearn.preprocessing import LabelEncoder
from sklearn import metrics

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    from google.cloud import storage
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print('File {} uploaded to {}.'.format(
        source_file_name,
        destination_blob_name))

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
    classifier.add(Dense(units = 40, activation = 'sigmoid'))

    # Compiling the CNN
    classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

    return classifier

if __name__=='__main__':

    url_x = 'https://www.googleapis.com/download/storage/v1/b/modified-mnist-bucket/o/train_x.csv?generation=1509260014086323&alt=media'
    url_y = 'https://www.googleapis.com/download/storage/v1/b/modified-mnist-bucket/o/train_y.csv?generation=1509256324554912&alt=media'
    url_kaggle = 'https://www.googleapis.com/download/storage/v1/b/modified-mnist-bucket/o/test_x.csv?generation=1509329534125830&alt=media'
    train_x_file = pd.read_csv(url_x, header=None)
    train_y_file = pd.read_csv(url_y, header=None)

    # Importing Data
    X = np.array(train_x_file.iloc[:])
    X = X.reshape(-1, 64, 64, 1)

    Y = np.array(train_y_file.iloc[:,0])

    labelEncoder = LabelEncoder()
    labelEncoder.fit(Y)
    Y = labelEncoder.transform(Y)
    Y = to_categorical(Y, 40)

    X_train = X[:40000]
    X_test = X[40000:]
    y_train = Y[:40000]
    y_test = Y[40000:]

    classifier = build_cnn()
    classifier.fit(X_train, y_train, epochs=100, batch_size=32)

    test_kaggle_file = pd.read_csv(url_kaggle, header=None)
    X_kaggle = np.array(test_kaggle_file.iloc[:])
    X_kaggle = X_kaggle.reshape(-1, 64, 64, 1)
    predictions = np.argmax(classifier.predict(X_kaggle), axis=1)
    predictions = labelEncoder.inverse_transform(predictions)
    export_kaggle_results("kaggle_results.csv", predictions)
    upload_blob("modified-mnist-bucket","kaggle_results.csv", "kaggle.csv")
    # Save model
    classifier.save('cnn.h5')
    upload_blob('modified-mnist-bucket','cnn.h5', 'cnn.h5')
