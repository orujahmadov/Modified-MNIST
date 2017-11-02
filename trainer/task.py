# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 20:46:30 2017

@author: orujahmadov
"""
import csv
import sys
import pandas as pd
import numpy as np
import os
import pickle

from sklearn.cluster import KMeans
from sklearn.externals import joblib
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


if __name__=='__main__':


    #url_kaggle = 'https://www.googleapis.com/download/storage/v1/b/modified-mnist-bucket/o/test_x.csv?generation=1509329534125830&alt=media'
    url_kaggle = 'trainer/test_x.csv'
    test_kaggle_file = pd.read_csv(url_kaggle, header=None)
    X_kaggle = np.array(test_kaggle_file.iloc[:])
    X_kaggle = X_kaggle.reshape(-1, 64,64)

    for row in range(len(X_kaggle)):
        for column in range(len(X_kaggle[row])):
            for index in range(len(X_kaggle[row][column])):
                if (X_kaggle[row][value][index] < 180):
                    X_kaggle[row][value][index] = 0

    cluster = KMeans(n_clusters=3, random_state=0).fit(X_kaggle[0])

    results = open('cluster_result.txt','w')

    for center in cluster.cluster_centers_:
        for coordinate in center:
            results.write(str(coordinate) + " ")
        results.write("\r\n")
    results.close()

    # Save model
    #upload_blob('modified-mnist-bucket','cluster_result.txt', 'cluster_result.txt')
