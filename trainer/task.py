# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 20:46:30 2017

@author: orujahmadov
"""
from PIL import Image
import csv
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt

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

def filter_image(image):
    margin_pixel_intensity = 220
    for i in range(len(image)):
        for j in range(len(image[i])):
            for z in range(len(image[i][j])):
                if image[i][j][z] < margin_pixel_intensity:
                    image[i][j][z] = 0    
    
    return image

def segment(image):
    x_scale = []
    for row in range(len(image)):
        for column in range(len(image[row])):
            if image[row][column] != 0:
                coordinates = []
                coordinates.append(row)
                coordinates.append(column)
                x_scale.append(coordinates)
        
    kmeans = KMeans(n_clusters=3, random_state=0).fit(x_scale)
    centers = kmeans.cluster_centers_
    margin = 20
    cluster1 = image[max(0, int(centers[0][0])-margin):min(64, int(centers[0][0])+margin), max(0, int(centers[0][1])-margin):min(64, int(centers[0][1])+margin)]
    cluster2 = image[max(0, int(centers[1][0])-margin):min(64, int(centers[1][0])+margin), max(0, int(centers[1][1])-margin):min(64, int(centers[1][1])+margin)]
    cluster3 = image[max(0, int(centers[2][0])-margin):min(64, int(centers[2][0])+margin), max(0, int(centers[2][1])-margin):min(64, int(centers[2][1])+margin)]
    
    clusters = [cluster1, cluster2, cluster3]
    
    return clusters

def export_results(file_name, header1_name, header2_name, results):
    with open(file_name, 'wb') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        filewriter.writerow([header1_name, header2_name])
        index = 0
        for result in results:
            filewriter.writerow([index,result])
            index+=1
            
if __name__=='__main__':


    #url_kaggle = 'https://www.googleapis.com/download/storage/v1/b/modified-mnist-bucket/o/test_x.csv?generation=1509329534125830&alt=media'
    import os, glob
    os.chdir(r'C:\Users\orujahmadov\Desktop\Modified-MNIST\letters\A')
    with open('A.csv', 'w') as csvfile:
        for file in glob.glob("*.png"):
            im = Image.open(file)
            im = im.resize((48,48))
            pixels = np.array(list(im.getdata()))
            pixels = pixels[:,3]
            pixels = pixels.reshape(2304)
            filewriter = csv.writer(csvfile)
            filewriter.writerow(pixels.tolist())
    import pandas as pd
    data_M = np.array(pd.read_csv('M.csv', header=None))
    data_A = np.array(pd.read_csv('A.csv', header=None))
    data = np.vstack((data_A, data_M))
    label_A = np.ones((len(data_A),))
    label_M = np.zeros((469,))
    labels = np.concatenate((label_A, label_M))
    plt.imshow(np.uint8(pixels))
    plt.show()
    # Save model
    #upload_blob('modified-mnist-bucket','cluster_result.txt', 'cluster_result.txt')
