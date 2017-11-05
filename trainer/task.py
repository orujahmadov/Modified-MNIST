import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.utils import to_categorical

import numpy as np

import pandas as pd

from keras.models import load_model

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
    margin = 14
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

def predict(image_segment):
    return "A"

if __name__=='__main__':
    x = np.array(pd.read_csv("test_x.csv", header=None))
    x = x.reshape(-1,64,64)
    x = filter_image(x)
    y = np.array(pd.read_csv("y.csv", header=None)).reshape(-1,1)
    for image in x:
        image = segment(image)
        character1 = predict(image[0])
        character2 = predict(image[1])
        character3 = predict(image[2])
