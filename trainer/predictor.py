from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.utils import to_categorical
import keras

import numpy as np
import csv
from sklearn.cluster import KMeans
import pandas as pd


from ast import literal_eval

import copy


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

def reshape(array):
    reshaped = np.zeros((28,28))

    if (len(array) != 28 or len(array[0]) != 28):
        reshaped[:len(array),:len(array[0])] = array
    else:
        reshaped = array

    return reshaped

def clean_clusters(kmeans, image, x_scale, label):
    copy_image = copy.copy(image)
    copy_scale = copy.copy(x_scale)
    for pixel in copy_scale:
        prediction = kmeans.predict([pixel])
        if prediction[0] != label:
            copy_image[pixel[0]][pixel[1]] = 0
    
    return copy_image
            
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
    centers = kmeans.cluster_centers_.astype('int')
    
    label1 = kmeans.predict([centers[0]])
    label2 = kmeans.predict([centers[1]])
    label3 = kmeans.predict([centers[2]])
    
    cluster_image_1 = clean_clusters(kmeans, image, x_scale, label1[0])
    cluster_image_2 = clean_clusters(kmeans, image, x_scale, label2[0])
    cluster_image_3 = clean_clusters(kmeans, image, x_scale, label3[0])
    
    margin = 14
    
    cluster1 = cluster_image_1[max(0, centers[0][0]-margin):min(64, centers[0][0]+margin), max(0, centers[0][1]-margin):min(64, centers[0][1]+margin)]
    cluster2 = cluster_image_2[max(0, centers[1][0]-margin):min(64, centers[1][0]+margin), max(0, centers[1][1]-margin):min(64, centers[1][1]+margin)]
    cluster3 = cluster_image_3[max(0, centers[2][0]-margin):min(64, centers[2][0]+margin), max(0, centers[2][1]-margin):min(64, centers[2][1]+margin)]
    
    # Reshape all clisters into 28x28
    character1 = reshape(cluster1)
    character2 = reshape(cluster2)
    character3 = reshape(cluster3)

    characters = [character1, character2, character3]

    return characters

def export_results(file_name, header1_name, header2_name, results):
    with open(file_name, 'wb') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        filewriter.writerow([header1_name, header2_name])
        index = 1
        for result in results:
            filewriter.writerow([index,result])
            index+=1

def apply_operation(characters):
    result = 0
    if (12 in characters):
        result = 1
        for character in characters:
            if (character != 12):
                result*=character
    else:
        result = 0
        for character in characters:
            if (character != 10 and character !=11):
                result+=character
    
    return result

def calculate_result(rows):
    letter_count = 0
    results = []
    result = 0
    count1letters = 0
    for row in rows:
        result = 0
        letter_count = 0
        for character in row:
            if character == 10 or character == 11 or character == 12:
                letter_count+=1
                
        if (letter_count == 1):
            count1letters+=1
            result = apply_operation(row)
        results.append(result)
    print(count1letters)
    return results

if __name__=='__main__':
    
    x = np.array(pd.read_csv("https://storage.googleapis.com/modified-mnist-bucket/test_x.csv", header=None))
    x = x.reshape(-1,64,64)
    x = filter_image(x)
    x = x.astype('float32')
    x /= 255
    
    x_original = np.array(pd.read_csv("trainer/test_x.csv", header=None))
    x_original = x_original.reshape(-1,64,64)
    
    import urllib
    urllib.urlretrieve ("https://storage.googleapis.com/modified-mnist-bucket1/emnist_model98.h5", "emnist_model98.h5")
    
    from keras.models import load_model
    model = load_model('emnist_model98.h5')
    
    index = 212
    plt.imshow(np.uint8(x[index]),cmap='gray')
    plt.show()
    
    plt.imshow(np.uint8(x_original[index]),cmap='gray')
    plt.show()
    
    segments = segment(x[index])
    plt.imshow(np.uint8(segments[0]), cmap='gray')
    plt.show()
    plt.imshow(np.uint8(segments[1]), cmap='gray')
    plt.show()
    plt.imshow(np.uint8(segments[2]), cmap='gray')
    plt.show()
    
    segments = np.array(segments)
    segments = segments.astype('float32')
    segments /= 255
    
    
#    predictions = model.predict(segments[2].reshape(-1,28,28,1), verbose=1)
#    
#    print(predictions[0])
    
    results = []
    for image in x:
        segments = segment(image)            
        character1 = np.argmax(model.predict(segments[0].reshape(-1,28,28,1)))
        character2 = np.argmax(model.predict(segments[1].reshape(-1,28,28,1)))
        character3 = np.argmax(model.predict(segments[2].reshape(-1,28,28,1)))
        results.append([character1, character2, character3])
        
    export_results("results_seg.csv",'Id', 'Label', results)
    
    results = pd.read_csv("results_seg.csv", delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
    
    results.Label = results.Label.apply(literal_eval)
    results = results.iloc[:,1]
    results = np.array(results.tolist())
    export_results('kaggle_seg.csv', 'Id', 'Label', calculate_result(results))
    upload_blob('modified-mnist-bucket1','kaggle_seg.csv', 'kaggle_seg.csv')
    
#    segments_copy = np.array(segments)
#    segments_copy = segments_copy.astype('float32')
#    segments_copy /= 255
    
    



    