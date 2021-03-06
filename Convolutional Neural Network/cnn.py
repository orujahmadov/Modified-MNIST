import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

from sklearn.cluster import KMeans
import csv
import numpy as np
import pandas as pd
from ast import literal_eval
import copy

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
        index = 0
        for result in results:
            filewriter.writerow([index,result])
            index+=1

def build_classifier():
    X = np.array(pd.read_csv("https://storage.googleapis.com/modified-mnist-bucket/x_b.csv",header=None))
    X = X.reshape(-1, 28, 28)
    X = filter_image(X)
    X = X.reshape(-1, 28, 28, 1)
    Y = np.array(pd.read_csv("https://storage.googleapis.com/modified-mnist-bucket/y_b.csv",header=None))
    Y = Y.reshape(-1,1)

    s = np.arange(X.shape[0])
    np.random.shuffle(s)
    X = X[s]
    Y = Y[s]
    x_train = X[:25000]
    x_test = X[25000:]
    y_train = Y[:25000]
    y_test = Y[25000:]

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

	# convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, 13)
    y_test = keras.utils.to_categorical(y_test, 13)

    epochs = 25

    datagen = ImageDataGenerator(
    rotation_range=45,
    width_shift_range=0.1,
    height_shift_range=0.1
    )

    # compute quantities required for featurewise normalization
    datagen.fit(x_train)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
					 activation='relu',
					 input_shape=(28,28,1)))
    model.add(Dropout(0.2))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.25))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(13, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
				  optimizer=keras.optimizers.Adadelta(),
				  metrics=['accuracy'])

    test_generator = ImageDataGenerator().flow(x_test, y_test, batch_size=32)

    # fits the model on batches with real-time data augmentation:
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
                        steps_per_epoch=len(x_train) / 32, epochs=epochs, validation_data=test_generator, validation_steps=1000)

    score = model.evaluate(x_test, y_test, verbose=0)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    return model

if __name__ == '__main__':

    classifier = build_classifier()
