

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.utils import to_categorical

import numpy as np
import pandas as pd

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

def build_classifier():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(12, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    return model

if __name__ == '__main__':
    X = np.array(pd.read_csv("https://storage.googleapis.com/modified-mnist-bucket/x.csv",header=None))
    X = X.reshape(-1, 28, 28, 1)
    Y = np.array(pd.read_csv("https://storage.googleapis.com/modified-mnist-bucket/y.csv",header=None))
    Y = Y.reshape(-1,1)
    Y = to_categorical(Y, 12)

    x_train = X[:200000]
    x_test = X[200000:]
    y_train = Y[:200000]
    y_test = Y[200000:]

    model = build_classifier()

    model.fit(x_train, y_train, batch_size=32, epochs=20, verbose=1)

    score = model.evaluate(x_test, y_test, verbose=0)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    model.save('emnist_model.h5')
    #Save model
    upload_blob('modified-mnist-bucket','emnist_model.h5', 'emnist_model.h5')
