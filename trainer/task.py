

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.utils import to_categorical
from sklearn.cross_validation import train_test_split

import numpy as np
import pandas as pd

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
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

if __name__ == '__main__':
    X = np.array(pd.read_csv("https://storage.googleapis.com/modified-mnist-bucket/x.csv",header=None))
    X = X.reshape(-1, 28, 28, 1)
    Y = np.array(pd.read_csv("https://storage.googleapis.com/modified-mnist-bucket/y.csv",header=None))
    Y = to_categorical(Y, 12)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=8, test_size=0.2)

    model = build_classifier()

    model.fit(x_train, y_train, batch_size=32, epochs=20, verbose=1)

    score = model.evaluate(x_test, y_test, verbose=0)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    model.save('emnist_model.h5')
    #Save model
    upload_blob('modified-mnist-bucket','emnist_model.h5', 'emnist_model.h5')
