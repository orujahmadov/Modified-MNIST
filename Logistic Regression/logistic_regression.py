
import csv
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split

def filter_image(image):
    margin_pixel_intensity = 220
    for i in range(len(image)):
        for j in range(len(image[i])):
            for z in range(len(image[i][j])):
                if image[i][j][z] < margin_pixel_intensity:
                    image[i][j][z] = 0

    return image

def build_classifier():
    X = np.array(pd.read_csv("https://storage.googleapis.com/modified-mnist-bucket/train_x.csv",header=None))
    X = X.reshape(-1, 64, 64)
    X = filter_image(X)
    X = X.reshape(-1, 4096)
    Y = np.array(pd.read_csv("https://storage.googleapis.com/modified-mnist-bucket/train_y.csv",header=None))
    Y = Y.reshape(-1,1)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=551)

    lr_model = LogisticRegression()
    lr_model.fit(X_train, y_train)

    print("Test accuracy is ", lr_model.score(X_test, y_test))

    return lr_model

if __name__ == '__main__':

    classifier = build_classifier()
