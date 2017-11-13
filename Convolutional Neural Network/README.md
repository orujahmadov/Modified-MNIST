# Modified-MNIST
There are two python files here. Each one is described below.

To train Convolutional Neural Network model, run
- python cnn.py

It will output test accuracy and return model. This python file downloads training data from Google Cloud public link.

To make predictions for kaggle test set, run
- python predictor.py

It will output kaggle.csv which contains results for Kaggle test set. This python file downloads kaggle test set from Google Cloud public link. This python file also downloads model from online google cloud link and use that to make predictions. Model can be download from here: https://storage.googleapis.com/modified-mnist-bucket1/emnist_model98.h5
