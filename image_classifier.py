# -*- coding: utf-8 -*-

import tensorflow as tf
print(tf.__version__)
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

(X_train, y_train), (X_test, y_test) = mnist.load_data()
class_names = list(map(str, range(10)))

def compare_and_visualize(X_test, y_test, y_pred):
    plt.subplots(constrained_layout=True)
    for i in range(20):
        plt.subplot(4,5,i+1)
        plt.imshow(X_test[i], cmap='gray', interpolation='none')
        plt.title("Label:{}\n Pred:{}".format(
            class_names[y_test[i]], class_names[int(y_pred[i][0])]))
        plt.axis('off')
    plt.show()

import dill as pickle   # Pure pickle fails here
with open('image_classifier_model.h5', 'rb') as f:
    clf = pickle.load(f)

compare_and_visualize(X_test[-20:], y_test[-20:], clf.predict(X_test[-20:]))
