#!/usr/bin/python3
import pandas
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

tf.random.set_seed(42)  # fix the random state


"""-------------------LeNet-------------------------
usage: used to compile LeNet model implementation
shape: shape of input data, defaults to standard LeNet dimensions
return value: compiled model 
"""


def LeNet(shape=(28, 28, 1)):
    model = keras.Sequential()  # initialize model

    model.add(layers.Conv2D(20, (5, 5), padding="same", activation='relu', input_shape=shape))
    model.add(layers.MaxPooling2D((2, 2), strides=2))

    model.add(layers.Conv2D(50, (5, 5), padding="same", activation='relu'))
    model.add(layers.MaxPooling2D((2, 2), strides=2))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(500, activation="relu"))

    model.add(layers.Dense(1000, activation="relu"))

    model.add(layers.Dense(1500, activation="relu"))

    model.add(layers.Dense(1, activation="sigmoid"))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # compile model

    return model


# load data
dataset = pandas.read_csv('datasets/data_shuffled.csv', header=None)[1:]
del(dataset[0])

# split data into training and test sets
features, target = dataset.iloc[:, 0:-1], dataset.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = np.array(X_train).reshape((X_train.shape[0], 28, 28, 1)), np.array(X_test).reshape((X_test.shape[0], 28, 28, 1)), np.array(y_train), np.array(y_test)

# train model
compiled_model = LeNet()
compiled_model.fit(X_train, y_train, verbose=0)
compiled_model.save('fd_model')

# print out model accuracy
loss, accuracy = compiled_model.evaluate(X_test, y_test, verbose=0)
print(accuracy)
