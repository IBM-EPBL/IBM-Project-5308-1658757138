import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras.models
import cv2
from keras import layers

from keras.models import Sequential
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

def init():
    tf.compat.v1.disable_eager_execution()
    num_classes = 10
    img_rows, img_cols = 28, 28
    input_shape = (img_rows, img_cols, 1)
    # model = Sequential()

    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    # model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    #
    # model.add(Conv2D(64, (3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))
    # model.add(Flatten())
    #
    # model.add(Dense(128, activation='relu'))
    # model.add(Dropout(0.5))
    #
    # model.add(Dense(num_classes, activation='softmax'))

    # model.add(Conv2D(64, (3, 3), input_shape=(28, 28, 1), activation="relu"))
    # model.add(Conv2D(32, (3, 3), activation="relu"))
    # model.add(MaxPooling2D((2, 2)))
    # model.add(Flatten())
    # model.add(Dense(num_classes, activation="softmax"))
    #

    # load woeights into new model


    model.load_weights("Detector.h5")
    print("Loaded Model from disk")
    print(model.summary())

    # compile and evaluate loaded model
    model.compile(loss="categorical_crossentropy",optimizer="Adam",metrics=["accuracy"]) #(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])

    # loss,accuracy = model.evaluate(X_test,y_test)
    # print('loss:', loss)
    # print('accuracy:', accuracy)

    graph = tf.compat.v1.get_default_graph()

    return model, graph
