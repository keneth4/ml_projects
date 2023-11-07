from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, Flatten, Dense, AvgPool2D
from tensorflow.keras.models import Sequential
import numpy as np


def load_train(path):
    train_datagen = ImageDataGenerator(rescale=1.0 / 255)


def create_model(input_shape):
    model = Sequential()
    model.add(
        Conv2D(
            filters=6,
            kernel_size=(5, 5),
            padding='same',
            activation="relu",
            input_shape=input_shape,
        )
    )
    model.add(AvgPool2D(pool_size=(2, 2), strides=None, padding='valid'))
    model.add(
        Conv2D(
            filters=16,
            kernel_size=(5, 5),
            activation="relu",
        )
    )
    model.add(AvgPool2D(pool_size=(2, 2), strides=None, padding='valid'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(12, activation='softmax')) 
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['acc'],
    )

    return model


def train_model(
    model,
    train_data,
    test_data,
    batch_size=32,
    epochs=5,
    steps_per_epoch=None,
    validation_steps=None,
):

    features_train, target_train = train_data
    features_test, target_test = test_data
    model.fit(
        features_train,
        target_train,
        validation_data=(features_test, target_test),
        batch_size=batch_size,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        verbose=2,
        shuffle=True,
    )

    return model