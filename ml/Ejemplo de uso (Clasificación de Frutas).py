from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, Flatten, Dense, AvgPool2D, MaxPool2D
from tensorflow.keras.models import Sequential
import numpy as np


def load_train(path):
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        vertical_flip=True,
        horizontal_flip=True,
        rotation_range=90,
        width_shift_range=.2,
        height_shift_range=.2,
    )
    return train_datagen.flow_from_directory(
        path,
        target_size=(150, 150),
        batch_size=16,
        class_mode='sparse',
        subset='training',
        seed=12345
    )


def create_model(input_shape=(150, 150, 3)):
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
    model.add(MaxPool2D(pool_size=(2, 2), strides=None, padding='valid'))
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
    batch_size=None,
    epochs=6,
    steps_per_epoch=None,
    validation_steps=None,
):
    if steps_per_epoch is None:
        steps_per_epoch = len(train_data)
    if validation_steps is None:
        validation_steps = len(test_data)

    model.fit(
        train_data,
        validation_data=test_data,
        batch_size=batch_size,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        verbose=2
    )

    return model