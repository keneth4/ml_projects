from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam


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
    backbone = ResNet50(input_shape=input_shape, weights='imagenet', include_top=False)

    # Make backbone trainable for fine-tuning
    for layer in backbone.layers:
        layer.trainable = True
    
    model = Sequential()
    model.add(backbone)
    model.add(GlobalAveragePooling2D())
    model.add(BatchNormalization())  # Add Batch Normalization
    model.add(Dropout(0.5))          # Add Dropout
    model.add(Dense(128, activation='relu'))  # More complex classifier
    model.add(Dropout(0.5))          # Add another Dropout
    model.add(Dense(12, activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=0.0001),  # Smaller learning rate
                  loss='sparse_categorical_crossentropy',
                  metrics=['acc'])
    
    return model


def train_model(
    model,
    train_data,
    test_data,
    batch_size=None,
    epochs=3,
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