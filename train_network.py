from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from keras.optimizers import Adam


if __name__ == '__main__':
    train_path = 'founded_images/train'
    valid_path = 'founded_images/valid'
    test_path = 'founded_images/test'

    train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size=(256, 256), classes=['bear', 'cat', 'dog', 'elephant', 'giraffe', 'gorilla', 'owl', 'parrot', 'penguin', 'zebra'], batch_size=10)
    valid_batches = ImageDataGenerator().flow_from_directory(valid_path, target_size=(256, 256), classes=['bear', 'cat', 'dog', 'elephant', 'giraffe', 'gorilla', 'owl', 'parrot', 'penguin', 'zebra'], batch_size=4)
    test_batches = ImageDataGenerator().flow_from_directory(test_path, target_size=(256, 256), classes=['bear', 'cat', 'dog', 'elephant', 'giraffe', 'gorilla', 'owl', 'parrot', 'penguin', 'zebra'], batch_size=10)

    #CNN model
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(256, 256, 3)),
        MaxPooling2D(pool_size=(2,2)),
        BatchNormalization(),

        Conv2D(64, kernel_size=(3,3), activation='relu'),
        MaxPooling2D(pool_size=(2,2)),
        BatchNormalization(),

        Conv2D(64, kernel_size=(3,3), activation='relu'),
        MaxPooling2D(pool_size=(2,2)),
        BatchNormalization(),

        Conv2D(96, kernel_size=(3,3), activation='relu'),
        MaxPooling2D(pool_size=(2,2)),
        BatchNormalization(),

        Conv2D(32, kernel_size=(3,3), activation='relu'),
        MaxPooling2D(pool_size=(2,2)),
        BatchNormalization(),
        Dropout(0.2),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(10, activation='softmax')
    ])

    model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit_generator(train_batches, steps_per_epoch=25, validation_data=valid_batches, validation_steps=20, epochs=28, verbose=2)
    print('saving the model to h5 file...')
    model.save('model.h5')
