import numpy as np
from keras.models import Sequential
from keras.models import load_model
from keras.layers.core import Dense, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from matplotlib import pyplot as plt

from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.layers import Activation
import keras
from keras import backend as K
from sklearn.metrics import confusion_matrix
import itertools


def plots(ims, figsize=(12,6), rows=1, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=figsize)
    cols = len(ims) // rows if len(ims) % 2 == 0 else len(ims) // rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i + 1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.show(ims[i], interpolation='None')  #, interpolation=None if interp else 'None'


if __name__ == '__main__':
    train_path = 'founded_images/train'
    valid_path = 'founded_images/valid'
    test_path = 'founded_images/test'

    #train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size=(256, 256), classes=['bear', 'cat', 'dog', 'elephant', 'giraffe', 'gorilla', 'owl', 'parrot', 'penguin', 'zebra'], batch_size=10)
    #valid_batches = ImageDataGenerator().flow_from_directory(valid_path, target_size=(256, 256), classes=['bear', 'cat', 'dog', 'elephant', 'giraffe', 'gorilla', 'owl', 'parrot', 'penguin', 'zebra'], batch_size=10)
    #test_batches = ImageDataGenerator().flow_from_directory(test_path, target_size=(256, 256), classes=['bear', 'cat', 'dog', 'elephant', 'giraffe', 'gorilla', 'owl', 'parrot', 'penguin', 'zebra'], batch_size=10)

    train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size=(256, 256), classes=['cat', 'dog'], batch_size=10)
    valid_batches = ImageDataGenerator().flow_from_directory(valid_path, target_size=(256, 256), classes=['cat', 'dog'], batch_size=4)
    test_batches = ImageDataGenerator().flow_from_directory(test_path, target_size=(256, 256), classes=['cat', 'dog'], batch_size=10)

    imgs, labels = next(train_batches)

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
        Dense(2, activation='softmax')
    ])

    #model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    #model.fit_generator(train_batches, steps_per_epoch=20, validation_data=valid_batches, validation_steps=4, epochs=5, verbose=2)
    #print('saving the model to h5 file...')
    #model.save('model.h5')

    test_imgs, test_labels = next(test_batches)

    del model

    model = load_model('model.h5')

    print('test labels:')
    print(test_labels)

    predictions = model.predict_generator(test_batches, steps=1, verbose=0)
    print('predictions:')
    for el in predictions:
        print("[0. 1.]" if el[0] < 0.5 else "[1. 0.]")
