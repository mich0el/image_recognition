import numpy as np
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from matplotlib import pyplot as plt
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
        plt.show(ims[i])#, interpolation=None if interp else 'None')


if __name__ == '__main__':
    train_path = 'founded_images/train'
    train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size=(256, 256), classes=['bear', 'cat', 'dog', 'elephant', 'giraffe', 'gorilla', 'owl', 'parrot', 'penguin', 'zebra'], batch_size=10)

    imgs, labels = next(train_batches)
    #plots(imgs, titles=labels)
    for l in labels:
        print(l)
