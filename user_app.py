from tkinter import Tk
from tkinter.filedialog import askopenfilename
from tkinter import messagebox
from keras.models import load_model
import sys
from keras.preprocessing.image import ImageDataGenerator
import shutil
import os
import cv2
import numpy


if __name__ == '__main__':
    Tk().withdraw()

    #load a model
    try:
        model = load_model('model.h5')
    except:
        messagebox.showerror("Error", "No \"model.h5\" file, please train a new model or place an existing model file in directory with this program and name it \"model.h5\".")
        sys.exit()

    #read file path
    messagebox.showinfo("Welcome!", "Please, select JPG/JPEG image for recognition in the next step. \n\nAvaliable animals: bear, cat, dog, elephant, giraffe, gorilla, owl, parrot, penguin, zebra.")
    filename = askopenfilename()

    #save resized tmp image
    if filename and ('jpg' in filename or 'jpeg' or 'png' in filename):
        os.makedirs("./user_img/dog")
        img = cv2.imread(filename)
        resized_img = cv2.resize(img, (256, 256))
        cv2.imwrite("./user_img/dog/tmp_fildddde.jpg", resized_img)
    else:
        messagebox.showerror("File error!", "Wrong file type (only JPG/JPEG acceptable).")
        sys.exit()

    recognize_batch = ImageDataGenerator().flow_from_directory("user_img/", target_size=(256, 256), classes=['bear', 'cat', 'dog', 'elephant', 'giraffe', 'gorilla', 'owl', 'parrot', 'penguin', 'zebra'], batch_size=1)

    #names for result messagebox
    animal_strings = [
        'bear',
        'cat',
        'dog',
        'elephant',
        'giraffe',
        'gorilla',
        'owl',
        'parrot',
        'penguin',
        'zebra'
    ]

    predictions = model.predict_generator(recognize_batch, steps=1, verbose=1)
    max_index = numpy.where(predictions == numpy.amax(predictions))

    messagebox.showinfo("Result", "With probability = " + str(round(numpy.amax(predictions) * 100, 2)) + "% it is: " + animal_strings[int(max_index[1])])

    #removing temporary files and folders
    shutil.rmtree("./user_img/")
