from tkinter import Tk
from tkinter.filedialog import askopenfilename
from tkinter import messagebox
from keras.models import load_model
import sys
from keras.preprocessing.image import ImageDataGenerator
import shutil
import os

from PIL import Image
import numpy as np

from keras import utils
import numpy




def image_converter(path_to_image):
    pass


if __name__ == '__main__':
    Tk().withdraw()

    try:
        model = load_model('model.h5')
    except:
        messagebox.showerror("Error", "No \"model.h5\" file, please train a new model or place an existing model file in directory with this program and name it \"model.h5\".")
        sys.exit()

    messagebox.showinfo("Welcome!", "Please, select JPG/JPEG image for recognition in the next step. \n\nAvaliable animals: bear, cat, dog, elephant, giraffe, gorilla, owl, parrot, penguin, zebra.")
    filename = askopenfilename()

    if filename and ('jpg' in filename or 'jpeg' in filename):
        print("copying")
        os.makedirs("./user_img/dog")
        shutil.copyfile(filename, "./user_img/dog/tmp_file.jpg")
    else:
        messagebox.showerror("File error!", "Wrong file type (only JPG/JPEG acceptable).")
        sys.exit()


    recognize_batch = ImageDataGenerator().flow_from_directory("user_img/", target_size=(256, 256), classes=['cat', 'dog'], batch_size=2)
    imgs, labels = next(recognize_batch)


    predictions = model.predict_generator(recognize_batch, steps=1, verbose=0)
    print(predictions)

    result = numpy.amax(predictions)
    messagebox.showinfo("Result", "with probability = " + str(round(result * 100, 2)) + "% it is " + "cat" if predictions[0][0] > predictions[0][1] else "dog")

    shutil.rmtree("./user_img/")
