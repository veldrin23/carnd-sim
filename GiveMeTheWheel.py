# import libraries
# import keras
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as pimg
import numpy as np
import cv2
import os
from random import sample
from random import shuffle
from sklearn.model_selection import train_test_split

# variables
batch_size = 128
image_dimension_x = 320
image_dimension_y = 160
image_dimension_depth = 3
use_all_three_images = True
# for when I want to use all three images (left, right and centre)
if use_all_three_images:
    image_dimension_depth *= 3
image_shape = (image_dimension_x, image_dimension_y, image_dimension_depth)


# normalize image
def normalize(img):
    return cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)


# grayscale image
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


# image read and process function
def read_and_process_img(file_name, normalize_img=True, grayscale_img=True):
    img = cv2.imread(file_name, cv2.IMREAD_COLOR)
    if normalize_img:
        img = normalize(img)
    if grayscale_img:
        img = grayscale(img)
    return img

shape = 0

# Generator function
# Thanks to Paul Heraty for this
def img_generator(images, angles):
    ii = 0
    while True:
        images_out = np.ndarray(shape=(batch_size, image_dimension_x, image_dimension_y, image_dimension_depth),
                                dtype=float)  # n*x*y*RGG
        angles_out = np.ndarray(shape=batch_size, dtype=float)
        for j in range(batch_size):
            if ii >= len(images):
                shuffle(images)
                ii = 0
            centre = read_and_process_img(images[ii][0])
            left = read_and_process_img(images[ii][1])
            right = read_and_process_img(images[ii][2])
            angle = angles[ii]
            if use_all_three_images:
                images_out[ii] = np.dstack((centre, left, right))
            else:
                images_out[ii] = centre
            angles_out[ii] = angle
            ii += 1
        yield ({'batchnormalization_input_1': images_out}, {'output': angles_out})


# import data

with open('driving_log.csv') as f:
    logs = pd.read_csv(f)
    nb_images = logs.shape[0]
    images_links = np.ndarray(shape=(nb_images, 3), dtype=object)
    angles = np.ndarray(shape=nb_images, dtype=float)
    i = 0
    for q in logs.iterrows():
        images_links[i, 0] = q[1][0]
        images_links[i, 1] = q[1][1]
        images_links[i, 2] = q[1][2]
        angles[i] = q[1][3]
        i += 1


# create (train, validation) and test data
x_train, x_test, y_train, y_test = train_test_split(images_links, angles, test_size=.2, random_state=0)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=.33, random_state=0)








