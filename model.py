# The script used to create and train the model
import time
import cv2
import numpy as np
import math
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
# from sklearn.utils import shuffle
# from scipy.misc import imresize

# Read in the data
def read_data(path):
    f = open(path)
    line = f.readline()
    # Data exploration: Find the min and max steering angle
    X_data_paths = []
    y_data = []
    while True:
        line = f.readline().strip()
        # If we reach the end of the file, return the data
        if line == '':
            return X_data_paths, y_data

        # Split the data from the line
        center, left, right, steering, throttle, brake, speed = line.split(', ')
        # Preprocess the data: Transform steering angle to float
        # The minimum and maximum of the steering angles are between -1 and 1.
        steering = float(steering)
        y_data.append(steering)

        # Image path
        X_data_paths.append(center)

# Load the data and shuffle it
X_data_paths, y_data = read_data('data/driving_log.csv')
X_data_paths, y_data = shuffle(X_data_paths, y_data)

# Python generator to read in the images
def generator(X_data_paths, y_data):
    i = 0
    while True:
        if i >= len(y_data):
            i = 0
        # Crop the center image so the sky is not in it
        image = mpimg.imread('data/' + X_data_paths[i])
        image = image[40:-20,:]
        # Reduce the size of the image
        # image = imresize(image, 0.5)
        # Resize the image to fit the NVIDIA network
        # The required input is (66, 200, 3)
        image = cv2.resize(image, (200, 66))
        steering_angle = y_data[i]
        i += 1
        yield({'input': image}, {'output': steering_angle})

# Python generator to read in the data file
def file_generator(path):
    while True:
        f = open(path)
        f.readline()
        for dataLine in f:
            # Read in the line with data
            dataLine = dataLine.strip()
            center, left, right, steering, throttle, brake, speed = dataLine.split(', ')

            center = float(center)
            # Crop the center image so the sky is not in it
            image = mpimg.imread('data/' + center)
            image = image[60:,:]
            # Reduce the size of the image
            print(image.size)
            smaller_image = imresize(image, 0.5)
            print(smaller_image.size)

            yield({'input': smaller_image}, {'output': steering})
        f.close()

def plot_generator():
    for x in generator(X_data_paths, y_data):
        print(x[0]['input'])
        print(x[1]['output'])
        image = x[0]['input']
        imgplot = plt.imshow(image)
        plt.show()
        print(x[0]['input'].shape)
        time.sleep(1)

def print_generator():
    for x in generator(X_data_paths, y_data):
        print(x[0]['input'])
        print(x[1]['output'])
        X_train = x[0]['input']
        # The shape should be (50, 160, 3)
        # The input shape to the nvidia network is (66, 200, 3)
        print(X_train.shape)
        # Pad images with 0s
        print(X_train.shape)
        time.sleep(1)


# print_generator()
# plot_generator()

# Network Architecture
# Create the Sequential model
model = Sequential()

# 1st Layer - Add a flatten layer
model.add(Flatten(input_shape=(32, 32, 3)))

# 2nd Layer - Add a fully connected layer
model.add(Dense(100))

# 3rd Layer - Add a ReLU activation layer
model.add(Activation('relu'))

# 4th Layer - Add a fully connected layer
model.add(Dense(60))

# 5th Layer - Add a ReLU activation layer
model.add(Activation('relu'))

# model.fit_generator(generator('data/driving_log.csv'), samples_per_epoch=10000, nb_epoch=10)
