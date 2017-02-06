# The script used to create and train the model
import pdb
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
from keras.layers.core import Lambda, Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.advanced_activations import ELU
# from sklearn.utils import shuffle
# from scipy.misc import imresize

batch_size = 3
samples_per_epoch = 24
nb_epoch = 20

# Set the font in the plots
font = {'size'   : 7}
plt.rc('font', **font)

# Read in the data
def read_data_from_file(path):
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
        # Append the steering angle to the output data array
        y_data.append(steering)

        # Append the image path to the input data array
        X_data_paths.append(center)

# Load the data and shuffle it
X_data_paths, y_data = read_data_from_file('data_overfitting/driving_log.csv')
X_data_paths, y_data = shuffle(X_data_paths, y_data)

# Python generator to read in the images
def generator(X_data_paths, y_data):
    j = 0
    num_examples = len(y_data)
    while True:
        if j >= num_examples:
            j = 0
            X_data_paths, y_data = shuffle(X_data_paths, y_data)
        for offset in range(0, num_examples, batch_size):
            end = offset + batch_size
            batch_x_paths, batch_y = X_data_paths[offset:end], y_data[offset:end]

            batch_x = []
            for i in range(len(batch_x_paths)):
                # Crop the center image so the sky is not in it
                image = mpimg.imread('data/' + batch_x_paths[i])
                image = image[40:-20,:]
                # Reduce the size of the image
                # image = imresize(image, 0.5)
                # Resize the image to fit the NVIDIA network
                # The required input is (66, 200, 3)
                image = cv2.resize(image, (200, 66))
                batch_x.append(image)

            # steering_angle = y_data[i]

            j += 1
            # yield({'input': image}, {'output': steering_angle})
            # pdb.set_trace()
            batch_x = np.array(batch_x)
            batch_y = np.array(batch_y)
            # yield({'input': batch_x}, {'output': batch_y})
            yield(batch_x, batch_y)

# For debugging purposes
def plot_generator():
    for x in generator(X_data_paths, y_data):
        print(x[0]['input'][0])
        print(x[1]['output'][0])
        image = x[0]['input'][0]
        imgplot = plt.imshow(image)
        plt.show()
        print(x[0]['input'][0].shape)
        time.sleep(1)

def print_generator():
    for x in generator(X_data_paths, y_data):
        print(x[0]['input'])
        print(x[1]['output'])
        X_train = x[0]['input']
        # The shape should be (50, 160, 3)
        # The input shape to the nvidia network is (66, 200, 3)
        print(X_train[0].shape)
        time.sleep(1)

# print_generator()
# plot_generator()

# Network Architecture
# Create the Sequential model
def nvidia_model():
    model = Sequential()

    # 1st Layer - Normalize the image to values between [0.5, 0.5]
    model.add(Lambda(lambda x: -1 + x/127.5, input_shape=(66, 200, 3)))

    # 3 layers of Convolution with 5x5 kernel and 2x2 stride
    # init='he_normal': Initialization of random weights with Gaussian
    # initialization
    model.add(Convolution2D(24, 5, 5, activation='relu', subsample=(2,2),
                            border_mode="valid", init='he_normal'))
    model.add(Convolution2D(36, 5, 5, activation='relu', subsample=(2,2),
                            border_mode="valid", init='he_normal'))
    model.add(Convolution2D(48, 5, 5, activation='relu', subsample=(2,2),
                            border_mode="valid", init='he_normal'))

    # 2 layers of Convolution with 3x3 kernel and no stride
    model.add(Convolution2D(64, 3, 3, activation='relu', subsample=(1,1),
                            border_mode="valid", init='he_normal'))
    model.add(Convolution2D(64, 3, 3, activation='relu', subsample=(1,1),
                            border_mode="valid", init='he_normal'))

    model.add(Flatten())
    model.add(Dense(1164, init='he_normal'))
    model.add(ELU())
    model.add(Dense(100, init='he_normal'))
    model.add(ELU())
    model.add(Dense(50, init='he_normal'))
    model.add(ELU())
    model.add(Dense(10, init='he_normal'))
    model.add(ELU())
    model.add(Dense(1, init='he_normal'))
    return model

def simple_model():
    model = Sequential()
    # 1st Layer - Normalize the image to values between [0.5, 0.5]
    model.add(Lambda(lambda x: -0.5 + x/255., input_shape=(66, 200, 3)))
    # model.add(Convolution2D(24, 5, 5, subsample=(2,2), border_mode="same", input_shape=(66, 200, 3)))
    # model.summary()
    model.add(Flatten())
    model.add(Dense(1, init='he_normal'))
    return model

model = nvidia_model()
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
history = model.fit_generator(generator(X_data_paths, y_data), samples_per_epoch=samples_per_epoch, nb_epoch=nb_epoch)


X_data_paths, y_data = shuffle(X_data_paths, y_data)
X_data = []
for i in range(len(y_data)):
    image = mpimg.imread('data/' + X_data_paths[i])
    image = image[40:-20,:]
    # Reduce the size of the image
    # image = imresize(image, 0.5)
    # Resize the image to fit the NVIDIA network
    # The required input is (66, 200, 3)
    image = cv2.resize(image, (200, 66))
    X_data.append(image)

X_data = np.array(X_data)
y_data = np.array(y_data)

# metrics = model.evaluate(X_data, y_data)
steering_angle = model.predict(X_data, batch_size=3)
print(steering_angle)

num_examples = len(y_data)
for i in range(num_examples):
    plt.subplot(num_examples, 1, i+1)
    imgplot = plt.imshow(X_data[i])
    plt.title('Steering angle: ' + str(steering_angle[i]))
    # plt.show()
batch_size = 3
samples_per_epoch = 24
nb_epoch = 20
plt.savefig('batch_size=' + str(batch_size) + 'samples_per_epoch=' + str(samples_per_epoch) + 'nb_epoch=' + str(nb_epoch) + '.pdf')
