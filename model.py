# The script used to create and train the model
import csv
import time
import numpy as np
import math
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# from sklearn.utils import shuffle

# Preprocess data
# Find the range of the steering angle
def get_min_max_steering_angle(path):
    f = open(path)
    line = f.readline()
    minimum = 0
    maximum = 0
    while True:
        line = f.readline().strip()
        if line == '':
            return minimum, maximum
        center, left, right, steering, throttle, brake, speed = line.split(', ')
        steering = float(steering)
        minimum = min(minimum, steering)
        maximum = max(maximum, steering)


# Python generator to read in the data file
def generator(path):
    while True:
        f = open(path)
        f.readline()
        for dataLine in f:
            dataLine = dataLine.strip()
            center, left, right, steering, throttle, brake, speed = dataLine.split(', ')
            # yield([center, left, right, steering, throttle, brake, speed])
            yield({'input': center}, {'output': steering})
        f.close()


def print_generator():
    for x in generator('data/driving_log.csv'):
        print(x[0]['input'])
        print(x[1]['output'])
        time.sleep(1)

# print_generator()

minimum, maximum = get_min_max_steering_angle('data/driving_log.csv')
print(minimum)
print(maximum)


# model.fit_generator(generator('data/driving_log.csv'), samples_per_epoch=10000, nb_epoch=10)
