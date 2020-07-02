
# Project 3: Use Deep Learning to Clone Driving Behavior

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
In this project, I have used what I've learned about deep neural networks and convolutional neural networks to clone driving behavior. I trained, validated and tested a model using Keras. The model outputs a steering angle to an autonomous vehicle.

Udacity provided a simulator where I could steer a car around a track for data collection. I used the image data and steering angles to train a neural network and then used this model to drive the car autonomously around the track.

## Instructions
The instructions can be found [here](https://github.com/kkufieta/CarND-BehavioralCloning/blob/master/instructions.md).

## Behavioral Cloning Project

The steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[center]: ./examples/center.jpg "Center image"
[recover_01]: ./examples/recovery_01.jpg "Recovery Image"
[recover_02]: ./examples/recovery_02.jpg "Recovery Image"
[recover_03]: ./examples/recovery_03.jpg "Recovery Image"
[recover_04]: ./examples/recovery_04.jpg "Recovery Image"
[recover_05]: ./examples/recovery_05.jpg "Recovery Image"
[normal]: ./examples/normal.jpg "Normal Image"
[flipped]: ./examples/flipped.jpg "Flipped Image"
[brightness]: ./examples/brightness.jpg "Image with changed brightness"
[cameras]: ./examples/left_center_right.png "Images from all three cameras"
[histogram]: ./examples/histogram.png "Histogram of dataset"

## Rubric Points
View the [rubric points](https://review.udacity.com/#!/rubrics/432/view) for project requirements. In the following I'll address how I've satisfied all requirements.

### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* `model.py` containing the script to create and train the model (though it's just copied code from `model.ipynb` to satisfy the rubric)
* `drive.py` for driving the car in autonomous mode
* `model.h5` containing a trained convolution neural network based on Udacity data
* `writeup_report.md` summarizing the results

Additional files:
* `model.ipynb` containing the script to create and train the model
* `hil_ram_500_200.h5` containing the latest trained convolution neural network based on own data, collected from three drivers (Hil, Ram, me)
* `video_model_h5.mp4` shows the car behavior based on model.h5
* `video_hil_ram_h5.mp4` shows the car behavior based on hil_ram_500_200.h5

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The `model.py` file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The model is implemented in model.py (Lines 513 - 555). It is implemented after the [NVIDIA end-to-end neural network](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) with only small modifications. It has 5 convolution layers with depths between 24 and 64, where the first three are made with a 5x5 filter and a stride of 2, and the last two are made with a 3x3 filter and a stride of 1. They have RELU layers in between them to introduce nonlinearity. (Lines 529 - 543).

The model has four fully connected layers with ELU layers in between to introduce nonlinearity. I added two Dropout layers to reduce overfitting in the model. (Lines 545 - 554).

The input is normalized in the model using a Keras lambda layer (Line 527).

The layers are more detailed (Lines 523 - 555):

1. Normalization layer, Input & Output: 66x200x3
2. Convolution layer (5x5 filter, stride 2, activation: relu), Output: 31x98x24
3. Convolution layer (5x5 filter, stride 2, activation: relu), Output: 14x47x36
4. Convolution layer (5x5 filter, stride 2, activation: relu), Output: 5x22x48
5. Convolution layer (3x3 filter, stride 1, activation: relu), Output: 3x20x64
6. Convolution layer (3x3 filter, stride 1, activation: relu), Output: 1x18x64
7. Flatten, Output: 1152
9. Fully connected (activation: elu), Output: 100
10. Dropout, p = 0.3
11. Fully connected(activation: elu), Output: 50
12. Fully connected(activation: elu), Output: 10
13. Dropout, p = 0.3
14. Fully connected, Output: 1

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 548 and 553). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code lines 565 - 574). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 580).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. 
* When using the Udacity data, I didn't know how the data was recorded. So I relied on using the left & right camera with an adjusted steering angle to introduce recovery data to the model. 
* For the second model where I was using my own data, I recorded both data that showed the car driving as much as possible in the middle, and data that showed the car how to get from the sides of the road back to the middle. I used additionally the data from the left and right cameras, because I didn't think I recorded enough recovery data. 

To create the data, I bought an XBOX 360 controller for $30. I brought in two of my friends who love playing video games, and they helped me record driving data. We drove the car on both tracks, in both directions, to get an equal number of left and right turn data.

The interesting part was that we had vastly different driving styles. Two of us tried to drive as much as possible in the middle, whereas the third driver added aggressive driving behavior such as cutting curves. Interestingly, we could see how our driving behavior was copied from the network.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to have a model that can translate street images into steering angles. The [NVIDIA end-to-end neural network](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) was therefore a great starting point, because it was based on a similar scenario as our project, where images were the input, and steering angles were the output.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. First, I tested my model on only a few images to see if it trained properly or not. I used the mean squared error for training, but the mean absolute error to manually check the performance. I found it easier to understand, since an absolute mean error of 0.1 tells you that the network actually has the tendency to estimate angles with +- 0.1 angle error. 

No matter how many different setups I had, I never managed to get a lower validation mean absolute error than 0.1, so I used that as a threshold for a "well trained model". Later on when I'll revisit the project, I'll try to achieve lower validation errors than that. I did not have problems with overfitting, since I was using Dropout right from the start.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. Particular problems were the dirt road, where my first models didn't manage to see that they should turn around the corner. Other problems might be that the car tended to steer towards water on the bridge or sharp turns.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road, though it is driving like a drunk. I am very motivated to improve the model further and have multiple ideas on how to do that (better data processing, record new data), but that is a task for the future.

#### 2. Final Model Architecture

I did not change the model architecture other than that I added the dropout layers (model.py lines 513 - 555), since it was working well for me. The architecture is visualized in the paper: [NVIDIA end-to-end neural network](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf).

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![Center image][center]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to correct itself if it came too close to the sides of the road. These images show what a recovery looks like starting from the right edge :

![Recovery image][recover_01]
![Recovery image][recover_02]
![Recovery image][recover_03]
![Recovery image][recover_04]
![Recovery image][recover_05]

Then I repeated this process on track two in order to get more data points.

I used also the left and right cameras to add more recovery data. To teach the data to correct towards the middle, I added a steering angle of +0.25 to the left camera images, and a steering angle of -0.25 to the right camera images. Here's an example of the same scene from the perspective of all three cameras.
![Perspective from all three cameras][cameras]

To augment the data set, I also flipped images and angles thinking that this would balance the data for left and right turns, and add more examples for how to turn in curves. Furthermore, I changed the brightness in some images, to prepare the car to deal with changes in brightness or shadows. A third way to add more data with non-zero angles was to keep an image the same, but perturb the angle slightly. For example, here is an image that has then been flipped and where the darkness was changed:

![Original image][normal]
![Flipped image][flipped]
![Changed brightness][brightness]

The original data set consisted of roughly 25000 images & angles. The augmented data set consisted of roughly 116000 images. In order to get the right behavior, I had to downsample the dataset to a desired distribution. A uniform distribution of the data has shown to give the best performance. Interestingly, the recorded data had naturally a uniformal distribution. It is depicted in the next image:
![Histogram][histogram]


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was between 5 and 12 as evidenced by many trials with various data. I ended up with an epoch of 12. I used an adam optimizer so that manually training the learning rate wasn't necessary.

