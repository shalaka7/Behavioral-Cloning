# **Behavioral Cloning** 




**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/nVidea_data.png "Model Visualization"
[image2]: ./images/centreLaneDriving.jpg "centreLaneDriving"
[image3]: ./images/Figure_1.png "validation Loss Graph"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model_final.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model_final.h5
```

#### 3. Submission code is usable and readable

The model_final.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed 
I decided to try the nvidia Autonomous Car Group model, and the car drove the complete first track after just three training epochs.
My model consists of a convolution neural network with 3x3  and 5x5 filter sizes and depths between 32 and 128 (line 116 - 120)
The model includes RELU layers to introduce nonlinearity  and the data is normalized in the model using a Keras lambda layer.(line 113)

#### 2. Attempts to reduce overfitting in the model

I decided to keep the training epochs low: only three epochs. In addition to that, I split my sample data into training and validation data. 
The model contains Maxpooling techniques  in order to reduce overfitting.
The model was trained and validated on different data sets to ensure that the model was not overfitting . The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.(Line 50 tO 89)

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (line 127).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. Also, the data provided by Udacity, I used the both track data. Provided csv file have three different images: center, left and right cameras.
Each image was used to train the model. We combine all the image to get one image and that image also trained. and also to keep the vehicle driving on the road. I used a combination of center lane driving, 
recovering from the left and right sides of the road.Given data also provide us sterring value which keep vehicle on track.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use a convolution neural network model.
the first track, the car went straight to the lake. I needed to do some pre-processing. A new Lambda layer was introduced to normalize the input images to zero means. This step allows the car to move a bit further, but it didn't get to the first turn. 
Another Cropping layer was introduced, and the first turn was almost there, but not quite because i use only centre images
In order to drive my car more prfectly, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 
To combat the overfitting, I modified the model so that i split the data into batches so all images or samples are properly going to train and we get the validate samples over the trained samples.
The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track to improve the driving behavior in these cases, I have to train my car properly in training mode.
At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The first layer of the network performs image normalization.The normalizer is hard-coded and is not adjusted in the learning process. 
Performing normalization in the network allows the normalization scheme to be altered with the network architecture, and to be accelerated via GPU processing.
A new Lambda layer was introduced to normalize the input images to zero means.This step allows the car to move a bit further, but it didn't get to the first turn. 
The second layer is cropping layer.The Cropping layer might be useful for choosing an area of interest that excludes the sky and/or the hood of the car.
The convolutional layers are designed to perform feature extraction, and are chosen empirically through a series of experiments that vary layer configurations. 
We then use strided convolutions in the first three convolutional layers with a 5×5 kernel, and a non-strided convolution with a 3×3 kernel size in the final two convolutional layers.
We follow the five convolutional layers with three fully connected layers, leading to a final output control value which is the inverse-turning-radius. The fully connected layers are designed to function as a controller for steering, but we noted that by training the system end-to-end, 
it is not possible to make a clean break between which parts of the network function primarily as feature extractor, and which serve as controller.
The final model architecture (model.py lines 112-125) consisted of a convolution neural network with the following layers and layer sizes
Here is a visualization of the architecture :

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]


After the collection process,I have a graph which shows the graph for mean square loss vs epoch which shows the training  and validation data.
I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 .
I used an adam optimizer so that manually training the learning rate wasn't necessary.

![alt text][image3]