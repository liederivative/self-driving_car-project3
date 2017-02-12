#**Behavioural Cloning** 

**Behavrioual Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behaviour
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* ```model.py``` containing the script to create and train the model
* ```drive.py``` for driving the car in autonomous mode
* ```model.h5``` containing a trained convolution neural network 
* ```model.json``` containing the model of the CNN
* ```utils.py``` containing the preprocess functions for the model
* ```model_video.mp4``` containing a video driving in autonomous mode
* ```writeup_report.md``` summarizing the results

In order to run the model, execute ``` python drive.py model.json ```.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolutional neural network (CNN) with 5x5 , 3x3, 1x1 and 2x2 filter sizes and depths between 8 and 64 (```model.py``` lines 79-92) 

The model includes RELU layers to introduce non-linearity (```model.py ```line 83-87), and the data is normalized in the model using a Keras BatchNormalization layer (```model.py``` line 80). 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (```model.py``` lines 89). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (```model.py``` line 110-112). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (```model.py``` line 99).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of centre lane driving, left and right sides of the road. I also create ''artificial data'' using augmentation in the same images.  

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to augment the images in order to generate new ones and avoid bias values in the angle prediction, this is presented in the Nvidia paper [1] and by Sakmann [2]. More details are presented in section 3. 

My first step was to use a convolution neural network model similar to the Wu [3], I thought this model might be appropriate because I ran out of credit for my AWS instance and this model can be trained in a modest CPU. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model thus I added additional convolutional layers to help the model to detect new features in the image and I increased the dropout rate to 0.5. 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track like driving through the bridge and the curve with view to the river. However, this undesired behaivour could be improved using MaxPooling between the new additon of convolutional layers.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (```model.py``` lines 80-91) consisted of a convolution neural network with the following layers :

-------------------------------------------------------------------
Layer | size 
------------------ | ---------
BatchNormalization|(input = 16x32x1)
Convolution |(5x5, 8)
MaxPooling	|(2x2)
Convolution |(1x1, 16)
MaxPooling|(2x2)
Convolution|(3x3, 32)
MaxPooling|(2x2)
Convolution|(2x2, 64)
MaxPooling|(4x4, stride = 4x4)
Dropout|(0.5)
FullyConnected|(output=1)

Total params: 15,361
Trainable params: 15,329
Non-trainable params: 32
____________________________________________________________________________________________________

The model includes RELU layers to introduce nonlinearity (```model.py``` line 83), and the data is normalized in the model using a Keras BatchNomalization layer (```model.py``` line 80). 


####3. Creation of the Training Set & Training Process

I used the images provided by Udacity, and new data was generated using augmentation. 

To augment the data set, I flipped images and angles thinking that this would balance the angles values , due to the data has a bias to zeros ,which is the most frequent driving behaviour of the track 1. Below are some examples of the augmentation applied. (augmentation functions are in ``` utils.py ``` line 55-137 )


----------
original image
![enter image description here](https://drive.google.com/uc?id=0B3Ji5KWByh0gQ0RPc3owOWRnaW8)
----------
modified brightness
![enter image description here](https://drive.google.com/uc?id=0B3Ji5KWByh0gcVVuMXV4UGtVM2M)
----------
sheared image
![enter image description here](https://drive.google.com/uc?id=0B3Ji5KWByh0gYm9VTW1tNFRia28)
----------
image in HSV coordinates
![enter image description here](https://drive.google.com/uc?id=0B3Ji5KWByh0gb05tc2Jadkh0Qk0)
----------
flipped image
![enter image description here](https://drive.google.com/uc?id=0B3Ji5KWByh0gbkRvWnJrTEl3RU0)
----------
rotated image 
![enter image description here](https://drive.google.com/uc?id=0B3Ji5KWByh0gM25DbjhRVWVyUXc)
----------


The images are the resize to 16x32 and used as input in the model. All the images for the training set are generated using a random selection of the camera position (centre, left, right), with equal probabilities. The same is applied for the augmentation with the exception of shearing that is implemented with a probability of 80% (```utils.py``` line 98) , higher than the others. More stability was presented with higher likelihood of shearing. The angles are modified according to its augmentation. 

The training set contains 39997 samples and for the validation 10000, about a 20%. The numbers of epochs was 100, but an early termination callback is implemented if no improvement is detected in the loss after 3 epochs. At the end just 22 epochs were enough for this model. 

In order to test the trained model I took a small portion (0.01%, ```model.py``` line 61) of the data set and plot a regression to observed how close to the data points are of the ideal model behaviour (line of equality). Below is the mentioned graph. 

![enter image description here](https://drive.google.com/uc?id=0B3Ji5KWByh0gdzh0Ql8tM2ZVcGc)



[3]:https://medium.com/@xslittlegrass/self-driving-car-in-a-simulator-with-a-tiny-neural-network-13d33b871234
[1]:http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
[2]:https://medium.com/@ksakmann/behavioral-cloning-make-a-car-drive-like-yourself-dc6021152713#.gs8wqg60l