# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 14:56:27 2020

@author: matcc
classifiy images of clothing (tensor flow/keras tutorial)
"""

from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

import os

print(tf.__version__)

os.chdir("M:\LIDA_internship\python_training\machine_learning")

#%% #import and load Fashion MNIST data directly from TensorFlow

#images = 28 x 28 numpy pixel arrays
#labels = integers from 0-9, corresponding to the clothing class that the image represents
#0 - top, 1 - trouser, 2 - pullover, 3 - dress, 4 - coat, 5 - sandal, 6 - shirt, 7 - trainer, 8 - bag, 9 - boot

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#%%

#store class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#preprocess the data
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

#scale the training and testing sets
train_images = train_images/255.0
test_images = test_images/255.0

#%% display first 25 images from the training set and display the class name 

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

#%% build the model
#build the neural netwrok - configure the model layers, then compile the model 

#1) set up the layers
# the layer is the basic building block of a neural network. 
# layers extract represnetations from the data fed into them
# deep learning - mostly consists of chaining simple layers together
# most layers have parameters that are learned during training

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)), #transfrom format from 2D (28 x 28) to 1D array (28*28 = 784 pixels)
    keras.layers.Dense(128, activation='relu'), #128 neurons
    keras.layers.Dense(10, activation='softmax') #10-neuron softmax layer. Returns array of 10 probability scores that sum to 1
])
#(dense layer = densely/fully connected)
    
#%% compile the model

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])    

#optimizer - how the model is updated based on the data it sees, and its loss function

#loss function - measures how accurate the model is during training. This should be minimised to steer the model in the right direction .

#metrics - used to monitor the training and testing steps. Accuracy = fraction of images that are correctly classified

#%% train the model

#1) feed in the training data
#2) the model learns to associate images and labels
#3) model makes prediction about the test set
#4) veryfiy that the predictions match the real results

#feed the model
model.fit(train_images, train_labels, epochs=10)

#%% evaluate accuracy

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

print('\nTest accuracy:', test_acc)

#accuracy on test dataset is less than accuracy on the training dataset - overfitting
#overfitting - when a ML model performs worse on new, previously unseen inputs than on the training data (it 'memorises' the training data)

#%% make predictions

predictions = model.predict(test_images) #returns a list of lists

#prediction = array of 10 numbers, representing the confidence that the image corresponds to each of the 10 different items

predict1 = np.argmax(predictions[0]) #which label has the highest confidence value for the first image
actual1 = test_labels[0] #the actual label for the first image

#%% visualise predictions

#function to plot image with prediction in caption

def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array, true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

#function to plot histogram of predictions
    
def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array, true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')


#%% plots
  
i = 25
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()


# Plot the first X test images, their predicted labels, and the true labels.
# Colour correct predictions in blue and incorrect predictions in red.
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()

#%% use the trained model to make a prediction about a single image
#tf.keras models are optimised to make predictions on a batch at once.

img0 = test_images[1] #select image

#add image to a batch (its the only member)
img = (np.expand_dims(img0,0))

#predict the label for this image:
predictions_single = model.predict(img)


plot_value_array(1, predictions_single[0], test_labels)

plt.xticks(range(10), class_names, rotation=45)

#%% find which items are misidentified most frequently

misidentified = []
for i in range(len(predictions)):
    predict_i = np.argmax(predictions[i])
    actual_i = test_labels[i]
    if predict_i != actual_i:
        misidentified.append(actual_i) #array of all misidentified items of clothing
    
#%%plot misidentified items 

#fraction_misid = list(range(0,10))
#for i in range(len(fraction_misid)):
fraction_misid = [100*(misidentified.count(i)/len(misidentified)) for i in range(10)]      

plt.bar(range(10), fraction_misid)

plt.xticks(range(10), class_names, rotation=45)








 