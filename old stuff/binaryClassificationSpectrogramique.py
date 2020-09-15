
##########################################################
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

plt.style.use('classic')
#############################################################
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense
#from keras import backend as K
####################################################
import os
import cv2
from PIL import Image
import numpy as np

# image_directory = 'cell_images/'
image_directory = '/Users/emmanueltoksadeniran/Desktop/BinaryClassication/'
SIZE = 150
dataset = [] 
label = []

interactive_images = os.listdir(image_directory + 'interactive/')
for i, image_name in enumerate(interactive_images):
    if (image_name.split('.')[1] == 'png'):
        image = cv2.imread(image_directory + 'interactive/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((SIZE, SIZE))
        dataset.append(np.array(image))
        label.append(0)

noninteractive_images = os.listdir(image_directory + 'noninteractive/')
for i, image_name in enumerate(noninteractive_images):
    if (image_name.split('.')[1] == 'png'):
        image = cv2.imread(image_directory + 'noninteractive/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((SIZE, SIZE))
        dataset.append(np.array(image))
        label.append(1)

dataset = np.array(dataset)
label = np.array(label)
# print("Dataset array: ", dataset)
# print("Label array: ", label)

from sklearn.model_selection import train_test_split
#from keras.utils import to_categorical

X_train, X_test, y_train, y_test = train_test_split(dataset, label, test_size = 0.20, random_state = 0)

# normalize to decimalize IOT enable convergence
from keras.utils import normalize
X_train = normalize(X_train, axis=1)
X_test = normalize(X_test, axis=1)


#y_train = to_categorical(y_train)
#y_test = to_categorical(y_test)

##############################################

###2 conv and pool layers. with some normalization and drops in between.

INPUT_SHAPE = (SIZE, SIZE, 3)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=INPUT_SHAPE))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), kernel_initializer = 'he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), kernel_initializer = 'he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(1))
model.add(Activation('sigmoid'))  
#Not using softmax for binary classification

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',             #also try adam
              metrics=['accuracy'])

print(model.summary())    
###############################################################  

##########################################################

# history = model.fit(X_train, 
#                          y_train, 
#                          batch_size = 64, 
#                          verbose = 1, 
#                          epochs = 100,      
#                          validation_data=(X_test,y_test),
#                          shuffle = False
#                      )

# model.save('interactivity_model_100epochs.h5')



# #plot the training and validation accuracy and loss at each epoch
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs = range(1, len(loss) + 1)
# plt.plot(epochs, loss, 'y', label='Training loss')
# plt.plot(epochs, val_loss, 'r', label='Validation loss')
# plt.title('Training and validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()


# acc = history.history['acc']
# val_acc = history.history['val_acc']
# plt.plot(epochs, acc, 'y', label='Training acc')
# plt.plot(epochs, val_acc, 'r', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()

#########################################################################################
#Test the model on one image (for 100 epochs)
#img 23 is interactive - correctly predicts near 0 probability
#Img 22, interactive, correctly lables (low value) but relatively high value.
#img 24 is uninfected, correctly predicts as uninfected
#img 26 is interactive but incorrectly gives high value for prediction, uninfected.

# n=24  #Select the index of image to be loaded for testing
# img = X_test[n]
# plt.imshow(img)
# input_img = np.expand_dims(img, axis=0) #Expand dims so the input is (num images, x, y, c)
# print("The prediction for this image is: ", model.predict(input_img))
# print("The actual label for this image is: ", y_test[n])

# Evaluate the model on all test data for accuracy
################################################################

# from keras.models import load_model
# # load model
model = load_model('interactivity_model_100epochs.h5')

#For 100 epochs, giving 100% accuracy on tiny

_, acc = model.evaluate(X_test, y_test)
print("Accuracy = ", (acc * 100.0), "%")

#How do we know how it is doing for interactive vs uninfected? 
################################################################

#Confusion matrix
#We compare labels and plot them based on correct or wrong predictions.
#Since sigmoid outputs probabilities we need to apply threshold to convert to label.

mythreshold=0.908
from sklearn.metrics import confusion_matrix

y_pred = (model.predict(X_test)>= mythreshold).astype(int)
cm=confusion_matrix(y_test, y_pred)  
print(cm)

# #Check the confusion matrix for various thresholds. Which one is good?
# #Need to balance positive, negative, false positive and false negative. 
# #ROC can help identify the right threshold.
# ##################################################################
# """
# Receiver Operating Characteristic (ROC) Curve is a plot that helps us 
# visualize the performance of a binary classifier when the threshold is varied. 
# """
# #ROC
from sklearn.metrics import roc_curve
y_preds = model.predict(X_test).ravel()

fpr, tpr, thresholds = roc_curve(y_test, y_preds)
# plt.figure(1)
# plt.plot([0, 1], [0, 1], 'y--')
# plt.plot(fpr, tpr, marker='.')
# plt.xlabel('False positive rate')
# plt.ylabel('True positive rate')
# plt.title('ROC curve')
# plt.show()

# """
# #One way to find the best threshold once we calculate the true positive 
# and false positive rates is ...
# The optimal cut off point would be where “true positive rate” is high 
# and the “false positive rate” is low. 
# Based on this logic let us find the threshold where tpr-(1-fpr) is zero (or close to 0)
# # """
import pandas as pd
i = np.arange(len(tpr)) 
roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'thresholds' : pd.Series(thresholds, index=i)})
ideal_roc_thresh = roc.iloc[(roc.tf-0).abs().argsort()[:1]]  #Locate the point where the value is close to 0
print("Ideal threshold is: ", ideal_roc_thresh['thresholds']) 

# # #Now use this threshold value in the confusion matrix to visualize the balance
# #between tp, fp, fp, and fn


# #AUC
# #Area under the curve (AUC) for ROC plot can be used to understand hpw well a classifier 
# #is performing. 
# # #% chance that the model can distinguish between positive and negative classes.

# from sklearn.metrics import auc
# auc_value = auc(fpr, tpr)
# print("Area under curve, AUC = ", auc_value)


#########################################


