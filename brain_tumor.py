# 1603110201257 - Raisa Tasnim
# 1703210201334 - Mehanaz Chowdhury
# 1703210201340 - Fatin Noor

!nvidia-smi

from google.colab import drive
drive.mount('/content/drive')

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np

%matplotlib inline

# Re-Shapning Training and Test Data
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = .2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

!unzip -uq "/content/drive/MyDrive/NNFLL Final Project (Brain Tumor Detection)/Brain_Tumor_Detection_Dataset.zip" -d "/content"

# Getting Train and Test data from designated folder
print('Getting the Training Data...')
training_set = train_datagen.flow_from_directory('/content/Brain_Tumor_Detection_Dataset/training set',
                                                 target_size = (64, 64),
                                                 shuffle = False,
                                                 class_mode='categorical')

print('Getting the Test Data...')
test_set = test_datagen.flow_from_directory('/content/Brain_Tumor_Detection_Dataset/test set',
                                                 target_size = (64, 64),
                                                 shuffle = False,
                                                 class_mode='categorical')

                                                 # Initializing the CNN classifier
classifier =  Sequential()

# Convolution Layer
classifier.add(Conv2D(32, (3,3), input_shape = (64, 64, 3), activation = 'relu'))

# Pooling
classifier.add(MaxPooling2D(pool_size= (2,2)))

# Flattening
classifier.add(Flatten())

# Fully Connected Dense Layers
classifier.add(Dense(units = 128, activation='relu'))    # Hidden Layer
classifier.add(Dense(units = 2, activation='softmax'))   # Output Layer

# Compiling the CNN model
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

import h5py
from tensorflow.keras.callbacks import ModelCheckpoint

# Creating best model with maximum accuracy
mc = ModelCheckpoint('Brain_Tumor_model.h5', monitor='accuracy', mode='max', verbose=1, patience=100, save_best_only=True)

# Fitting the CNN on the Data

history = classifier.fit(training_set,
                         steps_per_epoch=10,
                         epochs=50,
                         validation_data=test_set,
                         validation_steps=10,
                         callbacks=mc)

min(history.history['loss'])

# loading best saved model
from tensorflow.keras.models import load_model
saved_model =load_model('Brain_Tumor_model.h5')

'''Information of the Data when we use categorical labels with softmax'''
print('Number of Classes: ', training_set.num_classes)
print(training_set.class_indices)


no_index = np.where(training_set.labels == training_set.class_indices['no'])[0][0]
yes_index = np.where(training_set.labels == training_set.class_indices['yes'])[0][0]



no_batch_num = no_index//32
relative_index_of_first_no = no_batch_num % 32
no_label = training_set[no_batch_num][1][relative_index_of_first_no]


# finding the actual categorical label of our dogs
yes_batch_num = yes_index//32
relative_index_of_first_yes = yes_batch_num % 32
yes_label = training_set[yes_batch_num][1][relative_index_of_first_yes]

print("No = ", no_label)
print("Yes = ", yes_label)


result = saved_model.predict(test_set)
print(result)

result_label = []
for k in result:
  if(k[0] > k[1]):
    result_label.append('0')
  else:
    result_label.append('1')

print(result_label)

import csv

tsv_file = open("/content/Brain_Tumor_Detection_Dataset/test.tsv")
read_tsv = csv.reader(tsv_file, delimiter="\t")


X1 = []

for row in read_tsv:
  X1.append(row[1])


X = X1[1:]


tsv_file.close()



class_type = {'0':'No','1':'Yes'}

count  = 0

for index, doc in enumerate(X):

  print("Image{}  - Actual Label  - {} ({}) --> Prediction Label  - {} ({})".format(index+1,X[index],class_type[X[index]], result_label[index], class_type[result_label[index]] ))
  if(X[index] == result_label[index]):
    count+=1

percentage = (count/200)*100
print("\n\nAccurately Predicted:",percentage,"%")


