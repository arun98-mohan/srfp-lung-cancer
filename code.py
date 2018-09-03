print("Importing Libraries")
import glob
import numpy as np
import os.path as path
import cv2
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from sklearn.metrics import accuracy_score, f1_score
#from datetime import datetime
IMAGE_PATH='/Users/Arun/Desktop/temp'
file_paths = glob.glob(path.join(IMAGE_PATH, '*.png'))
file_paths.sort()
print("First Image's Path: ")
print(file_paths[0])
print("Reading All Images")
images = [cv2.imread(path) for path in file_paths]
print("Converting All to arrays using asarray")
images = np.asarray(images)
image_size = np.asarray([images.shape[1], images.shape[2], images.shape[3]])
print(image_size)
images = images / 255
n_images = images.shape[0]
print("###Importing Labels from CSV file")
labels = np.zeros(n_images)
import csv
file=open("/Users/Arun/Desktop/lbls.csv","r")
reader=csv.reader(file)
i=0
for line in reader:
 labels[i]=line[0]
 i=i+1
print("Checking a few labels")
print("Image 0(False): ")
print(labels[0])
print("Image 12(True): ")
print(labels[12])
for i in range(n_images):
 labels[i]=int(labels[i])
print("Splitting Training Data and Testing Data with 90% and 10% of total")
TRAIN_TEST_SPLIT = 0.9
split_index = int(TRAIN_TEST_SPLIT * n_images)
shuffled_indices = np.random.permutation(n_images)
train_indices = shuffled_indices[0:split_index]
test_indices = shuffled_indices[split_index:]
x_train = images[train_indices, :, :]
y_train = labels[train_indices]
x_test = images[test_indices, :, :]
y_test = labels[test_indices]
print("Trying to Visualize a few Images")
import matplotlib.pyplot as plt
def visualize_data(positive_images, negative_images):
 figure=plt.figure()
 count=0
 for i in range(positive_images.shape[0]):
  count+=1
  figure.add_subplot(2,positive_images.shape[0],count)
  plt.imshow(positive_images[i,:,:])
  plt.axis('off')
  plt.title("1")
  figure.add_subplot(1,negative_images.shape[0],count)
  plt.imshow(negative_images[i,:,:])
  plt.axis('off')
  plt.title("0")
 plt.show()
N_TO_VISUALIZE=3
positive_example_indices=(y_train==1)
positive_examples=x_train[positive_example_indices,:,:]
positive_examples=positive_examples[0:N_TO_VISUALIZE,:,:]
negative_example_indices = (y_train == 0)
negative_examples = x_train[negative_example_indices, :, :]
negative_examples = negative_examples[0:N_TO_VISUALIZE, :, :]
visualize_data(positive_examples, negative_examples)
N_LAYERS=8
def cnn(size,n_layers):
 MIN_NEURONS=10
 MAX_NEURONS=50
 KERNEL=(3,3)
 steps=np.floor(MAX_NEURONS/(n_layers+1))
 neurons=np.arange(MIN_NEURONS,MAX_NEURONS,steps)
 neurons=neurons.astype(np.int32)
 model=Sequential()
 for i in range(0,n_layers):
  if i==0:
   shape=(size[0],size[1],size[2])
   model.add(Conv2D(neurons[i],KERNEL,input_shape=shape))
  else:
   model.add(Conv2D(neurons[i],KERNEL))
  model.add(Activation('relu'))
 model.add(MaxPooling2D(pool_size=(2,2)))
 model.add(Flatten())
 model.add(Dense(MAX_NEURONS))
 model.add(Activation('relu'))
 model.add(Dense(1))
 model.add(Activation('sigmoid'))
 model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
 model.summary()
 return model
model=cnn(size=image_size,n_layers=N_LAYERS)
EPOCHS=5
BATCH_SIZE=10
model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
test_predictions=model.predict(x_test)
test_predictions=np.round(test_predictions)
accuracy=accuracy_score(y_test,test_predictions)
print("Accuracy: "+str(accuracy*100)+"%")
