##### SET UP #####
import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
import pathlib
from tensorflow.keras import layers
import matplotlib.pyplot as plt

##### download image #####

path_test = pathlib.Path(r'data\test\test')

path_train = pathlib.Path(r'data\train')

batch_size = 32
img_height = 64
img_width = 64

##### create training and validation data #####
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    path_train,
    validation_split = 0.2,
    subset = 'training',
    seed = 123,
    color_mode = 'grayscale',
    image_size = (img_height, img_width),
    batch_size = batch_size
) 

class_names = [0,1,2,3,4,5]

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    path_train,
    validation_split = 0.2,
    subset = 'validation',
    seed = 123,
    color_mode = 'grayscale', #default
    image_size = (img_height, img_width),
    batch_size = batch_size
) 


##### construct model #####

model = tf.keras.Sequential([ 
  layers.experimental.preprocessing.Rescaling(1./255), 
  layers.Conv2D(16, (2,2), input_shape=train_ds.shape[1:],activation='relu'),
  layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
  layers.Conv2D(32, (3,3), activation='relu'),
  layers.MaxPooling2D(pool_size=(3, 3), strides=(3, 3), padding='same'),
  layers.Conv2D(64, (5,5), activation='relu'),
  layers.MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding='same'),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dropout(0.2),
  layers.Dense(6, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=10,
  batch_size = 32,
  verbose = 2,
)

model.summary()

##### graph #####
epochs = 10
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

##### load test data #####
count = 0
ans = list()
for x in sorted(path_test.iterdir()):
  img = tf.keras.preprocessing.image.load_img(x, color_mode="grayscale",target_size=(img_height, img_width))
  img_array = tf.keras.preprocessing.image.img_to_array(img)
  img_array = tf.expand_dims(img_array, 0)
  img_array = tf.expand_dims(img_array, -1)
  predictions = model.predict(img_array)
  score = tf.nn.softmax(predictions[0])
  ans.append(class_names[np.argmax(score)])
  count += 1
  print(x," is predicted to be number ",np.argmax(score),end = '\n')