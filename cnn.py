print("sairam")

import tensorflow as tf
print(tf.__version__)

from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Sequential

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import numpy as np


train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)
test_datagen  = ImageDataGenerator(rescale=1./255, horizontal_flip=True)


training_set = train_datagen.flow_from_directory("E:\OneNeuron\Computer Vision_Sudhanshu\Image Classification\data\Train",
                                                target_size=(64, 64),
                                                batch_size=32
                                                )

test_set = test_datagen.flow_from_directory("E:\OneNeuron\Computer Vision_Sudhanshu\Image Classification\data\Valid",
                                                target_size=(64, 64),
                                                batch_size=32
                                                )

classifier = Sequential()

# Trying to follow the sequence described in https://poloclub.github.io/cnn-explainer/
classifier.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), activation='relu', input_shape=(64, 64, 3)))
classifier.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
classifier.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Flatten())

classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=6, activation='softmax'))


print(classifier.summary())

classifier.compile(metrics=['accuracy'], loss='categorical_crossentropy', optimizer='adam')
model = classifier.fit_generator(training_set, epochs=10, validation_data=test_set, steps_per_epoch=16)

classifier.save('chess_piece_classifier.h5')