import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard

import pickle
import time

NAME = "Cats-vs-dogs-CNN-64x2-{}".format(int(time.time()))

tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle", "rb")
y = pickle.load(pickle_in)

X = X/255.0

model = Sequential()

model.add(Conv2D(256, (3, 3), input_shape=X.shape[1:]))  # bez pierwszej wartości bo to -1 że niby wszystkie metadane
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

model.add(Dense(64))
model.add(Activation('relu'))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X, y, batch_size=32, epochs=3, validation_split=0.3, callbacks=[tensorboard])

"""
After having run this, you should have a new directory called logs. We can visualize the
initial results from this directory using tensorboard now. Open a console, change to your
working directory, and type: tensorboard --logdir=logs/. You should see a notice like: 
TensorBoard 1.10.0 at http://H-PC:6006 (Press CTRL+C to quit) where "h-pc" probably is 
whatever your machine's name is. Open a browser and head to this address.
"""