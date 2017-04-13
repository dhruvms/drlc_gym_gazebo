import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Reshape, Flatten, Lambda
from keras.layers.convolutional import Convolution2D, ZeroPadding2D, AveragePooling2D, MaxPooling2D

model = Sequential()
model.add(Convolution2D(filters=16, kernel_size=(8,8), strides=4, input_shape=(84,84,4), activation='relu'))
