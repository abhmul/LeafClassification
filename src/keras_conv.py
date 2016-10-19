from __future__ import print_function

import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

## Importing sklearn libraries

from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss

## Keras Libraries for Neural Networks

from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation, Convolution2D, MaxPooling2D, Flatten
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

import leaf99

seed = 7
np.random.seed(seed)

(X_num_tr, X_img_tr, y_tr), (X_num_val, X_img_val, y_val) = leaf99.load_train_data()

y_tr = to_categorical(y_tr)
print(y_tr.shape)

y_val = to_categorical(y_val)
print(y_tr.shape)

print(np.max(X_img_tr), np.max(X_img_val))


imgen = ImageDataGenerator(
    # rescale=1./255,
    rotation_range=90,
    width_shift_range=0.2,
    height_shift_range=0.2,
    # shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest')
imgen_train = imgen.flow(X_img_tr, y_tr)#, save_to_dir='../preview', save_prefix='leaf_train', save_format='jpg')

def conv_model():
    model = Sequential()
    model.add(Convolution2D(20, 5, 5, input_shape=(1, 96, 96), border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))

    model.add(Convolution2D(50, 5, 5, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(85))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(99))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    return model

def conv_model2():
    model = Sequential()
    model.add(Convolution2D(20, 5, 5, input_shape=(1, 96, 96), border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(5, 5), strides=(5,5)))
    model.add(Convolution2D(40, 5, 5, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(5, 5), strides=(5,5)))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    # model.add(Dropout(0.5))
    model.add(Dense(85))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(99))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    return model

def baseline_model():

    model = Sequential()
    model.add(Dense(85, input_dim=96*96, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(99))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    return model

model = conv_model()
history = model.fit_generator(imgen_train, samples_per_epoch=693, nb_epoch=10000, validation_data=(X_img_val, y_val), nb_val_samples=297, verbose=1)
# history = model.fit(X_img_tr, y_tr, nb_epoch=1000, validation_data=(X_img_val, y_val))

index, test, X_img_te = leaf99.load_test_data()

yPred_proba = model.predict_proba(test)

## Converting the test predictions in a dataframe as depicted by sample submission

yPred = pd.DataFrame(yPred_proba,index=index,columns=leaf99.LABELS)

fp = open('submission_nn_10_19-5.csv','w')
fp.write(yPred.to_csv())

yPred_r = np.around(yPred_proba)

yPred = pd.DataFrame(yPred_r,index=index,columns=leaf99.LABELS)

fp = open('submission_nn_10_19-5_ceil.csv','w')
fp.write(yPred.to_csv())


plt.plot(history.history['val_loss'],'o-')
plt.plot(history.history['loss'],'o-')
plt.xlabel('Number of Iterations')
plt.ylabel('Categorical Crossentropy')
plt.title('Train Error vs Number of Iterations')

plt.show()




