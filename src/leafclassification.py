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
rootdir = '../images'

## Set figure size to 20x10

from pylab import rcParams
rcParams['figure.figsize'] = 10,10

(X_num_tr, X_img_tr, y_tr), (X_num_val, X_img_val, y_val) = leaf99.load_train_data()

## We will be working with categorical crossentropy function
## It is required to further convert the labels into "one-hot" representation

y_tr = to_categorical(y_tr)
print(y_tr.shape)

## Developing a layered model for Neural Networks
## Input dimensions should be equal to the number of features
## We used softmax layer to predict a uniform probabilistic distribution of outcomes

def baseline_model():

    model = Sequential()
    model.add(Dense(85,input_dim=192, activation='relu'))
    model.add(Dropout(0.5))
    # model.add(Dense(70, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(99, activation='softmax'))

    ## Error is measured as categorical crossentropy or multiclass logloss
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model

tests = 5
models = []

model = baseline_model()

## Fitting the model on the whole training data
history = model.fit(X_num_tr, y_tr, batch_size=128, nb_epoch=430,verbose=0, validation_data=(X_num_val,y_val) , shuffle=True)
models.append(model)

## Plotting the error with the number of iterations
## With each iteration the error reduces smoothly

plt.plot(history.history['val_loss'],'o-',)
plt.plot(history.history['loss'],'o-')
plt.xlabel('Number of Iterations')
plt.ylabel('Categorical Crossentropy')
plt.title('Train Error vs Number of Iterations')

index, test, X_img_te = leaf99.load_test_data()

yPred = model.predict_proba(test)

## Converting the test predictions in a dataframe as depicted by sample submission

yPred = pd.DataFrame(yPred,index=index,columns=leaf99.LABELS)

fp = open('../submissions/submission_nn_10_16-5.csv','w')
fp.write(yPred.to_csv())

plt.show()
