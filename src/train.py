from __future__ import print_function

import os

import numpy as np
import pandas as pd


## Keras Libraries for Neural Networks

from keras.models import load_model

from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint

import leaf99
from keras_utils import ImageDataGenerator2
from models import best_combined_model, combined_model, deeper_combined, combined_generator

np.random.seed(7)
split_random_state = 4567
split = .9

(X_num_tr, X_img_tr, y_tr), (X_num_val, X_img_val, y_val) = leaf99.load_train_data(split=split, random_state=split_random_state)

y_tr_cat = to_categorical(y_tr)
print(y_tr_cat.shape)

y_val_cat = to_categorical(y_val)
print(y_tr_cat.shape)

print(np.max(X_img_tr), np.max(X_img_val))

print('Creating Data Augmenter...')
imgen = ImageDataGenerator2(
    rotation_range=10,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest')
imgen_train = imgen.flow(X_img_tr, y_tr_cat)
print('Finished making data augmenter...')

print('Creating the model...')
model = combined_model()
print('Model created!')

# autosave best Model
best_model_file = "../models/leafnet_v1.h5"
best_model = ModelCheckpoint(best_model_file, monitor='val_loss', verbose=1, save_best_only=True)

print('Training model...')
history = model.fit_generator(combined_generator(imgen_train, X_num_tr),
                              samples_per_epoch=X_num_tr.shape[0],
                              nb_epoch=100,
                              validation_data=([X_img_val, X_num_val], y_val_cat),
                              nb_val_samples=X_num_val.shape[0],
                              verbose=1,
                              callbacks=[best_model])

print('Loading the best model...')
model = load_model(best_model_file)
print('Best Model loaded!')

index, test, X_img_te = leaf99.load_test_data()

yPred_proba = model.predict([X_img_te, test])

## Converting the test predictions in a dataframe as depicted by sample submission

yPred = pd.DataFrame(yPred_proba,index=index,columns=leaf99.LABELS)

fp = open('../submissions/submission_nn_11_2-5.csv','w')
fp.write(yPred.to_csv())

# yPred_r = (yPred_proba >= np.vstack([np.max(yPred_proba, axis=1)]*yPred_proba.shape[1]).T).astype(float)
#
# yPred = pd.DataFrame(yPred_r,index=index,columns=leaf99.LABELS)
#
# fp = open('../submissions/submission_nn_11_2-5_ceil.csv','w')
# fp.write(yPred.to_csv())

# plt.plot(history.history['val_loss'],'o-')
# plt.plot(history.history['loss'],'o-')
# plt.xlabel('Number of Iterations')
# plt.ylabel('Categorical Crossentropy')
# plt.title('Train Error vs Number of Iterations')
#
# plt.show()