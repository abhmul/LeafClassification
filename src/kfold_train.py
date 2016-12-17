from __future__ import print_function

import datetime

import numpy as np
import pandas as pd


## Keras Libraries for Neural Networks

from keras.models import load_model
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint

import leaf99
from keras_utils import ImageDataGenerator2
from models import best_combined_model, combined_generator

np.random.seed(7)
split_random_state = 4567
kfold = True
n_folds = 6
augment = True
nbr_aug = 10
stratified = True

kf, (X_num_tr, X_img_tr, y_tr) = leaf99.load_train_data_kfold(n_folds=n_folds, random_state=split_random_state, stratified=True)

y_tr_cat = to_categorical(y_tr)
assert(y_tr_cat.shape[1] == len(leaf99.LABELS))

print('Initializing Data Augmenter...')
imgen = ImageDataGenerator2(
    rotation_range=10,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest')
print('Data Augmenter Initialized.')

histories = []

# Start the kfold
for i, (train_index, test_index) in enumerate(kf):

    # Get the fold dataset
    X_num_tr_fold = X_num_tr[train_index]
    X_img_tr_fold = X_img_tr[train_index]
    y_tr_cat_fold = y_tr_cat[train_index]
    X_num_val_fold = X_num_tr[test_index]
    X_img_val_fold = X_img_tr[test_index]
    y_val_cat_fold = y_tr_cat[test_index]

    # Flow the augmenter
    fold_imgen = imgen.flow(X_img_tr_fold, y_tr_cat_fold)

    print('Starting KFold number {}/{}'.format(i+1, n_folds))
    print('Split train: ', len(X_num_tr_fold), len(X_img_tr_fold), len(y_tr_cat_fold))
    print('Split valid: ', len(X_num_val_fold), len(X_img_val_fold), len(y_val_cat_fold))

    print('Creating the model...')
    model = best_combined_model()
    print('Model created!')

    # autosave best Model
    best_model_file = "../models/leafnet_v1.1_fold{}.h5".format(i+1)
    best_model = ModelCheckpoint(best_model_file, monitor='val_loss', verbose=1, save_best_only=True)

    print('Training model...')
    histories.append(model.fit_generator(combined_generator(fold_imgen, X_num_tr_fold),
                                         samples_per_epoch=X_num_tr_fold.shape[0],
                                         nb_epoch=100,
                                         validation_data=([X_img_val_fold, X_num_val_fold], y_val_cat_fold),
                                         nb_val_samples=X_num_val_fold.shape[0],
                                         verbose=1,
                                         callbacks=[best_model]))

# Now for the submission
# Clear some memory
kf, X_num_tr, X_img_tr, y_tr = (None,)*4
yPred_proba = None
ID, X_num_te, X_img_te = leaf99.load_test_data()
for i in range(n_folds):

    best_model_file = "../models/leafnet_v1.1_fold{}.h5".format(i + 1)
    print('Loading the best model fold {}/{}...'.format(i+1, n_folds))
    model = load_model(best_model_file)
    print('Best Model loaded!')

    for j in range(nbr_aug):

        imgen_te = imgen.flow(X_img_te, y=np.zeros(X_img_te.shape[0:1]), shuffle=False)

        if yPred_proba is None:
            yPred_proba = model.predict_generator(combined_generator(imgen_te, X_num_te, test=True),
                                                  X_num_te.shape[0])
        else:
            yPred_proba += model.predict_generator(combined_generator(imgen_te, X_num_te, test=True),
                                                   X_num_te.shape[0])


yPred_proba /= float(n_folds * nbr_aug)

print('Writing submission...')
## Converting the test predictions in a dataframe as depicted by sample submission
yPred = pd.DataFrame(yPred_proba, index=ID, columns=leaf99.LABELS)
now = datetime.datetime.now()
fp = open('../submissions/submission_{}fold_{}.csv'.format(n_folds, str(now.strftime("%Y-%m-%d-%H-%M"))), 'w')
fp.write(yPred.to_csv())
print('Finished writing submission!')
