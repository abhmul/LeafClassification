from __future__ import print_function
import os

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold, StratifiedKFold

# Keras stuff
from keras.preprocessing.image import img_to_array, load_img
from keras.backend import image_dim_ordering


root = '../'
LABELS = sorted(pd.read_csv(os.path.join(root, 'train.csv')).species.unique())



def load_numeric_training(standardize=True):
    """
    Loads the pre-extracted features for the training data
    and returns a tuple of the image ids, the data, and the labels
    """
    # Read data from the CSV file
    data = pd.read_csv(os.path.join(root, 'train.csv'))
    ID = data.pop('id')

    # Since the labels are textual, so we encode them categorically
    y = data.pop('species')
    y = LabelEncoder().fit(y).transform(y)
    # standardize the data by setting the mean to 0 and std to 1
    X = StandardScaler().fit(data).transform(data) if standardize else data.values

    return ID, X, y


def load_numeric_test(standardize=True):
    """
    Loads the pre-extracted features for the test data
    and returns a tuple of the image ids, the data
    """
    test = pd.read_csv(os.path.join(root, 'test.csv'))
    ID = test.pop('id')
    # standardize the data by setting the mean to 0 and std to 1
    test = StandardScaler().fit(test).transform(test) if standardize else test.values
    return ID, test


def resize_img(img, max_dim=96):
    """
    Resize the image to so the maximum side is of size max_dim
    Returns a new image of the right size
    """
    # Get the axis with the larger dimension
    max_ax = max((0, 1), key=lambda i: img.size[i])
    # Scale both axes so the image's largest dimension is max_dim
    scale = max_dim / float(img.size[max_ax])
    return img.resize((int(img.size[0] * scale), int(img.size[1] * scale)))


def load_image_data(ids, max_dim=96, center=True):
    """
    Takes as input an array of image ids and loads the images as numpy
    arrays with the images resized so the longest side is max-dim length.
    If center is True, then will place the image in the center of
    the output array, otherwise it will be placed at the top-left corner.
    """
    # Initialize the output array
    if image_dim_ordering() == 'tf':
        X = np.empty((len(ids), max_dim, max_dim, 1))
    elif image_dim_ordering() == 'th':
        X = np.empty((len(ids), 1, max_dim, max_dim))
    for i, idee in enumerate(ids):
        # Turn the image into an array
        x = resize_img(load_img(os.path.join(root, 'images', str(idee) + '.jpg'), grayscale=True), max_dim=max_dim)
        x = img_to_array(x)
        # Get the corners of the bounding box for the image
        if image_dim_ordering() == 'tf':
            length = x.shape[0]
            width = x.shape[1]
        elif image_dim_ordering() == 'th':
            length = x.shape[1]
            width = x.shape[2]
        if center:
            h1 = int((max_dim - length) / 2)
            h2 = h1 + length
            w1 = int((max_dim - width) / 2)
            w2 = w1 + width
        else:
            h1, w1 = 0, 0
            h2, w2 = (length, width)
        # Insert into image matrix
        if image_dim_ordering() == 'tf':
            X[i, h1:h2, w1:w2, 0:1] = x
        elif image_dim_ordering() == 'th':
            X[i, 0:1, h1:h2, w1:w2] = x
    # Scale the array values so they are between 0 and 1
    return np.around(X / 255.0)


def load_train_data(split=None, random_state=None):
    """
    Loads the pre-extracted feature and image training data and
    splits them into training and cross-validation.
    Returns one tuple for the training data and one for the validation
    data. Each tuple is in the order pre-extracted features, images,
    and labels.
    """
    # Load the pre-extracted features
    ID, X_num_tr, y = load_numeric_training()
    # Load the image data
    X_img_tr = load_image_data(ID)
    # Split them into validation and cross-validation
    X_num_tr, X_num_val, X_img_tr, X_img_val, y_tr, y_val = train_test_split(X_num_tr, X_img_tr, y, train_size=split, random_state=random_state)
    return (X_num_tr, X_img_tr, y_tr), (X_num_val, X_img_val, y_val)


def load_train_data_kfold(n_folds=3, random_state=None, stratified=False):

    # Load the pre-extracted features
    ID, X_num_tr, y = load_numeric_training()
    # Load the image data
    X_img_tr = load_image_data(ID)
    if stratified:
        kf = StratifiedKFold(y, n_folds=n_folds, shuffle=True, random_state=random_state)
    else:
        kf = KFold(len(ID), n_folds=n_folds, shuffle=True, random_state=random_state)
    return kf, (X_num_tr, X_img_tr, y)


def load_test_data():
    """
    Loads the pre-extracted feature and image test data.
    Returns a tuple in the order ids, pre-extracted features,
    and images.
    """
    # Load the pre-extracted features
    ID, X_num_te = load_numeric_test()
    # Load the image data
    X_img_te = load_image_data(ID)
    return ID, X_num_te, X_img_te


def load_data(kfold=False):
    (X_num_tr, X_img_tr, y_tr), (X_num_val, X_img_val, y_val) = load_train_data()
    ids, X_num_te, X_img_te = load_test_data()
    return (X_num_tr, X_img_tr, y_tr), (X_num_val, X_img_val, y_val), (ids, X_num_te, X_img_te)


