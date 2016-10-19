from __future__ import print_function
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split

# Keras stuff
from keras.preprocessing.image import img_to_array, load_img

LABELS = sorted(pd.read_csv('../train.csv').species.unique())

def load_numeric_training(standardize=True):
    # Read data from the CSV file
    data = pd.read_csv('../train.csv')
    ID = data.pop('id')

    ## Since the labels are textual, so we encode them categorically
    y = data.pop('species')
    y = LabelEncoder().fit(y).transform(y)
    # print('Labels Shape: {0}'.format(y.shape))

    X = StandardScaler().fit(data).transform(data) if standardize else data.values
    # print('Training data shape: {0}'.format(X.shape))

    return ID, X, y


def load_numeric_test(standardize=True):

    test = pd.read_csv('../test.csv')
    index = test.pop('id')
    test = StandardScaler().fit(test).transform(test)
    return index, test


def resize_img(img, max_dim=96):
    max_ax = max((0, 1), key=lambda i: img.size[i])
    scale = max_dim / float(img.size[max_ax])
    return img.resize((int(img.size[0] * scale), int(img.size[1] * scale)))


def load_image_data(ids, max_dim=96, center=True):

    rootdir = '../images'
    X = np.empty((len(ids), 1, max_dim, max_dim))
    for i, idee in enumerate(ids):
        x = resize_img(load_img(rootdir + '/' + str(idee) + '.jpg', grayscale=True), max_dim=max_dim)
        x = img_to_array(x)
        if center:
            h1 = (max_dim - x.shape[1]) / 2
            h2 = h1 + x.shape[1]
            w1 = (max_dim - x.shape[2]) / 2
            w2 = w1 + x.shape[2]
        else:
            h1, w1 = 0, 0
            h2, w2 = x.shape[1:]
        # Insert into image matrix
        X[i, 0, h1:h2, w1:w2] = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 1, height, width)
    return np.around(X / 255.0)


def load_train_data(split=.7, random_state=42):
    ID, X_num_tr, y = load_numeric_training()
    X_img_tr = load_image_data(ID)
    X_num_tr, X_num_val, X_img_tr, X_img_val, y_tr, y_val = train_test_split(X_num_tr, X_img_tr, y, train_size=split, random_state=random_state)
    return (X_num_tr, X_img_tr, y_tr), (X_num_val, X_img_val, y_val)


def load_test_data():
    ID, X_num_te = load_numeric_test()
    X_img_te = load_image_data(ID)
    return ID, X_num_te, X_img_te


def load_data():
    (X_num_tr, X_img_tr, y_tr), (X_num_val, X_img_val, y_val) = load_train_data()
    ids, X_num_te, X_img_te = load_test_data()
    return (X_num_tr, X_img_tr, y_tr), (X_num_val, X_img_val, y_val), (ids, X_num_te, X_img_te)


