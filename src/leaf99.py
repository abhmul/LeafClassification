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
    """
    Loads the pre-extracted features for the training data
    and returns a tuple of the image ids, the data, and the labels
    """
    # Read data from the CSV file
    data = pd.read_csv('../train.csv')
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
    test = pd.read_csv('../test.csv')
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
    rootdir = '../images'
    # Initialize the output array
    # NOTE: If using tensorflow, shape should be (len(ids), max_dim, max_dim, 1)
    X = np.empty((len(ids), 1, max_dim, max_dim))
    for i, idee in enumerate(ids):
        # Turn the image into an array
        x = resize_img(load_img(rootdir + '/' + str(idee) + '.jpg', grayscale=True), max_dim=max_dim)
        x = img_to_array(x)
        # Get the corners of the bounding box for the image
        if center:
            h1 = (max_dim - x.shape[1]) / 2
            h2 = h1 + x.shape[1]
            w1 = (max_dim - x.shape[2]) / 2
            w2 = w1 + x.shape[2]
        else:
            h1, w1 = 0, 0
            h2, w2 = x.shape[1:]
        # Insert into image matrix
        # NOTE: If using tensorflow, X should be indexed with [i, h1:h2, w1:w1, 0]
        X[i, 0, h1:h2, w1:w2] = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 1, height, width)
    # Scale the array values so they are between 0 and 1
    return np.around(X / 255.0)


def load_train_data(split=.9, random_state=None):
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


