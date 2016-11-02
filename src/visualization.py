from __future__ import print_function

import matplotlib.pyplot as plt
from keras import backend as K

## Keras Libraries for Neural Networks

from keras.models import load_model
from keras.layers import MaxPooling2D
from random import randint
from math import sqrt


import leaf99

# Function by gcalmettes from http://stackoverflow.com/questions/11159436/multiple-figures-in-a-single-window
def plot_figures(figures, nrows = 1, ncols=1):
    """Plot a dictionary of figures.

    Parameters
    ----------
    figures : <title, figure> dictionary
    ncols : number of columns of subplots wanted in the display
    nrows : number of rows of subplots wanted in the figure
    """

    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows)
    for ind,title in enumerate(sorted(figures.iterkeys(), key=lambda s: int(s[3:]))):
        axeslist.ravel()[ind].imshow(figures[title], cmap=plt.gray())
        axeslist.ravel()[ind].set_title(title)

    for ind in xrange(nrows*ncols):
        axeslist.ravel()[ind].set_axis_off()

    mng = plt.get_current_fig_manager()
    mng.window.maxsize()
    plt.show()


def get_dim(num):
    """
    Simple function to get the dimensions of a square-ish shape for plotting
    num images

    :param num: number of images or plots to plot
    :return: A tuple with the nb rows and nb cols
    """

    s = sqrt(num)
    if round(s) < s:
        return (int(s), int(s)+1)
    else:
        return (int(s)+1, int(s)+1)

# Load my best model
model = load_model('../models/3combined_weights.06-0.00.hdf5')   # The filename is not aptly named. This is an actual keras model

# Get the convolutional layers
convouts = [layer for layer in model.layers if isinstance(layer, MaxPooling2D)]

# Make sure we grabbed the right layers
print(convouts)

# Load the data
(X_num_tr, X_img_tr, y_tr), (X_num_val, X_img_val, y_val) = leaf99.load_train_data()

# Pick a random image to visualize
img_to_visualize = randint(0, len(X_img_val) - 1)

# Use a theano function to extract the conv layer data
convout_f = K.function([model.layers[0].input, K.learning_phase()], [layer.output for layer in convouts])
convolutions = convout_f([X_img_val[img_to_visualize: img_to_visualize+1], 0])

imshow = plt.imshow #alias
# Show the original image
plt.title("Image used: #%d (digit=%d)" % (img_to_visualize, y_val[img_to_visualize]))
imshow(X_img_val[img_to_visualize][0])
plt.show()

# Show how many filters we're using per layer
for i, conv in enumerate(convolutions):
    print("The second dimension tells us how many convolutions do we have for layer %d: %s (%d convolutions)" % (
        i,
        str(conv.shape),
        conv.shape[1]))

# Actually plot the filter images
for i, conv in enumerate(convolutions):
    print("Visualizing Convolutions Layer %d" % i)
    fig_dict = {'flt{0}'.format(i): convolution for i, convolution in enumerate(conv[0])}
    plot_figures(fig_dict, *get_dim(len(fig_dict)))