from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np

## Keras Libraries for Neural Networks

from keras import backend as K
from keras.models import load_model
from keras.layers import MaxPooling2D
from math import sqrt


import leaf99

NUM_LEAVES = 50
# model1_fn = 'combined_model2.02-0.01.hdf5'
# model2_fn = '3combined_weights.06-0.00.hdf5'

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
# model = load_model('../models/' + model1_fn)   # The filename is not aptly named. This is an actual keras model
model = load_model('weights.06-0.00.hdf5')

# Get the convolutional layers
convouts = [layer for layer in model.layers if isinstance(layer, MaxPooling2D)]

# Make sure we grabbed the right layers
print(convouts)

# Load the data
(X_num_tr, X_img_tr, y_tr), (X_num_val, X_img_val, y_val) = leaf99.load_train_data()

# Pick random images to visualize

imgs_to_visualize = np.random.choice(np.arange(0, len(X_img_val)), NUM_LEAVES)

# Use a keras function to extract the conv layer data
convout_f = K.function([model.layers[0].input, K.learning_phase()], [layer.output for layer in convouts])
convolutions = convout_f([X_img_val[imgs_to_visualize], 0])
predictions = model.predict([X_img_val[imgs_to_visualize], X_num_val[imgs_to_visualize]])


imshow = plt.imshow #alias

# Show how many filters we're using per layer
for i, conv in enumerate(convolutions):
    print("The second dimension tells us how many convolutions do we have for layer %d: %s (%d convolutions)" % (
        i,
        str(conv.shape),
        conv.shape[1]))

for ind, img_to_visualize in enumerate(imgs_to_visualize):

    # Get top 3
    # print(predictions.shape)
    top3_ind = predictions[ind].argsort()[-3:]
    # print(top3_ind)

    top3_species = np.array(leaf99.LABELS)[top3_ind]
    top3_preds = predictions[ind][top3_ind]

    actual = leaf99.LABELS[y_val[img_to_visualize]]

    print("Top 3 Predicitons:")
    for i in xrange(2, -1, -1):
        print("\t%s: %s" % (top3_species[i], top3_preds[i]))
    print("\nActual: %s" % actual)

    # Show the original image
    plt.title("Image used: #%d (digit=%d)" % (img_to_visualize, y_val[img_to_visualize]))
    imshow(X_img_val[img_to_visualize][0], cmap='gray')
    plt.show()

    # Actually plot the filter images
    for i, convs in enumerate(convolutions):
        conv = convs[ind]
        # print(conv.shape)
        print("Visualizing Convolutions Layer %d" % i)
        fig_dict = {'flt{0}'.format(i): convolution for i, convolution in enumerate(conv)}
        plot_figures(fig_dict, *get_dim(len(fig_dict)))


