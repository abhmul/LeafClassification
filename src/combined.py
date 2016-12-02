from __future__ import print_function

import os

import numpy as np
import pandas as pd


## Keras Libraries for Neural Networks

from keras.models import Sequential, Model, load_model
from keras.layers import Dense,Dropout,Activation, Convolution2D, MaxPooling2D, Flatten, Input, merge, BatchNormalization
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img, NumpyArrayIterator
from keras.callbacks import ModelCheckpoint

import leaf99

seed = 7
np.random.seed(seed)


class ImageDataGenerator2(ImageDataGenerator):
    def flow(self, X, y=None, batch_size=32, shuffle=True, seed=None,
             save_to_dir=None, save_prefix='', save_format='jpeg'):
        return NumpyArrayIterator2(
            X, y, self,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            dim_ordering=self.dim_ordering,
            save_to_dir=save_to_dir, save_prefix=save_prefix, save_format=save_format)


class NumpyArrayIterator2(NumpyArrayIterator):
    def next(self):
        # for python 2.x.
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch
        # see http://anandology.com/blog/using-iterators-and-generators/
        with self.lock:
            self.index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock so it can be done in parallel
        batch_x = np.zeros(tuple([current_batch_size] + list(self.X.shape)[1:]))
        for i, j in enumerate(self.index_array):
            x = self.X[j]
            x = self.image_data_generator.random_transform(x.astype('float32'))
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x
        if self.save_to_dir:
            for i in range(current_batch_size):
                img = array_to_img(batch_x[i], self.dim_ordering, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=current_index + i,
                                                                  hash=np.random.randint(1e4),
                                                                  format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        if self.y is None:
            return batch_x
        batch_y = self.y[self.index_array]
        return batch_x, batch_y

(X_num_tr, X_img_tr, y_tr), (X_num_val, X_img_val, y_val) = leaf99.load_train_data()

y_tr = to_categorical(y_tr)
print(y_tr.shape)

y_val = to_categorical(y_val)
print(y_tr.shape)

print(np.max(X_img_tr), np.max(X_img_val))


imgen = ImageDataGenerator2(
    # rescale=1./255,
    rotation_range=20,
    # width_shift_range=0.2,
    # height_shift_range=0.2,
    # shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest')
imgen_train = imgen.flow(X_img_tr, y_tr)


def combined_model():

    image = Input(shape=(1, 96, 96), name='image')
    x = BatchNormalization()(image)
    x = Convolution2D(20, 5, 5, input_shape=(1, 96, 96), border_mode='same')(x)
    x = (Activation('relu'))(x)
    x = (MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))(x)

    x = (Convolution2D(50, 5, 5, border_mode='same'))(x)
    x = (Activation('relu'))(x)
    x = (MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))(x)

    x = Flatten()(x)
    numerical = Input(shape=(192,), name='numerical')
    concatenated = merge([x, numerical], mode='concat')

    x = Dense(100, activation='relu')(concatenated)
    x = Dropout(.5)(x)

    out = Dense(99, activation='softmax')(x)

    model = Model(input=[image, numerical], output=out)

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    return model


def deeper_combined():

    image = Input(shape=(1, 96, 96), name='image')
    x = BatchNormalization()(image)
    x = Convolution2D(6, 5, 5, input_shape=(1, 96, 96), border_mode='same')(x)
    x = (Activation('relu'))(x)
    x = (MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))(x)

    x = (Convolution2D(16, 5, 5, border_mode='same'))(x)
    x = (Activation('relu'))(x)
    x = (MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))(x)

    x = (Convolution2D(16, 5, 5, border_mode='same'))(x)
    x = (Activation('relu'))(x)
    x = (MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))(x)
    #
    x = (Convolution2D(32, 5, 5, border_mode='same'))(x)
    x = (Activation('relu'))(x)
    x = (MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))(x)

    x = Flatten()(x)
    numerical = Input(shape=(192,), name='numerical')
    concatenated = merge([x, numerical], mode='concat')

    x = Dense(100, activation='relu')(concatenated)
    x = Dropout(.5)(x)

    out = Dense(99, activation='softmax')(x)

    model = Model(input=[image, numerical], output=out)

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    return model


def combined_generator(imgen, X):
    while True:
        for i in xrange(X.shape[0]):
            batch_img, batch_y = next(imgen)
            x = X[imgen.index_array]
            # print(batch_y.dot(np.arange(batch_y.shape[1])))
            # print(y_tr.dot(np.arange(batch_y.shape[1]))[imgen.index_array])
            yield [batch_img, x], batch_y


model = deeper_combined()
# model = load_model('../models/deep_combined_model.0106-0.00.hdf5')
history = model.fit_generator(combined_generator(imgen_train, X_num_tr),
                              samples_per_epoch=50*891,
                              nb_epoch=10,
                              validation_data=([X_img_val, X_num_val], y_val),
                              nb_val_samples=99,
                              verbose=1,
                              callbacks=[ModelCheckpoint('../models/deep_combined_model.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto')])


index, test, X_img_te = leaf99.load_test_data()

yPred_proba = model.predict([X_img_te, test])

## Converting the test predictions in a dataframe as depicted by sample submission

yPred = pd.DataFrame(yPred_proba,index=index,columns=leaf99.LABELS)

fp = open('../submissions/submission_nn_11_2-5.csv','w')
fp.write(yPred.to_csv())

yPred_r = (yPred_proba >= np.vstack([np.max(yPred_proba, axis=1)]*yPred_proba.shape[1]).T).astype(float)

yPred = pd.DataFrame(yPred_r,index=index,columns=leaf99.LABELS)

fp = open('../submissions/submission_nn_11_2-5_ceil.csv','w')
fp.write(yPred.to_csv())

# plt.plot(history.history['val_loss'],'o-')
# plt.plot(history.history['loss'],'o-')
# plt.xlabel('Number of Iterations')
# plt.ylabel('Categorical Crossentropy')
# plt.title('Train Error vs Number of Iterations')
#
# plt.show()