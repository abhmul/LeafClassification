from keras.layers import Dense,Dropout,Activation, Convolution2D, MaxPooling2D, Flatten, Input, merge, BatchNormalization
from keras.models import Model
from keras.backend import image_dim_ordering

input_shape = (96, 96, 1) if image_dim_ordering() == 'tf' else (1, 96, 96)

def combined_model():

    image = Input(shape=input_shape, name='image')
    x = BatchNormalization()(image)
    x = Convolution2D(20, 5, 5, border_mode='same')(x)
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

    image = Input(shape=input_shape, name='image')
    x = BatchNormalization()(image)
    x = Convolution2D(6, 5, 5, border_mode='same')(x)
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


def best_combined_model():

    # Define the image input
    image = Input(shape=input_shape, name='image')
    # Pass it through the first convolutional layer
    x = Convolution2D(8, 5, 5, border_mode='same')(image)
    x = (Activation('relu'))(x)
    x = (MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))(x)

    # Now through the second convolutional layer
    x = (Convolution2D(32, 5, 5, border_mode='same'))(x)
    x = (Activation('relu'))(x)
    x = (MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))(x)

    # Flatten our array
    x = Flatten()(x)
    # Define the pre-extracted feature input
    numerical = Input(shape=(192,), name='numerical')
    # Concatenate the output of our convnet with our pre-extracted feature input
    concatenated = merge([x, numerical], mode='concat')

    # Add a fully connected layer just like in a normal MLP
    x = Dense(100, activation='relu')(concatenated)
    x = Dropout(.5)(x)

    # Get the final output
    out = Dense(99, activation='softmax')(x)
    # How we create models with the Functional API
    model = Model(input=[image, numerical], output=out)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    return model


def combined_generator(imgen, X, test=False):
    """
    A generator to train our keras neural network. It
    takes the image augmenter generator and the array
    of the pre-extracted features.
    It yields a minibatch and will run indefinitely
    """
    while True:
        for i in range(X.shape[0]):
            if test:
                batch_img = next(imgen)
            else:
                # Get the image batch and labels
                batch_img, batch_y = next(imgen)
            # This is where that change to the source code we
            # made will come in handy. We can now access the indicies
            # of the images that imgen gave us.
            x = X[imgen.index_array]
            if test:
                yield [batch_img, x]
            else:
                yield [batch_img, x], batch_y